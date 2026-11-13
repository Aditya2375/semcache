/**
 * Deterministic dataset assembly.
 *
 * Reads the hand-authored seed files in data/helpdesk/ (canonical Q&A,
 * paraphrases, near-miss traps, guard queries) and emits:
 *   - data/helpdesk/workload.json  (labeled query stream for replay)
 *   - data/helpdesk/pairs.json     (labeled intent pairs for calibration)
 *
 * Ordering constraints that make the workload honest:
 *   - a canonical's first ask precedes its repeats and paraphrases (nothing
 *     to hit before the first store)
 *   - a near-miss query appears only after every canonical it forbids has
 *     been asked at least once (the trap must actually be armed)
 * Everything else is a seeded interleave, so output is byte-stable.
 */

import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const read = (p) => JSON.parse(readFileSync(join(root, p), 'utf8'));

const canonicals = read('data/helpdesk/canonical.json').canonicals;
const paraphrases = read('data/helpdesk/paraphrases.json').paraphrases;
const nearmiss = read('data/helpdesk/nearmiss.json').queries;
const guards = read('data/helpdesk/guards.json').queries;

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const rng = mulberry32(20261113);
const shuffle = (arr) => {
  const a = [...arr];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
};

// --- per-canonical event groups -------------------------------------------
const paraByCanonical = new Map(paraphrases.map((p) => [p.canonicalId, p.texts]));
const groups = new Map(); // canonicalId -> array of query records in valid order
for (const c of canonicals) {
  const paras = paraByCanonical.get(c.id) ?? [];
  const tail = shuffle([
    ...paras.map((t) => ({ text: t, expect: 'hit-semantic', canonicalId: c.id })),
    { text: c.question, expect: 'hit-exact' },
  ]);
  groups.set(c.id, [{ text: c.question, expect: 'miss', answerCanonicalId: c.id, note: 'first ask' }, ...tail]);
}

// --- interleave group streams round-robin in seeded group order -----------
const groupOrder = shuffle([...groups.keys()]);
const cursors = new Map(groupOrder.map((id) => [id, 0]));
const firstAskPos = new Map(); // canonicalId -> position in stream
const stream = [];
let progress = true;
while (progress) {
  progress = false;
  for (const gid of groupOrder) {
    const g = groups.get(gid);
    const cur = cursors.get(gid);
    if (cur < g.length) {
      if (cur === 0) firstAskPos.set(gid, stream.length);
      stream.push(g[cur]);
      cursors.set(gid, cur + 1);
      progress = true;
    }
  }
}

// --- place near-miss traps after their forbidden canonicals' first asks ----
const nmItems = shuffle(nearmiss.map((n) => ({
  text: n.text,
  expect: 'miss',
  forbidCanonicalIds: n.forbid,
  note: n.note,
  after: Math.max(...n.forbid.map((f) => firstAskPos.get(f) ?? 0)),
})));
for (const item of nmItems) {
  const pos = item.after + 1 + Math.floor(rng() * (stream.length - item.after - 1));
  const { after, ...rec } = item;
  stream.splice(pos, 0, rec);
}

// --- guards at seeded positions -------------------------------------------
for (const g of shuffle(guards)) {
  const pos = Math.floor(rng() * (stream.length + 1));
  stream.splice(pos, 0, { text: g.text, expect: 'guard', note: g.reason });
}

const workload = {
  version: 1,
  comment: 'Labeled replay workload for the NimbusDB helpdesk simulation. Assembled by scripts/gen-data.mjs from hand-authored seeds; regenerate with `node scripts/gen-data.mjs`.',
  queries: stream.map((q, i) => ({ id: `q${String(i + 1).padStart(3, '0')}`, ...q })),
};

// --- labeled pairs for calibration ----------------------------------------
// positives: canonical question vs its own paraphrase
const positives = paraphrases.map((p, i) => ({
  a: canonicals.find((c) => c.id === p.canonicalId).question,
  b: p.texts[0],
  same: 1,
  groupId: p.canonicalId,
}));
// hard negatives: near-miss text vs the canonical it forbids (first forbid)
const hardNegatives = nearmiss.map((n) => ({
  a: n.text,
  b: canonicals.find((c) => c.id === n.forbid[0]).question,
  same: 0,
  groupId: n.forbid[0],
}));
// easy negatives: random cross-topic canonical pairs
const easyNegatives = [];
const seenPairs = new Set();
while (easyNegatives.length < 36) {
  const a = canonicals[Math.floor(rng() * canonicals.length)];
  const b = canonicals[Math.floor(rng() * canonicals.length)];
  if (a.id === b.id || a.topic === b.topic) continue;
  const key = [a.id, b.id].sort().join('|');
  if (seenPairs.has(key)) continue;
  seenPairs.add(key);
  easyNegatives.push({ a: a.question, b: b.question, same: 0, groupId: a.id });
}
const allPairs = [...positives, ...hardNegatives, ...easyNegatives].map((p, i) => ({ id: `p${String(i + 1).padStart(3, '0')}`, ...p }));

// deterministic fold split by group: fnv(groupId) % 4 === 0 -> dev
function fnv(s) {
  let h = 0x811c9dc5 >>> 0;
  for (let i = 0; i < s.length; i++) { h ^= s.charCodeAt(i); h = Math.imul(h, 0x01000193); }
  return h >>> 0;
}
const train = allPairs.filter((p) => fnv(p.groupId) % 4 !== 0);
const dev = allPairs.filter((p) => fnv(p.groupId) % 4 === 0);
const rate = (xs) => (xs.reduce((a, p) => a + p.same, 0) / xs.length).toFixed(3);
console.log(`pairs: ${allPairs.length} total, train ${train.length} (same-rate ${rate(train)}), dev ${dev.length} (same-rate ${rate(dev)})`);

writeFileSync(join(root, 'data/helpdesk/workload.json'), JSON.stringify(workload, null, 2) + '\n');
writeFileSync(join(root, 'data/helpdesk/pairs.json'), JSON.stringify({ version: 1, train, dev }, null, 2) + '\n');
console.log(`workload: ${workload.queries.length} queries`);
const counts = {};
for (const q of workload.queries) counts[q.expect] = (counts[q.expect] ?? 0) + 1;
console.log(counts);

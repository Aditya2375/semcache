import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { parseCanonicalSet, parsePairs, parseWorkload } from '../../src/core/workload.js';

const read = (p: string) => JSON.parse(readFileSync(join(process.cwd(), p), 'utf8'));

test('bundled canonical set parses and covers 12 topics', () => {
  const cs = parseCanonicalSet(read('data/helpdesk/canonical.json'));
  assert.equal(cs.length, 60);
  assert.equal(new Set(cs.map((c) => c.topic)).size, 12);
});

test('bundled workload parses, references resolve, expectations are consistent', () => {
  const cs = parseCanonicalSet(read('data/helpdesk/canonical.json'));
  const ids = new Set(cs.map((c) => c.id));
  const w = parseWorkload(read('data/helpdesk/workload.json'), ids);
  assert.ok(w.length >= 250);
  const byExpect = new Map<string, number>();
  for (const q of w) byExpect.set(q.expect, (byExpect.get(q.expect) ?? 0) + 1);
  for (const k of ['miss', 'hit-exact', 'hit-semantic', 'guard']) assert.ok((byExpect.get(k) ?? 0) > 10, `expect ${k}`);
  // ordering invariant: first ask precedes repeats/paraphrases of a canonical
  const seenFirst = new Set<string>();
  const canonicalByQuestion = new Map(cs.map((c) => [c.question, c.id]));
  for (const q of w) {
    if (q.expect === 'miss' && q.canonicalId === undefined && !q.forbidCanonicalIds && q.note === 'first ask') {
      const cid = canonicalByQuestion.get(q.text);
      if (cid) seenFirst.add(cid);
    }
    if (q.expect === 'hit-semantic') assert.ok(seenFirst.has(q.canonicalId!), `first ask precedes ${q.id}`);
  }
});

test('workload validator rejects bad shapes', () => {
  const ids = new Set(['c01']);
  assert.throws(() => parseWorkload({ queries: [] }, ids));
  assert.throws(() => parseWorkload({ queries: [{ id: 'a', text: 'x y', expect: 'hit-semantic' }] }, ids));
  assert.throws(() => parseWorkload({ queries: [{ id: 'a', text: 'x y', expect: 'hit-semantic', canonicalId: 'nope' }] }, ids));
  assert.throws(() => parseWorkload({ queries: [{ id: 'a', text: 'x y', expect: 'nonsense' }] }, ids));
  assert.throws(() => parseWorkload({ queries: [{ id: 'a', text: 'x y', expect: 'miss', canonicalId: 'c01' }] }, ids));
});

test('pairs file parses, folds are group-disjoint and balanced-ish', () => {
  const { train, dev } = parsePairsFile();
  assert.ok(train.length >= 60 && dev.length >= 20);
  const trainGroups = new Set(train.map((p) => p.groupId));
  for (const p of dev) assert.ok(!trainGroups.has(p.groupId), `group ${p.groupId} leaks across folds`);
  const rate = (xs: Array<{ same: number }>) => xs.reduce((a, p) => a + p.same, 0) / xs.length;
  assert.ok(rate(train) > 0.3 && rate(train) < 0.7);
  assert.ok(rate(dev) > 0.2 && rate(dev) < 0.8);
});

function parsePairsFile() {
  const raw = read('data/helpdesk/pairs.json');
  const train = parsePairs({ pairs: raw.train });
  const dev = parsePairs({ pairs: raw.dev });
  return { train, dev };
}

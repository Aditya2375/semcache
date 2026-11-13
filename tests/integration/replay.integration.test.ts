/**
 * End-to-end replay of the bundled labeled workload.
 *
 * Two embedders, two jobs:
 *   - DatasetEmbedder assigns each intent cluster its own seeded direction
 *     with small per-text jitter - a competent embedder under which the
 *     replay protocol's accounting (categories, guards, exact path, cost)
 *     must come out exactly right
 *   - a handcrafted trap embedder engineers a false hit so wrong-hit
 *     accounting is pinned deterministically
 * Paraphrase hit QUALITY with the real model is verified separately against
 * precomputed bge vectors (scripts/verify-precomputed.mjs, README numbers).
 */
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import type { Embedder } from '../../src/core/embed.js';
import { SemanticCache } from '../../src/core/cache.js';
import { FixedThreshold } from '../../src/core/policy.js';
import { SimulatedBackend } from '../../src/core/stub.js';
import { runReplay } from '../../src/core/replay.js';
import { parseCanonicalSet, parseWorkload, type WorkloadQuery } from '../../src/core/workload.js';
import { ILLUSTRATIVE_PRICE } from '../../src/core/costmodel.js';
import { mulberry32 } from '../../src/core/rng.js';
import { normalize } from '../../src/core/vector.js';

const read = (p: string) => JSON.parse(readFileSync(join(process.cwd(), p), 'utf8'));
const canonicals = parseCanonicalSet(read('data/helpdesk/canonical.json'));
const workload = parseWorkload(read('data/helpdesk/workload.json'), new Set(canonicals.map((c) => c.id)));

/** Each cluster (canonical or standalone near-miss) gets a seeded direction; texts in a cluster get tiny jitter. */
class DatasetEmbedder implements Embedder {
  readonly id = 'dataset-embedder';
  readonly dims = 64;
  private readonly vectors = new Map<string, Float32Array>();
  constructor(workloadQueries: WorkloadQuery[], canonicalQuestions: string[], seed = 17) {
    const rng = mulberry32(seed);
    const direction = () => {
      const v = new Float32Array(this.dims);
      for (let i = 0; i < this.dims; i++) v[i] = rng() * 2 - 1;
      return normalize(v);
    };
    const jittered = (base: Float32Array, amount: number) => {
      const v = new Float32Array(base);
      for (let i = 0; i < this.dims; i++) v[i]! += (rng() * 2 - 1) * amount;
      return normalize(v);
    };
    const canonicalDir = new Map<string, Float32Array>();
    for (const q of canonicalQuestions) {
      const cid = q; // canonical question text belongs to its own cluster
      const dir = direction();
      canonicalDir.set(cid, dir);
      this.vectors.set(q, jittered(dir, 0.001));
    }
    const canonicalIdByQuestion = new Map(canonicals.map((c) => [c.question, c.id]));
    const dirByCanonicalId = new Map<string, Float32Array>();
    for (const [q, dir] of canonicalDir) dirByCanonicalId.set(canonicalIdByQuestion.get(q)!, dir);
    for (const q of workloadQueries) {
      if (this.vectors.has(q.text)) continue;
      if (q.expect === 'hit-semantic') {
        this.vectors.set(q.text, jittered(dirByCanonicalId.get(q.canonicalId!)!, 0.02));
      } else {
        this.vectors.set(q.text, jittered(direction(), 0.001));
      }
    }
  }
  embed(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(
      texts.map((t) => {
        const v = this.vectors.get(t);
        if (!v) throw new Error(`dataset-embedder: unknown text "${t}"`);
        return v;
      }),
    );
  }
}

async function replayWith(embedder: Embedder, seed = 17, tau = 0.85) {
  const cache = new SemanticCache<{ canonicalId: string | null; answer: string }>({
    embedder,
    policy: new FixedThreshold(tau),
    now: (() => {
      let t = 1_000_000;
      return () => (t += 100);
    })(),
  });
  const backend = new SimulatedBackend(canonicals, ILLUSTRATIVE_PRICE, seed);
  return runReplay(workload, cache, backend);
}

test('under a competent embedder the workload scores perfectly', async () => {
  const { summary } = await replayWith(new DatasetEmbedder(workload, canonicals.map((c) => c.question)));
  assert.equal(summary.total, workload.length);
  assert.equal(summary.correctHits, 180); // 60 exact + 120 semantic
  assert.equal(summary.wrongHits, 0);
  assert.equal(summary.missedHits, 0);
  assert.equal(summary.correctMisses, 84);
  assert.equal(summary.guardOk, 16);
  assert.equal(summary.guardMissed, 0);
  assert.equal(summary.hitPrecision, 1);
  assert.equal(summary.hitRecall, 1);
});

test('replay is deterministic for a fixed seed', async () => {
  const make = () => replayWith(new DatasetEmbedder(workload, canonicals.map((c) => c.question)));
  const a = await make();
  const b = await make();
  assert.deepEqual(a.summary, b.summary);
  assert.deepEqual(a.events.map((e) => e.category), b.events.map((e) => e.category));
});

test('exact repeats are served from the exact path', async () => {
  const { events } = await replayWith(new DatasetEmbedder(workload, canonicals.map((c) => c.question)));
  const exacts = events.filter((e) => workload.find((q) => q.id === e.queryId)!.expect === 'hit-exact');
  assert.equal(exacts.length, 60);
  for (const e of exacts) {
    assert.equal(e.actual, 'hit-exact', `${e.queryId} should exact-hit`);
    assert.equal(e.category, 'correct-hit');
  }
});

test('cache path costs far less than the no-cache baseline', async () => {
  const { summary } = await replayWith(new DatasetEmbedder(workload, canonicals.map((c) => c.question)));
  assert.ok(summary.costSavedPct > 50, `expected >50% savings, got ${summary.costSavedPct.toFixed(1)}%`);
  assert.ok(summary.p50LatencyCacheMs <= summary.p50LatencyBaselineMs);
});

test('an engineered trap is counted as a wrong hit, never silently', async () => {
  // trap: cached "delete database" entry, incoming "delete collection" at sim 0.93 > tau 0.9
  const emb: Embedder = {
    id: 'trap',
    dims: 2,
    embed: (texts: string[]) =>
      Promise.resolve(
        texts.map((t) =>
          normalize(
            t === 'cached question' ? new Float32Array([1, 0]) : new Float32Array([Math.cos(0.38), Math.sin(0.38)]),
          ),
        ),
      ),
  };
  const trapWorkload: WorkloadQuery[] = [
    { id: 't1', text: 'cached question', expect: 'miss', note: 'first ask' },
    { id: 't2', text: 'trap question', expect: 'miss', forbidCanonicalIds: ['c01'] },
  ];
  const cache = new SemanticCache<{ canonicalId: string | null; answer: string }>({
    embedder: emb,
    policy: new FixedThreshold(0.9),
  });
  const backend = new SimulatedBackend(canonicals, ILLUSTRATIVE_PRICE, 3);
  const { events, summary } = await runReplay(trapWorkload, cache, backend);
  assert.equal(events[1]!.category, 'wrong-hit');
  assert.equal(events[1]!.actual, 'hit-semantic');
  assert.equal(summary.wrongHits, 1);
  assert.equal(summary.hitPrecision, 0);
  assert.ok(summary.wrongHitRate > 0);
});

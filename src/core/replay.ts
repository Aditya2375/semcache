/**
 * Workload replay: the evaluation protocol.
 *
 * Every query in the labeled workload is replayed against a fresh cache, in
 * order, exactly as a production cache would see it: lookup, serve or
 * generate, store (unless a guard refuses). Each event is then scored
 * against its expectation:
 *
 *   expect hit-exact    -> a hit serving the canonical stored by the earlier
 *                          occurrence is a correct hit; a miss is a missed hit
 *   expect hit-semantic -> a hit is correct only if it serves canonicalId;
 *                          any other hit is a wrong hit; a miss is a missed hit
 *   expect miss         -> a miss is correct; any hit is a wrong hit
 *   expect guard        -> the guard must fire; anything else means an
 *                          un-cacheable query entered the cache
 *
 * The same query stream also runs through the no-cache baseline (every query
 * generates) so cost and latency deltas are measured, not asserted.
 * Determinism: seeded backend, seeded ANN, pinned embedder - replaying twice
 * yields byte-identical summaries.
 */

import type { SemanticCache, LookupResult } from './cache.js';
import type { SimulatedBackend } from './stub.js';
import { normalizeQuery } from './normalize.js';
import { percentile, wilson } from './metrics.js';
import type { WorkloadQuery } from './workload.js';

export interface CachedValue {
  canonicalId: string | null;
  answer: string;
}

export type EventCategory = 'correct-hit' | 'wrong-hit' | 'missed-hit' | 'correct-miss' | 'guard-ok' | 'guard-missed';

export interface ReplayEvent {
  queryId: string;
  text: string;
  category: EventCategory;
  actual: 'hit-exact' | 'hit-semantic' | 'miss' | 'guard';
  servedCanonicalId: string | null;
  expectedCanonicalId: string | null;
  similarity: number | null;
  p: number | null;
  reason: string;
  latencyMs: number;
  costUsd: number;
  baselineLatencyMs: number;
  baselineCostUsd: number;
}

export interface ReplaySummary {
  total: number;
  hits: number;
  correctHits: number;
  wrongHits: number;
  missedHits: number;
  correctMisses: number;
  guardOk: number;
  guardMissed: number;
  /** Of queries that hoped for a hit, fraction served the right answer. */
  hitRecall: number;
  hitRecallCi: [number, number];
  /** Of all hits served, fraction that served the right answer. */
  hitPrecision: number;
  hitPrecisionCi: [number, number];
  /** Of all queries, fraction served a wrong cached answer. The danger number. */
  wrongHitRate: number;
  wrongHitRateCi: [number, number];
  guardAccuracy: number;
  costCacheUsd: number;
  costBaselineUsd: number;
  costSavedUsd: number;
  costSavedPct: number;
  p50LatencyCacheMs: number;
  p95LatencyCacheMs: number;
  p50LatencyBaselineMs: number;
  p95LatencyBaselineMs: number;
}

export interface ReplayResult {
  events: ReplayEvent[];
  summary: ReplaySummary;
}

export async function runReplay(
  queries: WorkloadQuery[],
  cache: SemanticCache<CachedValue>,
  backend: SimulatedBackend,
): Promise<ReplayResult> {
  const events: ReplayEvent[] = [];
  /** normalized text -> canonical stored by its first occurrence */
  const truthByNorm = new Map<string, string | null>();
  let expectedHits = 0;

  for (const q of queries) {
    const norm = normalizeQuery(q.text);
    const expectedCanonical =
      q.expect === 'hit-semantic'
        ? q.canonicalId!
        : q.expect === 'hit-exact'
          ? truthByNorm.get(norm) ?? null
          : (q.answerCanonicalId ?? null); // first asks: what a correct backend answers; never used to score hits
    if (q.expect === 'hit-exact' || q.expect === 'hit-semantic') expectedHits += 1;

    const result: LookupResult<CachedValue> = await cache.lookup('workload', 'sim-v1', q.text);
    let actual: ReplayEvent['actual'];
    let category: EventCategory;
    let servedCanonicalId: string | null = null;
    let similarity: number | null = null;
    let p: number | null = null;
    let reason = '';
    let latencyMs = 0;
    let costUsd = backend.lookupCostUsd(q.text);

    if (result.kind === 'guard') {
      actual = 'guard';
      category = q.expect === 'guard' ? 'guard-ok' : 'guard-missed';
      reason = result.reason;
      const gen = backend.generate(q.text, expectedCanonical);
      latencyMs = gen.latencyMs;
      costUsd += gen.cost.usd;
      // guard: never stored
    } else if (result.kind === 'hit-exact' || result.kind === 'hit-semantic') {
      actual = result.kind;
      servedCanonicalId = result.entry.value.canonicalId;
      similarity = result.kind === 'hit-semantic' ? result.similarity : 1;
      p = result.kind === 'hit-semantic' ? (result.p ?? null) : null;
      reason = result.kind === 'hit-semantic' ? result.policyReason : 'exact key match';
      latencyMs = 4; // cache serve: embedding lookup is local; ~4 ms
      const right =
        q.expect === 'hit-exact'
          ? servedCanonicalId === (truthByNorm.get(norm) ?? null) && truthByNorm.has(norm)
          : q.expect === 'hit-semantic'
            ? servedCanonicalId === q.canonicalId
            : false;
      category = q.expect === 'miss' ? 'wrong-hit' : right ? 'correct-hit' : q.expect === 'guard' ? 'guard-missed' : 'wrong-hit';
    } else {
      actual = 'miss';
      reason = result.reason;
      if (q.expect === 'hit-exact' || q.expect === 'hit-semantic') category = 'missed-hit';
      else if (q.expect === 'guard') category = 'guard-missed';
      else category = 'correct-miss';
      const gen = backend.generate(q.text, expectedCanonical);
      latencyMs = gen.latencyMs;
      costUsd += gen.cost.usd;
      const stored = await cache.store('workload', 'sim-v1', q.text, { canonicalId: gen.canonicalId, answer: gen.text });
      if (!('skipped' in stored)) truthByNorm.set(norm, gen.canonicalId);
    }

    const baseline = backend.generate(q.text, expectedCanonical);
    events.push({
      queryId: q.id,
      text: q.text,
      category,
      actual,
      servedCanonicalId,
      expectedCanonicalId: expectedCanonical ?? null,
      similarity,
      p,
      reason,
      latencyMs,
      costUsd,
      baselineLatencyMs: baseline.latencyMs,
      baselineCostUsd: baseline.cost.usd,
    });
  }

  const hits = events.filter((e) => e.actual === 'hit-exact' || e.actual === 'hit-semantic').length;
  const correctHits = events.filter((e) => e.category === 'correct-hit').length;
  const wrongHits = events.filter((e) => e.category === 'wrong-hit').length;
  const missedHits = events.filter((e) => e.category === 'missed-hit').length;
  const correctMisses = events.filter((e) => e.category === 'correct-miss').length;
  const guardOk = events.filter((e) => e.category === 'guard-ok').length;
  const guardMissed = events.filter((e) => e.category === 'guard-missed').length;
  const guardTotal = guardOk + guardMissed;
  const costCache = events.reduce((a, e) => a + e.costUsd, 0);
  const costBaseline = events.reduce((a, e) => a + e.baselineCostUsd, 0);
  const latCache = events.map((e) => e.latencyMs).sort((a, b) => a - b);
  const latBaseline = events.map((e) => e.baselineLatencyMs).sort((a, b) => a - b);

  const summary: ReplaySummary = {
    total: events.length,
    hits,
    correctHits,
    wrongHits,
    missedHits,
    correctMisses,
    guardOk,
    guardMissed,
    hitRecall: expectedHits > 0 ? correctHits / expectedHits : Number.NaN,
    hitRecallCi: wilson(correctHits, Math.max(expectedHits, 1)),
    hitPrecision: hits > 0 ? correctHits / hits : Number.NaN,
    hitPrecisionCi: wilson(correctHits, Math.max(hits, 1)),
    wrongHitRate: wrongHits / events.length,
    wrongHitRateCi: wilson(wrongHits, events.length),
    guardAccuracy: guardTotal > 0 ? guardOk / guardTotal : Number.NaN,
    costCacheUsd: costCache,
    costBaselineUsd: costBaseline,
    costSavedUsd: costBaseline - costCache,
    costSavedPct: costBaseline > 0 ? (1 - costCache / costBaseline) * 100 : Number.NaN,
    p50LatencyCacheMs: percentile(latCache, 50),
    p95LatencyCacheMs: percentile(latCache, 95),
    p50LatencyBaselineMs: percentile(latBaseline, 50),
    p95LatencyBaselineMs: percentile(latBaseline, 95),
  };
  return { events, summary };
}

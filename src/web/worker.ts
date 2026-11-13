/**
 * Demo worker: owns the cache, the simulated backend, the workload replay
 * and the threshold sweep. Runs off the main thread so model downloads and
 * 25-replay sweeps never block the UI.
 *
 * Protocol (request -> response):
 *   init        -> boot stats (counts, embedder id, calibration dev metrics)
 *   query       -> console lookup/store against the live cache; falls back to
 *                  the lazy live model for texts outside the bundled artifact
 *   replay      -> full workload replay under one policy: summary + events
 *   sweep       -> precision/recall/wrong-hit/cost curves over tau and p*
 *   gallery     -> near-miss outcomes side by side under two policies
 */

import { SemanticCache, type LookupResult } from '../core/cache.js';
import { CalibratedPolicy, FixedThreshold, MarginPolicy, type HitPolicy } from '../core/policy.js';
import { PrecomputedEmbedder, loadPrecomputed } from '../core/precomputed.js';
import { SimulatedBackend } from '../core/stub.js';
import { runReplay, type CachedValue, type ReplayResult } from '../core/replay.js';
import { parseCanonicalSet, parseWorkload, type CanonicalQA, type WorkloadQuery } from '../core/workload.js';
import { classifyCacheability } from '../core/guards.js';
import { ILLUSTRATIVE_PRICE, generationCost } from '../core/costmodel.js';
import { auc, brier, logLoss } from '../core/metrics.js';
import { calibratedP } from '../core/policy.js';
import { CompositeEmbedder, TransformersEmbedder } from './models.js';
import type { CalibrationWeights } from '../core/policy.js';

export type PolicySpec =
  | { kind: 'fixed'; tau: number }
  | { kind: 'margin'; tau: number; margin: number }
  | { kind: 'calibrated'; pStar?: number; wrongMultiple?: number };

function makePolicy(spec: PolicySpec, weights: CalibrationWeights): HitPolicy {
  switch (spec.kind) {
    case 'fixed':
      return new FixedThreshold(spec.tau);
    case 'margin':
      return new MarginPolicy(spec.tau, spec.margin);
    case 'calibrated':
      return new CalibratedPolicy(weights, spec.pStar !== undefined ? { decisionP: spec.pStar } : { wrongAnswerCostMultiple: spec.wrongMultiple ?? 10 });
  }
}

interface Boot {
  canonicals: CanonicalQA[];
  workload: WorkloadQuery[];
  pre: ReturnType<typeof loadPrecomputed>;
  embedder: PrecomputedEmbedder;
  weights: CalibrationWeights;
  calibrationMeta: { referenceCacheSize: number };
  devMetrics: { auc: number; logloss: number; brier: number; pairs: number };
}

let boot: Boot | null = null;
const liveEmbedder = new TransformersEmbedder();
let composite: CompositeEmbedder;
let consoleCache: SemanticCache<CachedValue> | null = null;
let consoleSpend = { lookupUsd: 0, generationUsd: 0, baselineUsd: 0 };

async function loadJson(url: string): Promise<unknown> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  return res.json();
}

async function doInit(): Promise<Boot> {
  if (boot) return boot;
  const [canonicalsRaw, workloadRaw, calibrationRaw, preRaw] = await Promise.all([
    loadJson('./data/helpdesk/canonical.json'),
    loadJson('./data/helpdesk/workload.json'),
    loadJson('./data/helpdesk/calibration.json'),
    loadJson('./data/precomputed/helpdesk.bge-small-en-v1.5.json'),
  ]);
  const canonicals = parseCanonicalSet(canonicalsRaw);
  const workload = parseWorkload(workloadRaw, new Set(canonicals.map((c) => c.id)));
  const pre = loadPrecomputed(preRaw);
  const embedder = new PrecomputedEmbedder(pre);
  const cal = calibrationRaw as { referenceCacheSize: number; weights: CalibrationWeights };

  // dev-fold quality of the calibration fit, computed live from the artifact
  const pairsRaw = (await loadJson('./data/helpdesk/pairs.json')) as {
    dev: Array<{ a: string; b: string; same: 0 | 1 }>;
  };
  const scored: Array<{ p: number; label: 0 | 1 }> = [];
  for (const pair of pairsRaw.dev) {
    if (!embedder.has(pair.a) || !embedder.has(pair.b)) continue;
    const [aVec, bVec] = await embedder.embed([pair.a, pair.b]);
    let sim2 = -1;
    for (const c of canonicals) {
      if (c.question === pair.b) continue;
      const [cVec] = await embedder.embed([c.question]);
      const dotp = aVec!.reduce((acc, x, i) => acc + x * cVec![i]!, 0);
      if (dotp > sim2) sim2 = dotp;
    }
    const sim1 = aVec!.reduce((acc, x, i) => acc + x * bVec![i]!, 0);
    scored.push({
      p: calibratedP(cal.weights, { sim1, sim2, cacheSize: canonicals.length, queryTokens: pair.a.trim().split(/\s+/).length }),
      label: pair.same,
    });
  }
  const devMetrics = {
    auc: auc(scored.map((s) => ({ score: s.p, label: s.label }))),
    logloss: logLoss(scored),
    brier: brier(scored),
    pairs: scored.length,
  };

  composite = new CompositeEmbedder(embedder, liveEmbedder);
  boot = {
    canonicals,
    workload,
    pre,
    embedder,
    weights: cal.weights,
    calibrationMeta: { referenceCacheSize: cal.referenceCacheSize },
    devMetrics,
  };
  return boot;
}

function newCache(b: Boot, policy: HitPolicy): SemanticCache<CachedValue> {
  return new SemanticCache<CachedValue>({ embedder: composite ?? b.embedder, policy, maxEntries: 2000 });
}

function makeBackend(b: Boot): SimulatedBackend {
  return new SimulatedBackend(b.canonicals, ILLUSTRATIVE_PRICE, 23);
}

interface QueryTrace {
  stage: string;
  detail: string;
}

let progressPoster: ((fraction: number) => void) | null = null;

async function doQuery(text: string, spec: PolicySpec): Promise<Record<string, unknown>> {
  const b = await doInit();
  // A query outside the bundled artifact needs the live model; start its
  // download here so progress reaches the page instead of a silent stall.
  if (!b.embedder.has(text) && !liveEmbedder.loaded) {
    await liveEmbedder.load((f) => progressPoster?.(f));
  }
  const policy = makePolicy(spec, b.weights);
  if (!consoleCache) {
    consoleCache = new SemanticCache<CachedValue>({ embedder: composite, policy, maxEntries: 2000 });
    consoleSpend = { lookupUsd: 0, generationUsd: 0, baselineUsd: 0 };
  }
  const cache = consoleCache;
  const trace: QueryTrace[] = [];
  const backend = makeBackend(b);
  const verdict = classifyCacheability(text);
  trace.push({ stage: 'guard', detail: verdict.cacheable ? 'cacheable' : `rejected: ${verdict.reason}` });
  if (!verdict.cacheable) {
    const gen = backend.generate(text, null);
    consoleSpend.generationUsd += gen.cost.usd;
    consoleSpend.baselineUsd += gen.cost.usd;
    return { kind: 'guard', reason: verdict.reason, answer: gen.text, latencyMs: gen.latencyMs, costUsd: gen.cost.usd, trace, spend: consoleSpend, modelsUsed: composite.usedLive ? liveEmbedder.id : 'precomputed artifact' };
  }

  trace.push({ stage: 'exact', detail: 'checking normalized exact key' });
  const result: LookupResult<CachedValue> = await cache.lookup('console', 'sim-v1', text);
  const lookupUsd = backend.lookupCostUsd(text);
  consoleSpend.lookupUsd += lookupUsd;

  if (result.kind === 'guard') {
    const gen = backend.generate(text, null);
    consoleSpend.generationUsd += gen.cost.usd;
    consoleSpend.baselineUsd += gen.cost.usd;
    return { kind: 'guard', reason: result.reason, answer: gen.text, latencyMs: gen.latencyMs, costUsd: gen.cost.usd, trace, spend: consoleSpend, modelsUsed: composite.usedLive ? liveEmbedder.id : 'precomputed artifact' };
  }
  if (result.kind === 'hit-exact' || result.kind === 'hit-semantic') {
    trace.push({ stage: 'decision', detail: result.kind === 'hit-exact' ? 'exact key match' : result.policyReason });
    consoleSpend.baselineUsd += generationCost(text, result.entry.value.answer, ILLUSTRATIVE_PRICE).usd;
    return {
      kind: result.kind,
      answer: result.entry.value.answer,
      matchedQuery: result.entry.queryRaw,
      similarity: result.kind === 'hit-semantic' ? result.similarity : 1,
      p: result.kind === 'hit-semantic' ? (result.p ?? null) : null,
      entryHits: result.entry.hits,
      latencyMs: 4,
      costUsd: lookupUsd,
      trace,
      spend: consoleSpend,
      modelsUsed: composite.usedLive ? liveEmbedder.id : 'precomputed artifact',
    };
  }

  trace.push({ stage: 'decision', detail: `miss: ${result.reason}` });
  const gen = backend.generate(text, null);
  consoleSpend.generationUsd += gen.cost.usd;
  consoleSpend.baselineUsd += gen.cost.usd;
  const stored = await cache.store('console', 'sim-v1', text, { canonicalId: gen.canonicalId, answer: gen.text });
  trace.push({ stage: 'store', detail: 'skipped' in stored ? `not stored: ${stored.reason}` : 'stored for future lookups' });
  return {
    kind: 'miss',
    reason: result.reason,
    bestSimilarity: 'bestSimilarity' in result ? result.bestSimilarity : null,
    answer: gen.text,
    latencyMs: gen.latencyMs,
    costUsd: gen.cost.usd + lookupUsd,
    trace,
    spend: consoleSpend,
    modelsUsed: composite.usedLive ? liveEmbedder.id : 'precomputed artifact',
  };
}

async function doReplay(spec: PolicySpec): Promise<ReplayResult> {
  const b = await doInit();
  const cache = newCache(b, makePolicy(spec, b.weights));
  const backend = makeBackend(b);
  return runReplay(b.workload, cache, backend);
}

async function doSweep(): Promise<Record<string, unknown>> {
  const b = await doInit();
  const fixedCurve = [];
  for (let tau = 0.7; tau <= 0.981; tau += 0.02) {
    const { summary } = await doReplay({ kind: 'fixed', tau: Math.round(tau * 100) / 100 });
    fixedCurve.push({ param: Math.round(tau * 100) / 100, precision: summary.hitPrecision, recall: summary.hitRecall, wrongHitRate: summary.wrongHitRate, costSavedPct: summary.costSavedPct });
  }
  const calibratedCurve = [];
  for (const pStar of [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.99]) {
    const { summary } = await doReplay({ kind: 'calibrated', pStar });
    calibratedCurve.push({ param: pStar, precision: summary.hitPrecision, recall: summary.hitRecall, wrongHitRate: summary.wrongHitRate, costSavedPct: summary.costSavedPct });
  }
  return { fixedCurve, calibratedCurve, devMetrics: b.devMetrics };
}

async function doGallery(): Promise<Record<string, unknown>> {
  const b = await doInit();
  const naive = await doReplay({ kind: 'fixed', tau: 0.85 });
  const guarded = await doReplay({ kind: 'calibrated', wrongMultiple: 10 });
  const naiveWrong = new Map(naive.events.filter((e) => e.category === 'wrong-hit').map((e) => [e.queryId, e]));
  const rows = [];
  for (const q of b.workload) {
    if (q.expect !== 'miss' || !q.forbidCanonicalIds) continue;
    const nw = naiveWrong.get(q.id);
    const ge = guarded.events.find((e) => e.queryId === q.id);
    rows.push({
      queryId: q.id,
      text: q.text,
      note: q.note ?? '',
      naiveOutcome: nw ? `wrong hit: served ${nw.servedCanonicalId} (sim ${nw.similarity?.toFixed(3)})` : 'miss (correct)',
      calibratedOutcome: ge ? (ge.category === 'correct-miss' ? 'miss (correct)' : `${ge.category} (sim ${ge.similarity?.toFixed(3) ?? '-'}, p ${ge.p?.toFixed(3) ?? '-'})`) : 'n/a',
    });
  }
  return { rows };
}

interface RpcRequest {
  id: number;
  kind: 'init' | 'query' | 'replay' | 'sweep' | 'gallery' | 'loadModel';
  text?: string;
  policy?: PolicySpec;
}

self.onmessage = async (ev: MessageEvent<RpcRequest>) => {
  const req = ev.data;
  const reply = (ok: boolean, payload: unknown) => {
    (self as unknown as Worker).postMessage(ok ? { id: req.id, ok, result: payload } : { id: req.id, ok, error: String(payload) });
  };
  progressPoster = (fraction) => {
    (self as unknown as Worker).postMessage({ id: req.id, ok: true, progress: fraction });
  };
  try {
    switch (req.kind) {
      case 'init': {
        const b = await doInit();
        reply(true, {
          canonicals: b.canonicals.length,
          workload: b.workload.length,
          texts: b.pre.texts.length,
          embedder: b.embedder.id,
          devMetrics: b.devMetrics,
          price: ILLUSTRATIVE_PRICE,
        });
        break;
      }
      case 'query':
        reply(true, await doQuery(req.text ?? '', req.policy ?? { kind: 'calibrated', wrongMultiple: 10 }));
        break;
      case 'replay':
        reply(true, await doReplay(req.policy ?? { kind: 'calibrated', wrongMultiple: 10 }));
        break;
      case 'sweep':
        reply(true, await doSweep());
        break;
      case 'gallery':
        reply(true, await doGallery());
        break;
      case 'loadModel':
        await liveEmbedder.load((f) => (self as unknown as Worker).postMessage({ id: req.id, progress: f }));
        reply(true, { model: liveEmbedder.id });
        break;
    }
  } catch (err) {
    reply(false, err instanceof Error ? err.message : err);
  } finally {
    progressPoster = null;
  }
};

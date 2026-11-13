import test from 'node:test';
import assert from 'node:assert/strict';
import { fitLogistic, type FitSample } from '../../src/core/calibrate.js';
import { calibratedP } from '../../src/core/policy.js';
import { auc, logLoss } from '../../src/core/metrics.js';
import { mulberry32 } from '../../src/core/rng.js';

function separableSamples(n: number, seed: number): FitSample[] {
  const rng = mulberry32(seed);
  const out: FitSample[] = [];
  for (let i = 0; i < n; i++) {
    const label = i % 2 === 0 ? 1 : 0;
    const sim = label === 1 ? 0.85 + rng() * 0.1 : 0.3 + rng() * 0.2;
    const margin = label === 1 ? 0.2 + rng() * 0.3 : 0.0 + rng() * 0.05;
    out.push({ features: [sim, margin, Math.log1p(50), 6], label: label as 0 | 1 });
  }
  return out;
}

test('fitLogistic separates an easy problem', () => {
  const samples = separableSamples(120, 7);
  const w = fitLogistic(samples, { seed: 7 });
  const scored = samples.map((s) => ({ score: calibratedP(w, { sim1: s.features[0], sim2: s.features[0] - s.features[1], cacheSize: 50, queryTokens: s.features[3] }), label: s.label }));
  assert.ok(auc(scored) > 0.99);
  assert.ok(logLoss(scored.map((s) => ({ p: s.score, label: s.label }))) < 0.2);
});

test('fitLogistic is deterministic for a fixed seed', () => {
  const samples = separableSamples(60, 3);
  const a = fitLogistic(samples, { seed: 42 });
  const b = fitLogistic(samples, { seed: 42 });
  assert.deepEqual(a.coef, b.coef);
  assert.equal(a.bias, b.bias);
});

test('fitLogistic rejects tiny datasets', () => {
  assert.throws(() => fitLogistic(separableSamples(10, 1)));
});

test('constant features are frozen: std reported 1, coefficient pinned 0, no explosion at inference', () => {
  const rng = mulberry32(11);
  const samples: FitSample[] = [];
  for (let i = 0; i < 60; i++) {
    const label = i % 2 === 0 ? 1 : 0;
    samples.push({
      features: [label === 1 ? 0.85 + rng() * 0.1 : 0.3 + rng() * 0.2, rng() * 0.4, Math.log1p(60), 7],
      label: label as 0 | 1,
    });
  }
  const w = fitLogistic(samples, { seed: 5 });
  assert.equal(w.stds[2], 1, 'degenerate feature std must be reported as 1');
  assert.equal(w.coef[2], 0, 'degenerate feature coefficient must stay 0');
  // inference far from the training constant must not explode
  const p = calibratedP(w, { sim1: 0.5, sim2: 0.3, cacheSize: 100_000, queryTokens: 7 });
  assert.ok(Number.isFinite(p) && p >= 0 && p <= 1, `p must stay finite, got ${p}`);
  const p2 = calibratedP(w, { sim1: 0.5, sim2: 0.3, cacheSize: 2, queryTokens: 7 });
  assert.ok(Number.isFinite(p2) && p2 >= 0 && p2 <= 1);
});

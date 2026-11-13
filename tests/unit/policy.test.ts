import test from 'node:test';
import assert from 'node:assert/strict';
import { CalibratedPolicy, FixedThreshold, MarginPolicy, calibratedP, sigmoid, type CalibrationWeights } from '../../src/core/policy.js';

test('fixed threshold fires at and above tau', () => {
  const p = new FixedThreshold(0.8);
  assert.equal(p.decide({ sim1: 0.8, sim2: null, cacheSize: 10, queryTokens: 5 }).hit, true);
  assert.equal(p.decide({ sim1: 0.799, sim2: null, cacheSize: 10, queryTokens: 5 }).hit, false);
  assert.throws(() => new FixedThreshold(2));
});

test('margin policy requires both threshold and gap', () => {
  const p = new MarginPolicy(0.7, 0.1);
  assert.equal(p.decide({ sim1: 0.9, sim2: 0.5, cacheSize: 10, queryTokens: 5 }).hit, true);
  assert.equal(p.decide({ sim1: 0.9, sim2: 0.85, cacheSize: 10, queryTokens: 5 }).hit, false);
  assert.equal(p.decide({ sim1: 0.6, sim2: null, cacheSize: 10, queryTokens: 5 }).hit, false);
  // single entry in cache: gap is infinite, margin always satisfied
  assert.equal(p.decide({ sim1: 0.71, sim2: null, cacheSize: 1, queryTokens: 5 }).hit, true);
});

test('calibrated policy: probability rises with similarity and margin', () => {
  const weights: CalibrationWeights = {
    means: [0, 0, 0, 0],
    stds: [1, 1, 1, 1],
    coef: [2, 1, 0, 0],
    bias: 0,
    fittedOn: { pairs: 0, method: 'test' },
  };
  const base = { sim2: null, cacheSize: 10, queryTokens: 5 };
  const low = calibratedP(weights, { ...base, sim1: -1 });
  const mid = calibratedP(weights, { ...base, sim1: 0 });
  const high = calibratedP(weights, { ...base, sim1: 1 });
  assert.ok(low < mid && mid < high);
  // margin contributes positively
  const noMargin = calibratedP(weights, { sim1: 0, sim2: 0, cacheSize: 10, queryTokens: 5 });
  assert.ok(calibratedP(weights, { sim1: 0, sim2: null, cacheSize: 10, queryTokens: 5 }) > noMargin);
});

test('calibrated policy operating point from cost multiple', () => {
  const weights: CalibrationWeights = {
    means: [0, 0, 0, 0],
    stds: [1, 1, 1, 1],
    coef: [50, 0, 0, 0],
    bias: 0,
    fittedOn: { pairs: 0, method: 'test' },
  };
  const p = new CalibratedPolicy(weights, { wrongAnswerCostMultiple: 20 });
  assert.ok(Math.abs(p.operatingPoint - 0.95) < 1e-9);
  const hit = p.decide({ sim1: 0.2, sim2: null, cacheSize: 10, queryTokens: 5 });
  assert.equal(hit.hit, true);
  assert.ok(hit.p !== undefined && hit.p > 0.95);
  const miss = p.decide({ sim1: -0.2, sim2: null, cacheSize: 10, queryTokens: 5 });
  assert.equal(miss.hit, false);
  assert.throws(() => new CalibratedPolicy(weights, { wrongAnswerCostMultiple: 1 }));
});

test('sigmoid is symmetric and bounded', () => {
  assert.ok(Math.abs(sigmoid(0) - 0.5) < 1e-12);
  assert.ok(sigmoid(30) < 1 && sigmoid(-30) > 0);
  assert.equal(sigmoid(400), 1); // saturates without overflowing to NaN/Infinity
  assert.ok(sigmoid(-400) < 1e-100);
});

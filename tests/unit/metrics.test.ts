import test from 'node:test';
import assert from 'node:assert/strict';
import { auc, brier, logLoss, percentile, wilson } from '../../src/core/metrics.js';

test('wilson interval for zero successes and full success', () => {
  const [lo0, hi0] = wilson(0, 10);
  assert.equal(lo0, 0);
  assert.ok(hi0 > 0.2 && hi0 < 0.4);
  const [lo1, hi1] = wilson(10, 10);
  assert.equal(hi1, 1);
  assert.ok(lo1 > 0.6 && lo1 < 0.8);
});

test('wilson shrinks with n', () => {
  const [, hiSmall] = wilson(5, 10);
  const [loBig, hiBig] = wilson(500, 1000);
  assert.ok(hiBig - loBig < hiSmall - wilson(5, 10)[0]);
});

test('auc: perfect, reversed, tied', () => {
  assert.equal(auc([{ score: 0.9, label: 1 }, { score: 0.1, label: 0 }]), 1);
  assert.equal(auc([{ score: 0.1, label: 1 }, { score: 0.9, label: 0 }]), 0);
  assert.equal(auc([{ score: 0.5, label: 1 }, { score: 0.5, label: 0 }]), 0.5);
});

test('logloss and brier reward confident correct predictions', () => {
  const good = [{ p: 0.99, label: 1 as const }, { p: 0.01, label: 0 as const }];
  const bad = [{ p: 0.01, label: 1 as const }, { p: 0.99, label: 0 as const }];
  assert.ok(logLoss(good) < logLoss(bad));
  assert.ok(brier(good) < brier(bad));
});

test('percentile picks order statistics', () => {
  const xs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
  assert.equal(percentile(xs, 50), 5);
  assert.equal(percentile(xs, 100), 10);
  assert.ok(Number.isNaN(percentile([], 50)));
});

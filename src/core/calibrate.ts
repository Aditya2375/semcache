/**
 * Calibrated hit probability.
 *
 * A small logistic regression maps the evidence a lookup actually has -
 * best similarity, margin over the strongest competing intent, cache size,
 * query length - to P(the nearest cached entry answers the same question).
 * Fitted once, offline, on labeled query pairs with a group split (all
 * paraphrases of one canonical question stay in the same fold, so the dev
 * fold measures generalisation to unseen intents, not memorised phrasings).
 *
 * The model is deliberately tiny: four standardised features, L2-penalised
 * gradient descent, seeded shuffles. Small enough that every coefficient can
 * be printed and argued about in the Method tab.
 */

import { mulberry32 } from './rng.js';
import { sigmoid, type CalibrationWeights } from './policy.js';

export interface FitSample {
  /** [sim1, margin in [0,1], log1p(cacheSize), queryTokens] */
  features: [number, number, number, number];
  label: 0 | 1;
}

export interface FitOptions {
  seed?: number;
  epochs?: number;
  learningRate?: number;
  l2?: number;
}

export function fitLogistic(samples: FitSample[], opts: FitOptions = {}): CalibrationWeights {
  if (samples.length < 20) throw new Error('calibrate: need at least 20 samples');
  const seed = opts.seed ?? 5;
  const epochs = opts.epochs ?? 800;
  const lr = opts.learningRate ?? 0.1;
  const l2 = opts.l2 ?? 1e-3;

  const means: [number, number, number, number] = [0, 0, 0, 0];
  const stds: [number, number, number, number] = [0, 0, 0, 0];
  for (const s of samples) for (let i = 0; i < 4; i++) means[i]! += s.features[i]! / samples.length;
  for (const s of samples) for (let i = 0; i < 4; i++) stds[i]! += (s.features[i]! - means[i]!) ** 2 / samples.length;
  // zero-variance features leave float-noise variances (~1e-31); a truthiness
  // check would keep 2e-16 as a 'std' and explode at inference - use a floor
  const live: boolean[] = [];
  for (let i = 0; i < 4; i++) {
    const std = Math.sqrt(stds[i]!);
    live.push(std >= 1e-6);
    stds[i] = std >= 1e-6 ? std : 1;
  }

  const xs = samples.map((s) =>
    s.features.map((f, i) => (live[i]! ? (f - means[i]!) / stds[i]! : 0)) as [number, number, number, number],
  );
  const ys = samples.map((s) => s.label);
  const coef: [number, number, number, number] = [0, 0, 0, 0];
  let bias = 0;
  const rng = mulberry32(seed);
  const order = samples.map((_, i) => i);

  for (let epoch = 0; epoch < epochs; epoch++) {
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      const t = order[i]!;
      order[i] = order[j]!;
      order[j] = t;
    }
    for (const idx of order) {
      const x = xs[idx]!;
      let z = bias;
      for (let i = 0; i < 4; i++) z += coef[i]! * x[i]!;
      const err = sigmoid(z) - ys[idx]!;
      for (let i = 0; i < 4; i++) if (live[i]!) coef[i]! -= lr * (err * x[i]! + l2 * coef[i]!);
      bias -= lr * err;
    }
  }

  return {
    means,
    stds,
    coef,
    bias,
    fittedOn: { pairs: samples.length, method: `logistic, ${epochs} epochs, lr=${lr}, l2=${l2}, seed=${seed}` },
  };
}

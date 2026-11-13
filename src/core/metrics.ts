/**
 * Statistics used across the eval harness: Wilson score intervals for rates,
 * rank-based ROC AUC for the calibration fit, and latency percentiles.
 * Every headline number in the README and demo carries its interval - a rate
 * without its uncertainty is a marketing number, not a measurement.
 */

/** Wilson score interval for a binomial proportion. */
export function wilson(successes: number, total: number, z = 1.96): [number, number] {
  if (total === 0) return [0, 1];
  const p = successes / total;
  const denom = 1 + (z * z) / total;
  const centre = (p + (z * z) / (2 * total)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / total + (z * z) / (4 * total * total))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

/** ROC AUC via the Mann-Whitney U statistic; ties count half. */
export function auc(scores: Array<{ score: number; label: 0 | 1 }>): number {
  const pos = scores.filter((s) => s.label === 1);
  const neg = scores.filter((s) => s.label === 0);
  if (pos.length === 0 || neg.length === 0) return Number.NaN;
  let wins = 0;
  for (const p of pos) {
    for (const n of neg) {
      if (p.score > n.score) wins += 1;
      else if (p.score === n.score) wins += 0.5;
    }
  }
  return wins / (pos.length * neg.length);
}

export function logLoss(samples: Array<{ p: number; label: 0 | 1 }>): number {
  if (samples.length === 0) return Number.NaN;
  let sum = 0;
  for (const { p, label } of samples) {
    const pc = Math.min(1 - 1e-9, Math.max(1e-9, p));
    sum += label === 1 ? -Math.log(pc) : -Math.log(1 - pc);
  }
  return sum / samples.length;
}

export function brier(samples: Array<{ p: number; label: 0 | 1 }>): number {
  if (samples.length === 0) return Number.NaN;
  let sum = 0;
  for (const { p, label } of samples) sum += (p - label) ** 2;
  return sum / samples.length;
}

export function percentile(sorted: number[], p: number): number {
  if (sorted.length === 0) return Number.NaN;
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil((p / 100) * sorted.length) - 1));
  return sorted[idx]!;
}

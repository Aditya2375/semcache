/**
 * Hit policies: given the similarity evidence for a lookup, decide whether
 * serving the cached entry is safe enough.
 *
 * The fixed-threshold policy is the industry default and it has a structural
 * flaw: the right threshold depends on the query, on how crowded the cache
 * is around it, and on the cost of being wrong. The calibrated policy treats
 * the decision as a small classification problem - estimate P(this hit is
 * semantically correct) from similarity and margin features, then serve only
 * when the expected cost of a wrong answer beats the cost of regenerating:
 *
 *   serve cached  <=>  (1 - p) * W  <=  C      (W = wrong-answer penalty,
 *   =>  p >= 1 - C/W                             C = regeneration cost)
 *
 * With W/C = 20 (a wrong answer costs twenty regenerations) the operating
 * point is p >= 0.95. The ratio is explicit and configurable because it is a
 * product decision, not a property of the model.
 */

export interface PolicyContext {
  /** Similarity of the best cached candidate. */
  sim1: number;
  /** Similarity of the runner-up candidate, when one exists. */
  sim2: number | null;
  /** Live entries in the queried namespace. */
  cacheSize: number;
  /** Token count of the incoming query. */
  queryTokens: number;
}

export interface PolicyDecision {
  hit: boolean;
  /** Calibrated probability, present only for the calibrated policy. */
  p?: number;
  reason: string;
}

export interface HitPolicy {
  readonly id: string;
  decide(ctx: PolicyContext): PolicyDecision;
}

export class FixedThreshold implements HitPolicy {
  readonly id: string;
  private readonly tau: number;
  constructor(tau: number) {
    if (!(tau > -1 && tau <= 1)) throw new Error('fixed-threshold: tau must be in (-1, 1]');
    this.tau = tau;
    this.id = `fixed(tau=${tau})`;
  }
  decide(ctx: PolicyContext): PolicyDecision {
    const hit = ctx.sim1 >= this.tau;
    return { hit, reason: hit ? `sim ${ctx.sim1.toFixed(3)} >= tau ${this.tau}` : `sim ${ctx.sim1.toFixed(3)} < tau ${this.tau}` };
  }
}

export class MarginPolicy implements HitPolicy {
  readonly id: string;
  private readonly tau: number;
  private readonly margin: number;
  constructor(tau: number, margin: number) {
    if (!(tau > -1 && tau <= 1)) throw new Error('margin-policy: tau must be in (-1, 1]');
    if (!(margin >= 0 && margin <= 2)) throw new Error('margin-policy: margin must be in [0, 2]');
    this.tau = tau;
    this.margin = margin;
    this.id = `margin(tau=${tau},m=${margin})`;
  }
  decide(ctx: PolicyContext): PolicyDecision {
    const gap = ctx.sim2 === null ? Number.POSITIVE_INFINITY : ctx.sim1 - ctx.sim2;
    const hit = ctx.sim1 >= this.tau && gap >= this.margin;
    return {
      hit,
      reason: hit
        ? `sim ${ctx.sim1.toFixed(3)} >= ${this.tau}, margin ${gap === Number.POSITIVE_INFINITY ? 'inf' : gap.toFixed(3)} >= ${this.margin}`
        : `rejected: sim ${ctx.sim1.toFixed(3)} / margin ${gap === Number.POSITIVE_INFINITY ? 'inf' : gap.toFixed(3)}`,
    };
  }
}

export interface CalibrationWeights {
  /** Feature standardization used at fit time; applied identically here. */
  means: [number, number, number, number];
  stds: [number, number, number, number];
  /** Logistic coefficients: [sim1, margin, log1p(cacheSize), queryTokens] + bias. */
  coef: [number, number, number, number];
  bias: number;
  fittedOn: { pairs: number; method: string };
}

export function sigmoid(x: number): number {
  if (x >= 0) return 1 / (1 + Math.exp(-x));
  const e = Math.exp(x);
  return e / (1 + e);
}

export function calibratedP(weights: CalibrationWeights, ctx: PolicyContext): number {
  const margin = ctx.sim2 === null ? 1 : ctx.sim1 - ctx.sim2;
  const feats: [number, number, number, number] = [
    ctx.sim1,
    Math.min(1, Math.max(0, margin)),
    Math.log1p(ctx.cacheSize),
    ctx.queryTokens,
  ];
  let z = weights.bias;
  for (let i = 0; i < 4; i++) {
    const std = weights.stds[i]! < 1e-9 ? 1 : weights.stds[i]!;
    z += weights.coef[i]! * ((feats[i]! - weights.means[i]!) / std);
  }
  return sigmoid(z);
}

export class CalibratedPolicy implements HitPolicy {
  readonly id: string;
  private readonly weights: CalibrationWeights;
  private readonly pStar: number;
  constructor(weights: CalibrationWeights, opts: { wrongAnswerCostMultiple?: number; decisionP?: number } = {}) {
    if (opts.decisionP !== undefined && !(opts.decisionP > 0 && opts.decisionP < 1)) {
      throw new Error('calibrated: decisionP must be in (0, 1)');
    }
    const multiple = opts.wrongAnswerCostMultiple ?? 20;
    if (!(multiple > 1)) throw new Error('calibrated: wrongAnswerCostMultiple must be > 1');
    this.weights = weights;
    this.pStar = opts.decisionP ?? 1 - 1 / multiple;
    this.id = `calibrated(p*=${this.pStar.toFixed(3)})`;
  }
  get operatingPoint(): number {
    return this.pStar;
  }
  decide(ctx: PolicyContext): PolicyDecision {
    const p = calibratedP(this.weights, ctx);
    const hit = p >= this.pStar;
    return {
      hit,
      p,
      reason: hit
        ? `p(correct) ${p.toFixed(3)} >= operating point ${this.pStar.toFixed(3)}`
        : `p(correct) ${p.toFixed(3)} < operating point ${this.pStar.toFixed(3)}`,
    };
  }
}

/**
 * Simulated LLM backend.
 *
 * The demo never calls a real model: generation is mocked by the product
 * documentation itself. Given a matched canonical question the backend
 * returns its documented answer; for anything else it composes an extractive
 * reply from the closest documentation entries. Latency and token counts are
 * simulated from a seeded generator so runs are reproducible, and the UI
 * labels the whole path as a simulation.
 */

import type { CanonicalQA } from './workload.js';
import { mulberry32 } from './rng.js';
import { generationCost, lookupCost, type GenerationCost, type PriceTable } from './costmodel.js';
import { removeStopwords, termFreq, tokenize } from './tokenize.js';
import { estimateTokens } from './costmodel.js';

export interface GenerationResult {
  text: string;
  canonicalId: string | null;
  latencyMs: number;
  cost: GenerationCost;
}

const BASE_LATENCY_MS = 420;
const PER_OUTPUT_TOKEN_MS = 32;

export class SimulatedBackend {
  private readonly byId = new Map<string, CanonicalQA>();
  private readonly rng: () => number;
  private readonly idf = new Map<string, number>();
  private readonly docVectors = new Map<string, Map<string, number>>();

  constructor(
    private readonly canonicals: CanonicalQA[],
    private readonly price: PriceTable,
    seed = 23,
  ) {
    for (const c of canonicals) this.byId.set(c.id, c);
    this.rng = mulberry32(seed);
    // idf over the documentation set, so words that appear everywhere
    // ("nimbusdb") stop dominating the overlap score
    const df = new Map<string, number>();
    for (const c of canonicals) {
      const terms = new Set([...removeStopwords(tokenize(c.question)), ...removeStopwords(tokenize(c.answer))]);
      for (const t of terms) df.set(t, (df.get(t) ?? 0) + 1);
    }
    for (const [t, d] of df) this.idf.set(t, Math.log(1 + canonicals.length / d));
    for (const c of canonicals) this.docVectors.set(c.id, this.vectorOf(`${c.question} ${c.question} ${c.answer}`));
  }

  private vectorOf(text: string): Map<string, number> {
    const tf = termFreq(removeStopwords(tokenize(text)));
    const v = new Map<string, number>();
    for (const [t, n] of tf) v.set(t, n * (this.idf.get(t) ?? Math.log(1 + this.canonicals.length)));
    return v;
  }

  /** Generate an answer for a fresh query (cache miss path). */
  generate(query: string, canonicalId: string | null): GenerationResult {
    const canonical = canonicalId ? this.byId.get(canonicalId) : undefined;
    let text: string;
    let resolvedId: string | null;
    if (canonical) {
      text = canonical.answer;
      resolvedId = canonical.id;
    } else {
      const best = this.closestByOverlap(query);
      if (best) {
        text = best.answer;
        resolvedId = best.id;
      } else {
        text =
          'I could not find anything in the NimbusDB documentation that answers this. Try rephrasing, or check the docs index for the closest topic.';
        resolvedId = null;
      }
    }
    const tokens = estimateTokens(text);
    // seeded jitter: lognormal-ish multiplier in [0.75, 1.5]
    const jitter = Math.exp((this.rng() - 0.5) * 0.7);
    const latencyMs = Math.round((BASE_LATENCY_MS + PER_OUTPUT_TOKEN_MS * tokens) * jitter);
    const cost = generationCost(query, text, this.price);
    return { text, canonicalId: resolvedId, latencyMs, cost };
  }

  /** Cost of a cache lookup itself (embedding the incoming query). */
  lookupCostUsd(query: string): number {
    return lookupCost(query, this.price);
  }

  private closestByOverlap(query: string): CanonicalQA | null {
    const qv = this.vectorOf(query);
    if (qv.size === 0) return null;
    const qnorm = Math.sqrt([...qv.values()].reduce((a, b) => a + b * b, 0));
    let best: CanonicalQA | null = null;
    let bestScore = 0;
    for (const c of this.canonicals) {
      const dv = this.docVectors.get(c.id)!;
      let dot = 0;
      for (const [t, w] of qv) if (dv.has(t)) dot += w * dv.get(t)!;
      const dnorm = Math.sqrt([...dv.values()].reduce((a, b) => a + b * b, 0));
      const score = dot / ((qnorm * dnorm) || 1);
      if (score > bestScore) {
        bestScore = score;
        best = c;
      }
    }
    return bestScore > 0 ? best : null;
  }
}

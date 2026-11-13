/**
 * Illustrative cost model for cache savings.
 *
 * The price table below is a snapshot of published per-token prices, kept so
 * the demo can translate "cache hits" into an order-of-magnitude dollar
 * figure. Prices change; the table carries its source and retrieval date and
 * the README states plainly that the numbers are illustrative, not a quote.
 */

export interface PriceTable {
  model: string;
  inputPerMillionUsd: number;
  outputPerMillionUsd: number;
  embeddingPerMillionUsd: number;
  asOf: string;
  source: string;
}

export const ILLUSTRATIVE_PRICE: PriceTable = {
  model: 'gpt-4o-mini',
  inputPerMillionUsd: 0.15,
  outputPerMillionUsd: 0.6,
  embeddingPerMillionUsd: 0.02,
  asOf: '2026-11-13',
  source: 'https://openai.com/api/pricing/',
};

/** Rough token estimate: ~4 characters per token for English text. */
export function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

export interface GenerationCost {
  inputTokens: number;
  outputTokens: number;
  usd: number;
}

export function generationCost(promptText: string, completionText: string, price: PriceTable): GenerationCost {
  const inputTokens = estimateTokens(promptText);
  const outputTokens = estimateTokens(completionText);
  const usd =
    (inputTokens * price.inputPerMillionUsd + outputTokens * price.outputPerMillionUsd) / 1_000_000;
  return { inputTokens, outputTokens, usd };
}

/** A cache lookup still pays for embedding the incoming query. */
export function lookupCost(queryText: string, price: PriceTable): number {
  return (estimateTokens(queryText) * price.embeddingPerMillionUsd) / 1_000_000;
}

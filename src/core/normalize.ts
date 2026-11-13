/**
 * Conservative query normalization for exact-match keying.
 *
 * Only case, unicode compatibility, whitespace and punctuation are folded.
 * Anything that could change meaning (word order, numbers, negation) is kept,
 * so two queries share an exact key only when they are the same string up to
 * formatting. Semantic similarity handles the rest - exact keys must never
 * merge two queries that could have different answers.
 */

export function normalizeQuery(text: string): string {
  return text
    .normalize('NFKC')
    .toLowerCase()
    .replace(/[\u2018\u2019]/g, "'")
    .replace(/[\u201C\u201D]/g, '"')
    .replace(/[^\p{L}\p{N}\s]/gu, ' ')
    .replace(/\s+/g, ' ')
    .trim();
}

function fnv1a(input: string): string {
  let h = 0x811c9dc5 >>> 0;
  for (let i = 0; i < input.length; i++) {
    h ^= input.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return (h >>> 0).toString(16).padStart(8, '0');
}

/** Stable exact-match key for a raw query string. */
export function exactKey(text: string): string {
  return fnv1a(normalizeQuery(text));
}

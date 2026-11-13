/**
 * Cacheability guards: queries whose correct answer can change between the
 * moment it is cached and the moment it is served must never enter the cache.
 *
 * These are deterministic lexical guards, not a model. They err toward
 * "do not cache" and every rejection carries a human-readable reason so the
 * decision stays auditable. Two deliberate asymmetries:
 *   - how-to questions stay cacheable even when they mention time or
 *     credentials ("how do I restore to yesterday", "how do I reset my
 *     password"): the ANSWER is the same for everyone on every day
 *   - asking FOR a value of private state ("what is my bill", "how much of
 *     my quota is used") is never cached: the answer differs per account
 *     and per day
 * A production deployment would add a classifier on top; the lexical layer
 * remains as the explainable floor.
 */

export type GuardVerdict = { cacheable: true } | { cacheable: false; reason: string };

const HOW_TO = /^\s*(how|how do|how does|how can|how to|how much does|how many)\b/i;

/** Time-relative phrasing that makes an answer go stale. */
const TIME_RELATIVE =
  /\b(today|tonight|yesterday|tomorrow|right now|currently|latest|upcoming|this (week|month|quarter|year|morning|afternoon|evening)|as of|status page|outage|down right now)\b/i;

/** Nouns whose per-account values must never be served from a shared cache. */
const STATE_NOUN =
  /\b(bill|billing|balance|invoices?|orders?|payments?|payouts?|subscriptions?|plans?|quotas?|usage|limits?|spend|charges?|refunds?|tiers?)\b/i;

/** Secrets: asking to reveal them is never cacheable, however phrased. */
const SECRET_NOUN = /\b(api keys?|passwords?|credentials|secrets?)\b/i;

const FIRST_PERSON = /\b(my|our)\b/i;

/** Phrasings that ask for a value rather than for instructions. */
const REVEAL = /^\s*(what|what's|whats|show|list|tell|display|give|fetch|get|check|how much|how many)\b/i;

/** Commands that change something, as opposed to questions about changing. */
const SIDE_EFFECT_VERB =
  /^\s*(delete|remove|create|update|rotate|change|disable|enable|cancel|upgrade|downgrade|restart|redeploy|deploy|purge|reset|revoke|rename|move|stop|start|terminate)\b/i;

const QUESTION_SHAPE = /\b(how|what|why|when|where|which|can|could|should|does|do|is|are|explain|difference|meaning)\b/i;

const MIN_TOKENS = 2;
const MAX_CHARS = 2000;

export function classifyCacheability(raw: string): GuardVerdict {
  const text = raw.trim();
  if (text.length === 0) return { cacheable: false, reason: 'empty query' };
  if (text.length > MAX_CHARS) return { cacheable: false, reason: `query exceeds ${MAX_CHARS} characters` };
  const tokens = text.split(/\s+/).filter((t) => t.length > 0);
  if (tokens.length < MIN_TOKENS) return { cacheable: false, reason: 'too short to answer reliably' };
  const howTo = HOW_TO.test(text);
  if (TIME_RELATIVE.test(text) && !howTo) {
    return { cacheable: false, reason: 'time-relative: the answer can change between caching and serving' };
  }
  if (FIRST_PERSON.test(text) && STATE_NOUN.test(text) && (REVEAL.test(text) || /\bstatus\b/i.test(text))) {
    return { cacheable: false, reason: 'asks for private account state: the answer differs per account and over time' };
  }
  if (FIRST_PERSON.test(text) && SECRET_NOUN.test(text) && !howTo) {
    return { cacheable: false, reason: 'asks about the caller\'s own credentials: never served from a shared cache' };
  }
  if (SIDE_EFFECT_VERB.test(text) && !QUESTION_SHAPE.test(text)) {
    return { cacheable: false, reason: 'imperative request: performs an action rather than asking a question' };
  }
  return { cacheable: true };
}

/**
 * Workload and dataset types with strict validation, in the same spirit as
 * recall's corpus importer: every file the engine consumes is checked field
 * by field, sizes are capped, ids must be unique, and references must
 * resolve. A malformed dataset fails loudly at load, never silently
 * mid-replay.
 */

export interface CanonicalQA {
  id: string;
  topic: string;
  question: string;
  answer: string;
}

export type Expectation = 'hit-exact' | 'hit-semantic' | 'miss' | 'guard';

export interface WorkloadQuery {
  id: string;
  text: string;
  expect: Expectation;
  /** For hit-semantic: the canonical whose answer must be served. */
  canonicalId?: string;
  /** For miss (first asks): the documentation entry a correct backend answers from. Never used for hit expectations. */
  answerCanonicalId?: string;
  /** For miss: canonicals a naive cache might wrongly serve (near-misses). */
  forbidCanonicalIds?: string[];
  note?: string;
}

export interface LabeledPair {
  id: string;
  a: string;
  b: string;
  /** 1 = same intent (a correct cache hit), 0 = different intent. */
  same: 0 | 1;
  /** Paraphrase group: all pairs sharing a canonical stay in one fold. */
  groupId: string;
}

const MAX_TEXT = 2000;
const MAX_ANSWER = 8000;
const MAX_QUERIES = 10_000;

function isRecord(x: unknown): x is Record<string, unknown> {
  return typeof x === 'object' && x !== null && !Array.isArray(x);
}

function reqString(r: Record<string, unknown>, field: string, max: number): string {
  const v = r[field];
  if (typeof v !== 'string' || v.trim().length === 0) throw new Error(`dataset: "${field}" must be a non-empty string`);
  if (v.length > max) throw new Error(`dataset: "${field}" exceeds ${max} characters`);
  return v;
}

function optString(r: Record<string, unknown>, field: string, max: number): string | undefined {
  const v = r[field];
  if (v === undefined) return undefined;
  if (typeof v !== 'string' || v.length === 0 || v.length > max) throw new Error(`dataset: "${field}" must be a string of at most ${max} characters`);
  return v;
}

export function parseCanonicalSet(input: unknown): CanonicalQA[] {
  if (!isRecord(input) || !Array.isArray(input.canonicals)) throw new Error('dataset: root must have a "canonicals" array');
  const seen = new Set<string>();
  return input.canonicals.map((raw, i) => {
    if (!isRecord(raw)) throw new Error(`dataset: canonical ${i} must be an object`);
    const id = reqString(raw, 'id', 64);
    if (seen.has(id)) throw new Error(`dataset: duplicate canonical id "${id}"`);
    seen.add(id);
    return {
      id,
      topic: reqString(raw, 'topic', 80),
      question: reqString(raw, 'question', MAX_TEXT),
      answer: reqString(raw, 'answer', MAX_ANSWER),
    };
  });
}

const EXPECTATIONS: ReadonlySet<string> = new Set(['hit-exact', 'hit-semantic', 'miss', 'guard']);

export function parseWorkload(input: unknown, canonicalIds: ReadonlySet<string>): WorkloadQuery[] {
  if (!isRecord(input) || !Array.isArray(input.queries)) throw new Error('dataset: root must have a "queries" array');
  if (input.queries.length === 0) throw new Error('dataset: workload is empty');
  if (input.queries.length > MAX_QUERIES) throw new Error(`dataset: workload exceeds ${MAX_QUERIES} queries`);
  const seen = new Set<string>();
  return input.queries.map((raw, i) => {
    if (!isRecord(raw)) throw new Error(`dataset: query ${i} must be an object`);
    const id = reqString(raw, 'id', 64);
    if (seen.has(id)) throw new Error(`dataset: duplicate query id "${id}"`);
    seen.add(id);
    const expect = reqString(raw, 'expect', 32);
    if (!EXPECTATIONS.has(expect)) throw new Error(`dataset: query "${id}" has unknown expect "${expect}"`);
    const out: WorkloadQuery = { id, text: reqString(raw, 'text', MAX_TEXT), expect: expect as Expectation };
    const canonicalId = optString(raw, 'canonicalId', 64);
    if (expect === 'hit-semantic') {
      if (!canonicalId) throw new Error(`dataset: hit-semantic query "${id}" needs canonicalId`);
      if (!canonicalIds.has(canonicalId)) throw new Error(`dataset: query "${id}" references unknown canonical "${canonicalId}"`);
      out.canonicalId = canonicalId;
    } else if (canonicalId !== undefined) {
      throw new Error(`dataset: query "${id}" sets canonicalId but expect is "${expect}"`);
    }
    const answerCanonicalId = optString(raw, 'answerCanonicalId', 64);
    if (answerCanonicalId !== undefined) {
      if (expect !== 'miss') throw new Error(`dataset: query "${id}" sets answerCanonicalId but expect is "${expect}"`);
      if (!canonicalIds.has(answerCanonicalId)) throw new Error(`dataset: query "${id}" answers from unknown canonical "${answerCanonicalId}"`);
      out.answerCanonicalId = answerCanonicalId;
    }
    const forbid = raw['forbidCanonicalIds'];
    if (forbid !== undefined) {
      if (!Array.isArray(forbid) || forbid.some((f) => typeof f !== 'string')) throw new Error(`dataset: query "${id}" forbidCanonicalIds must be string[]`);
      for (const f of forbid as string[]) {
        if (!canonicalIds.has(f)) throw new Error(`dataset: query "${id}" forbids unknown canonical "${f}"`);
      }
      out.forbidCanonicalIds = forbid as string[];
    }
    const note = optString(raw, 'note', 300);
    if (note !== undefined) out.note = note;
    return out;
  });
}

export function parsePairs(input: unknown): LabeledPair[] {
  if (!isRecord(input) || !Array.isArray(input.pairs)) throw new Error('dataset: root must have a "pairs" array');
  if (input.pairs.length < 20) throw new Error('dataset: need at least 20 labeled pairs');
  const seen = new Set<string>();
  return input.pairs.map((raw, i) => {
    if (!isRecord(raw)) throw new Error(`dataset: pair ${i} must be an object`);
    const id = reqString(raw, 'id', 64);
    if (seen.has(id)) throw new Error(`dataset: duplicate pair id "${id}"`);
    seen.add(id);
    const same = raw['same'];
    if (same !== 0 && same !== 1) throw new Error(`dataset: pair "${id}" same must be 0 or 1`);
    return {
      id,
      a: reqString(raw, 'a', MAX_TEXT),
      b: reqString(raw, 'b', MAX_TEXT),
      same: same as 0 | 1,
      groupId: reqString(raw, 'groupId', 64),
    };
  });
}

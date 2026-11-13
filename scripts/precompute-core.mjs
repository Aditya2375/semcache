/**
 * Shared computation behind scripts/precompute.mjs (writes artifacts) and
 * scripts/verify-precomputed.mjs (recomputes and compares). Both must agree
 * exactly on what goes into data/.
 *
 * Embedding choice: query-to-query matching is symmetric, so no retrieval
 * instruction prefix is used (bge's instruction is for asymmetric
 * query->passage retrieval). Both sides are short questions.
 */
import { readFileSync } from 'node:fs';
import { pipeline, env } from '@huggingface/transformers';

env.cacheDir = new URL('../.cache/hf', import.meta.url).pathname;

export const EMBED_MODEL = 'Xenova/bge-small-en-v1.5';
export const EMBED_REVISION = 'ea104dacec62c0de699686887e3f920caeb4f3e3';

export function f32b64(arr) {
  return Buffer.from(Float32Array.from(arr).buffer).toString('base64');
}
export function b64f32(b64) {
  const buf = Buffer.from(b64, 'base64');
  // Buffer.pool: the returned Buffer can sit at a nonzero byteOffset inside a
  // shared 8KB pool - always honor byteOffset/byteLength
  return new Float32Array(buf.buffer, buf.byteOffset, buf.byteLength / 4);
}
export const cosine = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

export function loadDatasets(root) {
  const canonicals = JSON.parse(readFileSync(`${root}/data/helpdesk/canonical.json`, 'utf8')).canonicals;
  const workload = JSON.parse(readFileSync(`${root}/data/helpdesk/workload.json`, 'utf8')).queries;
  const pairs = JSON.parse(readFileSync(`${root}/data/helpdesk/pairs.json`, 'utf8'));
  return { canonicals, workload, pairs };
}

export function collectTexts({ canonicals, workload, pairs }) {
  const texts = new Set();
  for (const c of canonicals) texts.add(c.question);
  for (const q of workload) texts.add(q.text);
  for (const p of [...pairs.train, ...pairs.dev]) {
    texts.add(p.a);
    texts.add(p.b);
  }
  return [...texts].sort();
}

export async function embedTexts(texts, log = () => {}) {
  const embedder = await pipeline('feature-extraction', EMBED_MODEL, { dtype: 'q8', revision: EMBED_REVISION });
  const out = [];
  const batch = 32;
  for (let i = 0; i < texts.length; i += batch) {
    log(`embedding ${i + 1}-${Math.min(i + batch, texts.length)} of ${texts.length}...`);
    const res = await embedder(texts.slice(i, i + batch), { pooling: 'mean', normalize: true });
    for (const row of res.tolist()) out.push(row.map((x) => Math.fround(x)));
  }
  return out;
}

export async function computePrecomputed(root, log = () => {}) {
  const datasets = loadDatasets(root);
  const texts = collectTexts(datasets);
  log(`${texts.length} unique texts`);
  const vectors = await embedTexts(texts, log);
  return {
    version: 1,
    embedModel: EMBED_MODEL,
    embedRevision: EMBED_REVISION,
    dims: vectors[0].length,
    texts,
    vectors: vectors.map(f32b64),
  };
}

/**
 * Calibration features per labeled pair, computed against the reference
 * cache of all canonical questions:
 *   sim1   = cos(a, b)                        (a: incoming query, b: candidate)
 *   margin = sim1 - best cos(a, other-intent canonical question)
 *   size   = log1p(reference cache size)
 *   tokens = word count of a
 * Documented approximation: live margins are against the cache's current
 * contents, not the full canonical set; the reference set stands in for it.
 */
export function computeCalibration(root, vectorsByText, fitLogistic, log = () => {}) {
  const { canonicals, pairs } = loadDatasets(root);
  const canonicalByText = new Map();
  for (const c of canonicals) canonicalByText.set(c.question, c.id);
  const paras = JSON.parse(readFileSync(`${root}/data/helpdesk/paraphrases.json`, 'utf8')).paraphrases;
  for (const p of paras) for (const t of p.texts) canonicalByText.set(t, p.canonicalId);
  const canonicalQuestions = canonicals.map((c) => ({ id: c.id, text: c.question, vec: vectorsByText.get(c.question) }));

  const toSample = (pair) => {
    const aVec = vectorsByText.get(pair.a);
    const bVec = vectorsByText.get(pair.b);
    const bCanonical = canonicalByText.get(pair.b);
    if (!aVec || !bVec || !bCanonical) throw new Error(`calibration: pair ${pair.id} has unmapped text`);
    const sim1 = cosine(aVec, bVec);
    let sim2 = -1;
    for (const c of canonicalQuestions) {
      if (c.id === bCanonical) continue;
      sim2 = Math.max(sim2, cosine(aVec, c.vec));
    }
    const margin = Math.min(1, Math.max(0, sim1 - sim2));
    return { features: [sim1, margin, Math.log1p(canonicalQuestions.length), pair.a.trim().split(/\s+/).length], label: pair.same };
  };

  const train = pairs.train.map(toSample);
  const dev = pairs.dev.map(toSample);
  log(`fitting on ${train.length} pairs, dev ${dev.length}`);
  const weights = fitLogistic(train, { seed: 5 });
  return { weights, train, dev };
}

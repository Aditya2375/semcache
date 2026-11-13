/**
 * CI gate: recomputes the bundled precomputed embedding artifact with the
 * pinned model and fails if the committed file has drifted. Vectors must
 * match at cosine >= 0.995 per row; the gate is 0.995 rather than tighter
 * because ONNX wasm SIMD yields slightly different floats across CPUs - a
 * corrupted or wrong-model artifact lands orders of magnitude lower.
 */
import { readFileSync } from 'node:fs';
import { computePrecomputed, b64f32, cosine } from './precompute-core.mjs';

const root = new URL('..', import.meta.url).pathname;
const stored = JSON.parse(readFileSync(`${root}data/precomputed/helpdesk.bge-small-en-v1.5.json`, 'utf8'));
const fresh = await computePrecomputed(root, () => {});

const fail = (msg) => { console.error(`verify-precomputed: ${msg}`); process.exit(1); };

for (const key of ['version', 'embedModel', 'embedRevision', 'dims']) {
  if (JSON.stringify(stored[key]) !== JSON.stringify(fresh[key])) fail(`metadata mismatch on ${key}`);
}
if (JSON.stringify(stored.texts) !== JSON.stringify(fresh.texts)) fail('text list mismatch');
if (stored.vectors.length !== fresh.vectors.length) fail('vector count mismatch');

let worst = 1;
for (let i = 0; i < stored.vectors.length; i++) {
  const c = cosine(b64f32(stored.vectors[i]), b64f32(fresh.vectors[i]));
  if (c < worst) worst = c;
}
if (worst < 0.995) fail(`worst vector cosine ${worst.toFixed(6)} < 0.995`);
console.log(`verify-precomputed ok: ${stored.vectors.length} vectors, worst cosine ${worst.toFixed(6)}`);

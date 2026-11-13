/**
 * Print the README numbers: replay the bundled workload under each hit
 * policy against the precomputed bge vectors. This is the same runner the
 * live demo uses, so the printed numbers and the on-page numbers agree.
 */
import { readFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { PrecomputedEmbedder, loadPrecomputed } from '../dist/src/core/precomputed.js';
import { SemanticCache } from '../dist/src/core/cache.js';
import { FixedThreshold, MarginPolicy, CalibratedPolicy } from '../dist/src/core/policy.js';
import { SimulatedBackend } from '../dist/src/core/stub.js';
import { runReplay } from '../dist/src/core/replay.js';
import { parseCanonicalSet, parseWorkload } from '../dist/src/core/workload.js';
import { ILLUSTRATIVE_PRICE } from '../dist/src/core/costmodel.js';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const canonicals = parseCanonicalSet(JSON.parse(readFileSync(join(root, 'data/helpdesk/canonical.json'), 'utf8')));
const workload = parseWorkload(
  JSON.parse(readFileSync(join(root, 'data/helpdesk/workload.json'), 'utf8')),
  new Set(canonicals.map((c) => c.id)),
);
const pre = loadPrecomputed(JSON.parse(readFileSync(join(root, 'data/precomputed/helpdesk.bge-small-en-v1.5.json'), 'utf8')));
const calibration = JSON.parse(readFileSync(join(root, 'data/helpdesk/calibration.json'), 'utf8'));

const policies = [
  () => new FixedThreshold(0.75),
  () => new FixedThreshold(0.85),
  () => new FixedThreshold(0.92),
  () => new MarginPolicy(0.75, 0.05),
  () => new CalibratedPolicy(calibration.weights, { wrongAnswerCostMultiple: 20 }),
];

const pct = (x) => (x * 100).toFixed(1);
console.log(`workload: ${workload.length} queries | embedder: ${new PrecomputedEmbedder(pre).id}`);
console.log('');
for (const make of policies) {
  const policy = make();
  const cache = new SemanticCache({ embedder: new PrecomputedEmbedder(pre), policy });
  const backend = new SimulatedBackend(canonicals, ILLUSTRATIVE_PRICE, 23);
  const { summary } = await runReplay(workload, cache, backend);
  console.log(
    [
      policy.id.padEnd(28),
      `hits ${String(summary.hits).padStart(3)}`,
      `precision ${pct(summary.hitPrecision).padStart(5)}% [${pct(summary.hitPrecisionCi[0])}-${pct(summary.hitPrecisionCi[1])}]`,
      `recall ${pct(summary.hitRecall).padStart(5)}% [${pct(summary.hitRecallCi[0])}-${pct(summary.hitRecallCi[1])}]`,
      `wrong-hit ${pct(summary.wrongHitRate)}%`,
      `saved $${summary.costSavedUsd.toFixed(4)} (${summary.costSavedPct.toFixed(1)}%)`,
    ].join('  '),
  );
}

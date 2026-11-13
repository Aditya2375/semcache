/** Regenerate data/precomputed/helpdesk.bge-small-en-v1.5.json and data/helpdesk/calibration.json. */
import { writeFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { computePrecomputed, computeCalibration } from './precompute-core.mjs';
import { fitLogistic } from '../dist/src/core/calibrate.js';
import { b64f32 } from './precompute-core.mjs';

const root = join(dirname(fileURLToPath(import.meta.url)), '..');
const pre = await computePrecomputed(root, console.log);
mkdirSync(join(root, 'data/precomputed'), { recursive: true });
writeFileSync(join(root, 'data/precomputed/helpdesk.bge-small-en-v1.5.json'), JSON.stringify(pre) + '\n');
console.log(`wrote precomputed artifact: ${pre.texts.length} texts x ${pre.dims} dims`);

const vectorsByText = new Map(pre.texts.map((t, i) => [t, b64f32(pre.vectors[i])]));
const { weights, train, dev } = computeCalibration(root, vectorsByText, fitLogistic, console.log);
const calibration = {
  version: 1,
  embedModel: pre.embedModel,
  embedRevision: pre.embedRevision,
  referenceCacheSize: 60,
  weights,
};
writeFileSync(join(root, 'data/helpdesk/calibration.json'), JSON.stringify(calibration, null, 2) + '\n');
console.log('wrote calibration.json; dev sample count', dev.length);

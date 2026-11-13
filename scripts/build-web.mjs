/**
 * Build the self-hosted web bundle into docs/.
 * 1. esbuild src/web/app.ts -> docs/app.js (IIFE, minified).
 * 2. esbuild src/web/worker.ts -> docs/worker.js (IIFE, minified).
 * 3. Copy core dist (worker imports ../core/*.js as a path, resolved at build
 *    time to ../../dist/core/*.js by esbuild? No: worker imports from
 *    '../core/...' relative to src/web, i.e. src/core -> we bundle with esbuild
 *    so imports get inlined; ONNX runtime is loaded from a local file at
 *    runtime).
 * 4. Copy data artifacts, wasm/vendor files, index.html, style.css.
 */
import { build } from 'esbuild';
import { cpSync, copyFileSync, mkdirSync, rmSync, existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const DOCS = join(ROOT, 'docs');
const rm = (p) => rmSync(p, { recursive: true, force: true });

rm(DOCS);
mkdirSync(DOCS, { recursive: true });

const aliasPlugin = {
  name: 'dist-core',
  setup(b) {
    b.onResolve({ filter: /^\.\.\/core\// }, (args) => {
      const rel = args.path.replace('../core/', '');
      return { path: join(ROOT, 'dist', 'src', 'core', rel.endsWith('.js') ? rel : rel + '.js') };
    });
  },
};

for (const [entry, out] of [
  ['src/web/app.ts', 'app.js'],
  ['src/web/worker.ts', 'worker.js'],
]) {
  await build({
    entryPoints: [join(ROOT, entry)],
    outfile: join(DOCS, out),
    bundle: true,
    format: 'iife',
    minify: true,
    target: 'es2022',
    plugins: [aliasPlugin],
    logLevel: 'warning',
  });
}

copyFileSync(join(ROOT, 'src/web/index.html'), join(DOCS, 'index.html'));
copyFileSync(join(ROOT, 'src/web/style.css'), join(DOCS, 'style.css'));

// ONNX runtime wasm files (single-threaded; Pages cannot send COOP/COEP).
mkdirSync(join(DOCS, 'ort'), { recursive: true });
for (const f of readdirSync(join(ROOT, 'ort'))) {
  copyFileSync(join(ROOT, 'ort', f), join(DOCS, 'ort', f));
}
mkdirSync(join(DOCS, 'data/helpdesk'), { recursive: true });
for (const f of ['workload.json', 'canonical.json', 'calibration.json', 'pairs.json']) {
  copyFileSync(join(ROOT, 'data/helpdesk', f), join(DOCS, 'data/helpdesk', f));
}
mkdirSync(join(DOCS, 'data/precomputed'), { recursive: true });
copyFileSync(
  join(ROOT, 'data/precomputed/helpdesk.bge-small-en-v1.5.json'),
  join(DOCS, 'data/precomputed/helpdesk.bge-small-en-v1.5.json'),
);

// Sanity: every runtime path referenced by index.html/worker must exist.
for (const f of [
  'app.js',
  'worker.js',
  'index.html',
  'style.css',
  'ort/ort-wasm-simd-threaded.wasm',
  'ort/ort-wasm-simd-threaded.mjs',
  'data/helpdesk/workload.json',
  'data/helpdesk/canonical.json',
  'data/helpdesk/calibration.json',
  'data/helpdesk/pairs.json',
  'data/precomputed/helpdesk.bge-small-en-v1.5.json',
]) {
  if (!existsSync(join(DOCS, f))) throw new Error(`docs build missing ${f}`);
}
console.log('docs build complete');

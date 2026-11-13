/**
 * Live model adapter for the browser: the same Embedder interface the core
 * defines, backed by ONNX Runtime Web through transformers.js. The model is
 * pinned to an exact revision and downloads lazily - only when the console
 * sees a query outside the bundled artifact. The workload replay, sweep and
 * gallery never need a download.
 *
 * No retrieval instruction prefix: cache lookup is symmetric query-to-query
 * matching, and bge's instruction is meant for asymmetric query-to-passage
 * retrieval.
 */

import type { Embedder } from '../core/embed.js';

export const EMBED_MODEL = 'Xenova/bge-small-en-v1.5';
export const EMBED_REVISION = 'ea104dacec62c0de699686887e3f920caeb4f3e3';

type FeatureExtractionPipeline = (texts: string[], options: Record<string, unknown>) => Promise<{ tolist: () => number[][] }>;

/**
 * Points ONNX Runtime Web at the wasm files the repo ships under docs/ort/
 * instead of its jsdelivr CDN default, so the demo's CSP only ever talks to
 * this site and the pinned Hugging Face hosts. Must run before the first
 * pipeline/session is created.
 */
type OrtWasmEnv = { wasmPaths?: { mjs: string; wasm: string }; proxy?: boolean };
let ortConfigured = false;
async function configureOrt(): Promise<void> {
  if (ortConfigured) return;
  const { env } = await import('@huggingface/transformers');
  const onnx = (env.backends as Record<string, unknown> | undefined)?.['onnx'] as { wasm?: OrtWasmEnv } | undefined;
  if (onnx?.wasm) {
    onnx.wasm.wasmPaths = {
      mjs: './ort/ort-wasm-simd-threaded.mjs',
      wasm: './ort/ort-wasm-simd-threaded.wasm',
    };
    onnx.wasm.proxy = false;
  }
  ortConfigured = true;
}

export class TransformersEmbedder implements Embedder {
  readonly id = `${EMBED_MODEL}@${EMBED_REVISION.slice(0, 8)}`;
  readonly dims = 384;
  private pipe: FeatureExtractionPipeline | null = null;

  async load(onProgress?: (fraction: number) => void): Promise<void> {
    if (this.pipe !== null) return;
    await configureOrt();
    const { pipeline } = await import('@huggingface/transformers');
    const p = await pipeline('feature-extraction', EMBED_MODEL, {
      dtype: 'q8',
      revision: EMBED_REVISION,
      progress_callback: (info: { status?: string; progress?: number }) => {
        if (info.status === 'progress' && typeof info.progress === 'number') onProgress?.(info.progress / 100);
      },
    });
    this.pipe = p as unknown as FeatureExtractionPipeline;
  }

  get loaded(): boolean {
    return this.pipe !== null;
  }

  async embed(texts: string[]): Promise<Float32Array[]> {
    if (this.pipe === null) await this.load();
    const out = await this.pipe!(texts, { pooling: 'mean', normalize: true });
    return out.tolist().map((row) => Float32Array.from(row.map((x) => Math.fround(x))));
  }
}

/**
 * Routes embedding calls: bundled artifact first, live model for anything
 * else. Safe because both are the same model at the same pinned revision -
 * vectors interchange within wasm/CPU float drift, which CI gates at
 * cosine >= 0.995 (scripts/verify-precomputed.mjs).
 */
export class CompositeEmbedder implements Embedder {
  readonly id: string;
  readonly dims: number;
  usedLive = false;
  constructor(
    private readonly pre: { has(text: string): boolean } & Embedder,
    private readonly live: TransformersEmbedder,
  ) {
    this.id = `${pre.id} + ${live.id} (lazy)`;
    this.dims = pre.dims;
  }
  async embed(texts: string[]): Promise<Float32Array[]> {
    const out: Float32Array[] = new Array(texts.length);
    const missing: number[] = [];
    texts.forEach((t, i) => {
      if (this.pre.has(t)) return;
      missing.push(i);
    });
    const preTexts = texts.filter((t) => this.pre.has(t));
    const preVecs = preTexts.length > 0 ? await this.pre.embed(preTexts) : [];
    const preByText = new Map(preTexts.map((t, i) => [t, preVecs[i]!]));
    for (let i = 0; i < texts.length; i++) {
      const v = preByText.get(texts[i]!);
      if (v !== undefined && !missing.includes(i)) out[i] = v;
    }
    if (missing.length > 0) {
      this.usedLive = true;
      const liveVecs = await this.live.embed(missing.map((i) => texts[i]!));
      missing.forEach((textIdx, j) => {
        out[textIdx] = liveVecs[j]!;
      });
    }
    return out;
  }
}

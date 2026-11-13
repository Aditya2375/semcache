/**
 * Loader and adapter for precomputed embedding artifacts. The demo replays
 * the whole workload from vectors produced by the exact model at build time,
 * verified in CI. Texts outside the bundled artifact are rejected, so
 * precomputed data can never silently masquerade as live model output.
 */

import type { Embedder } from './embed.js';

export interface Precomputed {
  version: 1;
  embedModel: string;
  embedRevision: string;
  dims: number;
  texts: string[];
  vectors: string[];
}

function b64ToBytes(b64: string): Uint8Array {
  // Buffer in Node, atob in the browser; the bundle must not depend on Node globals.
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(b64, 'base64');
    return new Uint8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}

function b64ToF32(b64: string, dims: number): Float32Array {
  const bytes = b64ToBytes(b64);
  const vec = new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  if (vec.length !== dims) throw new Error(`precomputed: expected ${dims} dims, got ${vec.length}`);
  return new Float32Array(vec);
}

export function loadPrecomputed(input: unknown): Precomputed {
  const fail = (reason: string): never => {
    throw new Error(`precomputed: ${reason}`);
  };
  if (typeof input !== 'object' || input === null) fail('not an object');
  const p = input as Precomputed;
  if (p.version !== 1) fail('version must be 1');
  if (typeof p.embedModel !== 'string' || typeof p.embedRevision !== 'string') fail('model identity missing');
  if (!Number.isInteger(p.dims) || p.dims <= 0) fail('dims invalid');
  if (!Array.isArray(p.texts) || !Array.isArray(p.vectors)) fail('texts/vectors missing');
  if (p.texts.length !== p.vectors.length) fail('text/vector count mismatch');
  if (p.texts.length === 0) fail('empty artifact');
  for (const b64 of p.vectors) b64ToF32(b64, p.dims);
  return p;
}

export class PrecomputedEmbedder implements Embedder {
  readonly id: string;
  readonly dims: number;
  private readonly byText = new Map<string, Float32Array>();

  constructor(pre: Precomputed) {
    this.dims = pre.dims;
    this.id = `${pre.embedModel}@${pre.embedRevision.slice(0, 8)} (precomputed)`;
    pre.texts.forEach((text, i) => {
      if (this.byText.has(text)) throw new Error(`precomputed: duplicate text at ${i}`);
      this.byText.set(text, b64ToF32(pre.vectors[i]!, pre.dims));
    });
  }

  has(text: string): boolean {
    return this.byText.has(text);
  }

  embed(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(
      texts.map((t) => {
        const vec = this.byText.get(t);
        if (vec === undefined) throw new Error('precomputed embedder: text not in the bundled artifact');
        return vec;
      }),
    );
  }
}

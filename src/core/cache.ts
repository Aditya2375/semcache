/**
 * SemanticCache: cache LLM answers by meaning.
 *
 * Lookup pipeline, in order, each stage auditable in the result:
 *   1. guard       - un-cacheable queries (time-relative, private state,
 *                    commands) never read from or write to the cache
 *   2. exact match - normalized-string key, zero embedding cost
 *   3. semantic    - HNSW nearest neighbours over query embeddings, then a
 *                    hit policy (fixed threshold, margin, or calibrated)
 *
 * Correctness invariants:
 *   - a namespace and a model key partition the cache: tenant A never sees
 *     tenant B's entries, and changing the backing model or prompt version
 *     (the model key) makes old entries unreachable, not stale-servable
 *   - entries expire by TTL and are evicted LRU; deletions are tombstoned in
 *     the ANN index and the graph is rebuilt once tombstones pass a quarter
 *   - a semantic hit always reports which entry it matched and the policy's
 *     own words for why it fired
 */

import type { Embedder } from './embed.js';
import { HnswIndex, type HnswOptions } from './hnsw.js';
import { exactKey, normalizeQuery } from './normalize.js';
import { classifyCacheability } from './guards.js';
import type { HitPolicy } from './policy.js';
import { dot } from './vector.js';

export interface CacheOptions {
  embedder: Embedder;
  policy: HitPolicy;
  maxEntries?: number;
  defaultTtlMs?: number;
  hnsw?: HnswOptions;
  /** Injectable clock for tests and replay. Milliseconds. */
  now?: () => number;
  /** ANN search breadth. */
  ef?: number;
}

export interface CacheEntry<V> {
  id: string;
  namespace: string;
  modelKey: string;
  queryRaw: string;
  queryNorm: string;
  value: V;
  createdAt: number;
  lastHitAt: number;
  hits: number;
  ttlMs: number;
}

export type LookupResult<V> =
  | { kind: 'guard'; reason: string }
  | { kind: 'miss'; reason: string; bestSimilarity?: number }
  | { kind: 'hit-exact'; entry: CacheEntry<V> }
  | { kind: 'hit-semantic'; entry: CacheEntry<V>; similarity: number; p?: number; policyReason: string };

interface Stored<V> extends CacheEntry<V> {
  vector: Float32Array;
  deleted: boolean;
}

let nextEntrySeq = 0;

export class SemanticCache<V> {
  private readonly opts: {
    embedder: Embedder;
    policy: HitPolicy;
    maxEntries: number;
    defaultTtlMs: number;
    ef: number;
    hnsw?: HnswOptions;
  };
  private readonly now: () => number;
  private readonly entries = new Map<string, Stored<V>>();
  /** composite scope key "namespace​modelKey" -> ANN index */
  private readonly indices = new Map<string, HnswIndex>();
  private readonly tombstones = new Map<string, number>();
  /** exact key -> entry ids (only current scope entries are resolvable) */
  private readonly byExactKey = new Map<string, string[]>();
  private lruCounter = 0;

  constructor(options: CacheOptions) {
    this.opts = {
      embedder: options.embedder,
      policy: options.policy,
      maxEntries: options.maxEntries ?? 10_000,
      defaultTtlMs: options.defaultTtlMs ?? 30 * 60 * 1000,
      ef: options.ef ?? 50,
      ...(options.hnsw !== undefined ? { hnsw: options.hnsw } : {}),
    };
    if (this.opts.maxEntries < 1) throw new Error('cache: maxEntries must be >= 1');
    if (this.opts.defaultTtlMs < 1) throw new Error('cache: defaultTtlMs must be >= 1');
    this.now = options.now ?? (() => Date.now());
  }

  get size(): number {
    return this.entries.size;
  }

  private scope(namespace: string, modelKey: string): string {
    return `${namespace} ${modelKey}`;
  }

  private indexFor(scopeKey: string): HnswIndex {
    let idx = this.indices.get(scopeKey);
    if (!idx) {
      const seed = (this.opts.hnsw?.seed ?? 11) >>> 0;
      idx = new HnswIndex(this.opts.embedder.dims, { ...this.opts.hnsw, seed });
      this.indices.set(scopeKey, idx);
      this.tombstones.set(scopeKey, 0);
    }
    return idx;
  }

  async store(namespace: string, modelKey: string, query: string, value: V, ttlMs?: number): Promise<CacheEntry<V> | { skipped: 'guard'; reason: string }> {
    const verdict = classifyCacheability(query);
    if (!verdict.cacheable) return { skipped: 'guard', reason: verdict.reason };

    const norm = normalizeQuery(query);
    const key = exactKey(query);
    const scopeKey = this.scope(namespace, modelKey);
    const existingIds = this.byExactKey.get(key) ?? [];
    for (const id of existingIds) {
      const e = this.entries.get(id);
      if (e && !e.deleted && this.scope(e.namespace, e.modelKey) === scopeKey) {
        // refresh: same question stored again in the same scope renews the entry
        e.value = value;
        e.createdAt = this.now();
        e.lastHitAt = e.createdAt;
        return this.publicEntry(e);
      }
    }

    const [vector] = await this.opts.embedder.embed([query]);
    if (!vector) throw new Error('cache: embedder returned no vector');
    const entry: Stored<V> = {
      id: `e${++nextEntrySeq}`,
      namespace,
      modelKey,
      queryRaw: query,
      queryNorm: norm,
      value,
      createdAt: this.now(),
      lastHitAt: this.now(),
      hits: 0,
      ttlMs: ttlMs ?? this.opts.defaultTtlMs,
      vector,
      deleted: false,
    };
    this.entries.set(entry.id, entry);
    const list = this.byExactKey.get(key) ?? [];
    list.push(entry.id);
    this.byExactKey.set(key, list);
    this.indexFor(scopeKey).add(entry.id, vector);
    if (this.entries.size > this.opts.maxEntries) this.evictLru();
    return this.publicEntry(entry);
  }

  async lookup(namespace: string, modelKey: string, query: string): Promise<LookupResult<V>> {
    const verdict = classifyCacheability(query);
    if (!verdict.cacheable) return { kind: 'guard', reason: verdict.reason };
    const scopeKey = this.scope(namespace, modelKey);
    const key = exactKey(query);

    const exactIds = this.byExactKey.get(key) ?? [];
    for (const id of exactIds) {
      const e = this.entries.get(id);
      if (e && !e.deleted && this.scope(e.namespace, e.modelKey) === scopeKey && !this.isExpired(e)) {
        e.hits += 1;
        e.lastHitAt = this.now();
        return { kind: 'hit-exact', entry: this.publicEntry(e) };
      }
    }

    const index = this.indices.get(scopeKey);
    if (!index || index.size === 0) return { kind: 'miss', reason: 'namespace is empty' };

    const [vector] = await this.opts.embedder.embed([query]);
    if (!vector) throw new Error('cache: embedder returned no vector');
    const probe = index.search(vector, Math.min(8, index.size), this.opts.ef);
    let best: Stored<V> | null = null;
    let bestSim = -1;
    let secondSim: number | null = null;
    for (const hit of probe) {
      const e = this.entries.get(hit.id);
      if (!e || e.deleted || this.isExpired(e)) continue;
      const sim = dot(vector, e.vector);
      if (!best || sim > bestSim) {
        if (best) secondSim = bestSim;
        best = e;
        bestSim = sim;
      } else if (secondSim === null || sim > secondSim) {
        secondSim = sim;
      }
    }
    if (!best) return { kind: 'miss', reason: 'no live entries in namespace' };

    const queryTokens = query.trim().split(/\s+/).length;
    const decision = this.opts.policy.decide({
      sim1: bestSim,
      sim2: secondSim,
      cacheSize: index.size - (this.tombstones.get(scopeKey) ?? 0),
      queryTokens,
    });
    if (!decision.hit) {
      const out: LookupResult<V> = { kind: 'miss', reason: `policy: ${decision.reason}`, bestSimilarity: bestSim };
      return out;
    }
    best.hits += 1;
    best.lastHitAt = this.now();
    const out: LookupResult<V> = {
      kind: 'hit-semantic',
      entry: this.publicEntry(best),
      similarity: bestSim,
      policyReason: decision.reason,
    };
    if (decision.p !== undefined) out.p = decision.p;
    return out;
  }

  /** Fraction of live entries kept, for observability and tests. */
  scopeSize(namespace: string, modelKey: string): number {
    const idx = this.indices.get(this.scope(namespace, modelKey));
    if (!idx) return 0;
    return idx.size - (this.tombstones.get(this.scope(namespace, modelKey)) ?? 0);
  }

  clear(): void {
    this.entries.clear();
    this.indices.clear();
    this.tombstones.clear();
    this.byExactKey.clear();
  }

  private isExpired(e: Stored<V>): boolean {
    return this.now() - e.createdAt > e.ttlMs;
  }

  private evictLru(): void {
    let oldest: Stored<V> | null = null;
    for (const e of this.entries.values()) {
      if (!oldest || e.lastHitAt < oldest.lastHitAt) oldest = e;
    }
    if (!oldest) return;
    this.tombstone(oldest);
  }

  private tombstone(e: Stored<V>): void {
    e.deleted = true;
    this.entries.delete(e.id);
    const scopeKey = this.scope(e.namespace, e.modelKey);
    this.tombstones.set(scopeKey, (this.tombstones.get(scopeKey) ?? 0) + 1);
    const idx = this.indices.get(scopeKey);
    if (idx && (this.tombstones.get(scopeKey) ?? 0) > Math.max(8, idx.size / 4)) {
      this.rebuild(scopeKey);
    }
  }

  private rebuild(scopeKey: string): void {
    const idx = new HnswIndex(this.opts.embedder.dims, { ...this.opts.hnsw, seed: (this.opts.hnsw?.seed ?? 11) >>> 0 });
    for (const e of this.entries.values()) {
      if (!e.deleted && this.scope(e.namespace, e.modelKey) === scopeKey) idx.add(e.id, e.vector);
    }
    this.indices.set(scopeKey, idx);
    this.tombstones.set(scopeKey, 0);
  }

  private publicEntry(e: Stored<V>): CacheEntry<V> {
    const { vector: _vector, deleted: _deleted, ...pub } = e;
    return pub;
  }
}

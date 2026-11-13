import test from 'node:test';
import assert from 'node:assert/strict';
import { SemanticCache } from '../../src/core/cache.js';
import { FixedThreshold } from '../../src/core/policy.js';
import type { Embedder } from '../../src/core/embed.js';
import { normalize } from '../../src/core/vector.js';

/** Handcrafted vectors so similarity outcomes are fully controlled. */
class MapEmbedder implements Embedder {
  readonly id = 'map-embedder';
  readonly dims = 2;
  private readonly map = new Map<string, [number, number]>();
  set(text: string, x: number, y: number): void {
    this.map.set(text, [x, y]);
  }
  embed(texts: string[]): Promise<Float32Array[]> {
    return Promise.resolve(
      texts.map((t) => {
        const v = this.map.get(t);
        if (!v) throw new Error(`map-embedder: unknown text "${t}"`);
        return normalize(new Float32Array(v));
      }),
    );
  }
}

function makeCache(embedder: MapEmbedder, extra: Record<string, unknown> = {}) {
  let clock = 1_000_000;
  const cache = new SemanticCache<string>({
    embedder,
    policy: new FixedThreshold(0.9),
    now: () => clock,
    ...extra,
  });
  return { cache, advance: (ms: number) => (clock += ms) };
}

const E = new MapEmbedder();
E.set('alpha question', 1, 0);
E.set('alpha rephrase', 0.995, 0.1); // cosine ~0.995 vs alpha question
E.set('beta question', 0, 1);
E.set('gamma question', -1, 0);

test('exact hit needs no semantic decision and refreshes stats', async () => {
  const { cache } = makeCache(E);
  await cache.store('ns', 'm1', 'alpha question', 'A1');
  const res = await cache.lookup('ns', 'm1', 'Alpha Question?!');
  assert.equal(res.kind, 'hit-exact');
  if (res.kind === 'hit-exact') {
    assert.equal(res.entry.value, 'A1');
    assert.equal(res.entry.hits, 1);
  }
});

test('semantic hit fires above threshold, miss below', async () => {
  const { cache } = makeCache(E);
  await cache.store('ns', 'm1', 'alpha question', 'A1');
  const hit = await cache.lookup('ns', 'm1', 'alpha rephrase');
  assert.equal(hit.kind, 'hit-semantic');
  if (hit.kind === 'hit-semantic') {
    assert.ok(hit.similarity > 0.9);
    assert.equal(hit.entry.value, 'A1');
    assert.ok(hit.policyReason.length > 0);
  }
  const miss = await cache.lookup('ns', 'm1', 'gamma question');
  assert.equal(miss.kind, 'miss');
  if (miss.kind === 'miss') assert.ok(miss.reason.startsWith('policy:'));
});

test('namespaces and model keys isolate entries', async () => {
  const { cache } = makeCache(E);
  await cache.store('tenant-a', 'm1', 'alpha question', 'A1');
  const otherNs = await cache.lookup('tenant-b', 'm1', 'alpha question');
  assert.equal(otherNs.kind, 'miss');
  const otherModel = await cache.lookup('tenant-a', 'm2', 'alpha question');
  assert.equal(otherModel.kind, 'miss');
  const sameScope = await cache.lookup('tenant-a', 'm1', 'alpha question');
  assert.equal(sameScope.kind, 'hit-exact');
});

test('TTL expiry makes entries unservable', async () => {
  const { cache, advance } = makeCache(E, { defaultTtlMs: 1000 });
  await cache.store('ns', 'm1', 'alpha question', 'A1');
  advance(2000);
  const res = await cache.lookup('ns', 'm1', 'alpha question');
  assert.equal(res.kind, 'miss');
});

test('re-storing the same query refreshes the entry instead of duplicating', async () => {
  const { cache, advance } = makeCache(E, { defaultTtlMs: 1000 });
  await cache.store('ns', 'm1', 'alpha question', 'A1');
  advance(900);
  await cache.store('ns', 'm1', 'alpha question', 'A1-new');
  advance(900);
  const res = await cache.lookup('ns', 'm1', 'alpha question');
  assert.equal(res.kind, 'hit-exact');
  if (res.kind === 'hit-exact') assert.equal(res.entry.value, 'A1-new');
  assert.equal(cache.size, 1);
});

test('LRU eviction drops the oldest entry and keeps the rest findable', async () => {
  const { cache, advance } = makeCache(E, { maxEntries: 2 });
  await cache.store('ns', 'm1', 'alpha question', 'A1');
  advance(10);
  await cache.store('ns', 'm1', 'beta question', 'B1');
  advance(10);
  await cache.lookup('ns', 'm1', 'alpha question'); // touch alpha so beta is oldest
  advance(10);
  await cache.store('ns', 'm1', 'gamma question', 'G1'); // evicts beta
  assert.equal(cache.size, 2);
  assert.equal((await cache.lookup('ns', 'm1', 'beta question')).kind, 'miss');
  assert.equal((await cache.lookup('ns', 'm1', 'alpha question')).kind, 'hit-exact');
  assert.equal((await cache.lookup('ns', 'm1', 'gamma question')).kind, 'hit-exact');
});

test('guarded queries are never stored or served', async () => {
  const { cache } = makeCache(E);
  const stored = await cache.store('ns', 'm1', 'what is my current bill', 'BILL');
  assert.ok('skipped' in stored && stored.skipped === 'guard');
  const res = await cache.lookup('ns', 'm1', 'what is my current bill');
  assert.equal(res.kind, 'guard');
  assert.equal(cache.size, 0);
});

test('lookup on an empty cache misses cleanly', async () => {
  const { cache } = makeCache(E);
  const res = await cache.lookup('ns', 'm1', 'alpha question');
  assert.equal(res.kind, 'miss');
});

test('heavy eviction triggers rebuild without losing live entries', async () => {
  const { cache, advance } = makeCache(E, { maxEntries: 4 });
  const texts = ['alpha question', 'beta question', 'gamma question', 'alpha rephrase'];
  for (const t of texts) {
    await cache.store('ns', 'm1', t, `v:${t}`);
    advance(5);
  }
  // force 12 evictions one at a time by re-storing beyond capacity
  for (let i = 0; i < 12; i++) {
    E.set(`filler ${i}`, Math.cos(i), Math.sin(i));
    await cache.store('ns', 'm1', `filler ${i}`, `f${i}`);
    advance(5);
  }
  assert.equal(cache.size, 4);
  // the most recently used entries survive and are findable after rebuilds
  const res = await cache.lookup('ns', 'm1', 'filler 11');
  assert.equal(res.kind, 'hit-exact');
});

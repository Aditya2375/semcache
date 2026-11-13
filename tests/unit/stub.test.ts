import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { SimulatedBackend } from '../../src/core/stub.js';
import { parseCanonicalSet } from '../../src/core/workload.js';
import { ILLUSTRATIVE_PRICE } from '../../src/core/costmodel.js';

const canonicals = parseCanonicalSet(JSON.parse(readFileSync(join(process.cwd(), 'data/helpdesk/canonical.json'), 'utf8')));

test('backend returns the documented answer for a matched canonical', () => {
  const be = new SimulatedBackend(canonicals, ILLUSTRATIVE_PRICE);
  const r = be.generate('anything', 'c01');
  assert.equal(r.text, canonicals.find((c) => c.id === 'c01')!.answer);
  assert.equal(r.canonicalId, 'c01');
  assert.ok(r.latencyMs > 0);
  assert.ok(r.cost.usd > 0);
});

test('backend is deterministic for a fixed seed', () => {
  const a = new SimulatedBackend(canonicals, ILLUSTRATIVE_PRICE, 9);
  const b = new SimulatedBackend(canonicals, ILLUSTRATIVE_PRICE, 9);
  const ra = a.generate('how do backups work', 'c21');
  const rb = b.generate('how do backups work', 'c21');
  assert.equal(ra.latencyMs, rb.latencyMs);
  assert.equal(ra.cost.usd, rb.cost.usd);
});

test('unmatched queries fall back to the overlap composer', () => {
  const be = new SimulatedBackend(canonicals, ILLUSTRATIVE_PRICE);
  const r = be.generate('how do I enable private networking endpoints', null);
  assert.ok(r.text.length > 0);
  assert.equal(r.canonicalId, 'c44');
  const gibberish = be.generate('zqwx vbnm kjhgf', null);
  assert.equal(gibberish.canonicalId, null);
});

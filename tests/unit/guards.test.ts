import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { classifyCacheability } from '../../src/core/guards.js';

const read = (p: string) => JSON.parse(readFileSync(join(process.cwd(), p), 'utf8'));

test('every guard seed query is rejected with a reason', () => {
  const seeds = read('data/helpdesk/guards.json').queries as Array<{ text: string }>;
  assert.ok(seeds.length >= 16);
  for (const q of seeds) {
    const v = classifyCacheability(q.text);
    assert.equal(v.cacheable, false, `should reject: ${q.text}`);
    if (!v.cacheable) assert.ok(v.reason.length > 5, 'reason must be human-readable');
  }
});

test('every canonical question and paraphrase is cacheable', () => {
  const canonicals = read('data/helpdesk/canonical.json').canonicals as Array<{ question: string }>;
  const paras = read('data/helpdesk/paraphrases.json').paraphrases as Array<{ texts: string[] }>;
  const nearmiss = read('data/helpdesk/nearmiss.json').queries as Array<{ text: string }>;
  const all = [...canonicals.map((c) => c.question), ...paras.flatMap((p) => p.texts), ...nearmiss.map((n) => n.text)];
  assert.ok(all.length > 200);
  for (const text of all) {
    const v = classifyCacheability(text);
    assert.ok(v.cacheable, `should accept: ${text} (${v.cacheable ? '' : v.reason})`);
  }
});

test('first-person how-to questions stay cacheable; state questions do not', () => {
  assert.ok(classifyCacheability('How do I reset my password?').cacheable);
  assert.ok(classifyCacheability('How do I rotate an API key without downtime?').cacheable);
  assert.equal(classifyCacheability('What is my current bill?').cacheable, false);
  assert.equal(classifyCacheability('Show me my usage').cacheable, false);
});

test('imperatives are rejected only without question shape', () => {
  assert.equal(classifyCacheability('Delete my database').cacheable, false);
  assert.ok(classifyCacheability('How do I delete a database?').cacheable);
});

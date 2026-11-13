import test from 'node:test';
import assert from 'node:assert/strict';
import { exactKey, normalizeQuery } from '../../src/core/normalize.js';

test('normalize folds case, punctuation and whitespace', () => {
  assert.equal(normalizeQuery('  How  do I   Connect?! '), 'how do i connect');
  assert.equal(normalizeQuery('NimbusDB.'), normalizeQuery('nimbusdb'));
});

test('normalize keeps meaning-bearing tokens intact', () => {
  assert.notEqual(normalizeQuery('delete a table'), normalizeQuery('delete a database'));
  assert.notEqual(normalizeQuery('ports 7432'), normalizeQuery('ports 7433'));
  assert.notEqual(normalizeQuery('do not delete'), normalizeQuery('do delete'));
});

test('exactKey is stable and order-sensitive', () => {
  assert.equal(exactKey('How do I connect?'), exactKey('how do i connect'));
  assert.notEqual(exactKey('rotate my keys'), exactKey('my keys rotate'));
});

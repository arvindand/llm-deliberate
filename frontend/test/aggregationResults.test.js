import test from 'node:test'
import assert from 'node:assert/strict'
import { outcomeLabel, scoreRows, scoreWidth } from '../src/aggregationResults.js'

const labels = {
  a: 'same-model · round 1 · a',
  b: 'same-model · round 1 · b',
  c: 'same-model · round 2 · c',
}

test('score rows keep samples and rounds distinct and give equal scores equal ranks', () => {
  assert.deepEqual(scoreRows({ a: 1, b: 1, c: 0 }, labels), [
    { id: 'a', label: labels.a, score: 1, rank: 1 },
    { id: 'b', label: labels.b, score: 1, rank: 1 },
    { id: 'c', label: labels.c, score: 0, rank: 3 },
  ])
})

test('outcomes display tied response labels and unresolved counts separately', () => {
  assert.equal(outcomeLabel({ status: 'tie', winner_ids: ['a', 'b'] }, labels), `Tie: ${labels.a}; ${labels.b}`)
  assert.equal(outcomeLabel({ status: 'winner', winner_ids: ['c'] }, labels), labels.c)
  assert.equal(outcomeLabel({ status: 'unresolved', winner_ids: [] }, labels), 'Unresolved: elimination tie')
})

test('zero-score ties have finite score bars', () => {
  assert.equal(scoreWidth(0, 0), 0)
  assert.equal(scoreWidth(2, 4), 50)
  assert.equal(scoreWidth(0, 1), 0)
})

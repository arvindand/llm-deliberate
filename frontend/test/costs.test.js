import assert from 'node:assert/strict'
import test from 'node:test'

import { estimateTokenCost } from '../src/costs.js'

test('estimates OpenRouter prices expressed in USD per token', () => {
  const cost = estimateTokenCost(
    { prompt: 0.0000025, completion: 0.00001 },
    1000,
    500,
  )

  assert.equal(cost, 0.0075)
})

test('handles missing pricing safely', () => {
  assert.equal(estimateTokenCost(null, 1000, 500), 0)
})

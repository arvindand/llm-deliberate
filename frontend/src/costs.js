export function estimateTokenCost(pricing, inputTokens = 0, outputTokens = 0) {
  if (!pricing) return 0

  const promptPrice = Number(pricing.prompt) || 0
  const completionPrice = Number(pricing.completion) || 0

  return (inputTokens * promptPrice) + (outputTokens * completionPrice)
}

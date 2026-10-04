export function responseLabel(responseId, labels = {}) {
  return labels[responseId] ?? responseId
}

export function outcomeLabel(result, labels = {}) {
  if (result.status === 'unresolved') return 'Unresolved: elimination tie'
  if (result.status === 'no_ballots') return 'No eligible ballots'
  const winners = result.winner_ids ?? (result.winner ? [result.winner] : [])
  const names = winners.map(id => responseLabel(id, labels))
  return names.length > 1 ? `Tie: ${names.join('; ')}` : names[0] ?? 'No winner'
}

export function scoreRows(scores, labels = {}) {
  const rows = Object.entries(scores).sort((a, b) => b[1] - a[1])
  let rank = 0
  let rankScore
  return rows.map(([id, score], index) => {
    const tolerance = Math.max(1e-9, Math.max(Math.abs(score), Math.abs(rankScore ?? 0)) * 1e-9)
    if (rankScore === undefined || Math.abs(score - rankScore) > tolerance) {
      rank = index + 1
      rankScore = score
    }
    return { id, score, rank, label: responseLabel(id, labels) }
  })
}

export function scoreWidth(score, maxScore) {
  return maxScore > 0 ? Math.max(0, Math.min(100, score / maxScore * 100)) : 0
}

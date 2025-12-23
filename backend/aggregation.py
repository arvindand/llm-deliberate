"""
Aggregation algorithms for LLM deliberation.

This module implements various voting and ranking aggregation methods
from social choice theory, adapted for LLM council deliberation.

References:
- Borda Count: https://en.wikipedia.org/wiki/Borda_count
- Condorcet/Copeland: https://en.wikipedia.org/wiki/Copeland%27s_method
- Ranked Pairs (Tideman): https://en.wikipedia.org/wiki/Ranked_pairs
"""
from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .models import Ranking


@dataclass
class AggregationResult:
    """Result of an aggregation computation."""
    method: str
    scores: dict[str, float]
    winner: str
    details: Optional[dict] = None


def plurality(rankings: list[Ranking], candidates: list[str]) -> dict[str, float]:
    """
    Simple plurality voting - count first-place votes.
    
    Each ranking's top choice gets 1 point.
    Winner is the candidate with the most first-place votes.
    
    Pros: Simple, intuitive
    Cons: Ignores all preference information beyond first choice
    """
    first_places = [r.rankings[0] for r in rankings if r.rankings]
    counts = Counter(first_places)
    
    # Ensure all candidates appear in results
    return {c: float(counts.get(c, 0)) for c in candidates}


def borda_count(rankings: list[Ranking], candidates: list[str]) -> dict[str, float]:
    """
    Borda Count - positional voting system.
    
    For n candidates:
    - 1st place gets n-1 points
    - 2nd place gets n-2 points
    - ...
    - Last place gets 0 points
    
    Pros: Uses full ranking information, tends to elect broadly acceptable candidates
    Cons: Can be manipulated by strategic nomination of candidates
    
    Research note: "The Borda count gives an approximately maximum likelihood 
    estimator of the best candidate" (Van Newenhizen, 1992)
    """
    n = len(candidates)
    scores = dict.fromkeys(candidates, 0.0)
    
    for ranking in rankings:
        for position, candidate_id in enumerate(ranking.rankings):
            if candidate_id in scores:
                # n-1 for first place, n-2 for second, etc.
                scores[candidate_id] += (n - 1 - position)
    
    return scores


def weighted_borda(rankings: list[Ranking], candidates: list[str]) -> dict[str, float]:
    """
    Confidence-Weighted Borda Count.
    
    Same as Borda, but each ranking is weighted by the judge's confidence score.
    
    Research note: "CW-Borda tends to be more adequate than standard Borda 
    as group size and sensitivity of confidence weighting increased"
    (Wisdom of crowds research, 2020)
    """
    n = len(candidates)
    scores = dict.fromkeys(candidates, 0.0)
    
    for ranking in rankings:
        weight = ranking.confidence
        for position, candidate_id in enumerate(ranking.rankings):
            if candidate_id in scores:
                scores[candidate_id] += (n - 1 - position) * weight
    
    return scores


def _count_pairwise_preferences(rankings: list[Ranking], c1: str, c2: str) -> tuple[int, int]:
    """Count how many judges prefer c1 over c2 and vice versa."""
    c1_preferred = 0
    c2_preferred = 0
    
    for ranking in rankings:
        if c1 in ranking.rankings and c2 in ranking.rankings:
            pos1 = ranking.rankings.index(c1)
            pos2 = ranking.rankings.index(c2)
            if pos1 < pos2:  # Lower position = better
                c1_preferred += 1
            elif pos2 < pos1:
                c2_preferred += 1
    
    return c1_preferred, c2_preferred


def _award_pairwise_points(wins: dict[str, float], c1: str, c2: str, 
                           c1_preferred: int, c2_preferred: int) -> None:
    """Award points based on pairwise comparison."""
    if c1_preferred > c2_preferred:
        wins[c1] += 1
    elif c2_preferred > c1_preferred:
        wins[c2] += 1
    else:
        # Tie: half point each
        wins[c1] += 0.5
        wins[c2] += 0.5


def copeland_score(rankings: list[Ranking], candidates: list[str]) -> dict[str, float]:
    """
    Copeland's Method (simplified Condorcet).
    
    For each pair of candidates, count who is preferred by more judges.
    Score = number of pairwise victories.
    
    A Condorcet winner (beats everyone head-to-head) will have score = n-1.
    
    Pros: Satisfies Condorcet criterion, resistant to spoilers
    Cons: Often produces ties when there's no clear Condorcet winner
    """
    wins = dict.fromkeys(candidates, 0.0)
    
    # Compare each pair
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i+1:]:
            c1_preferred, c2_preferred = _count_pairwise_preferences(rankings, c1, c2)
            _award_pairwise_points(wins, c1, c2, c1_preferred, c2_preferred)
    
    return wins



def _build_preference_matrix(rankings: list[Ranking], candidates: list[str]) -> dict[str, dict[str, int]]:
    """Build pairwise preference matrix from rankings."""
    pref = {c1: dict.fromkeys(candidates, 0) for c1 in candidates}
    
    for ranking in rankings:
        for i, c1 in enumerate(ranking.rankings):
            for c2 in ranking.rankings[i+1:]:
                if c1 in pref and c2 in pref[c1]:
                    pref[c1][c2] += 1
    
    return pref


def _calculate_margin_pairs(candidates: list[str], pref: dict[str, dict[str, int]]) -> list[tuple[str, str, int]]:
    """Calculate margins and create pairs for ranked pairs method."""
    pairs = []
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i+1:]:
            margin = pref[c1][c2] - pref[c2][c1]
            if margin > 0:
                pairs.append((c1, c2, margin))
            elif margin < 0:
                pairs.append((c2, c1, -margin))
    return pairs


def _creates_cycle(locked: set[tuple[str, str]], winner: str, loser: str) -> bool:
    """Check if adding winner->loser would create a cycle."""
    # BFS to see if loser can reach winner through locked pairs
    visited = set()
    queue = [loser]
    while queue:
        current = queue.pop(0)
        if current == winner:
            return True
        if current in visited:
            continue
        visited.add(current)
        for w, l in locked:
            if w == current:
                queue.append(l)
    return False


def _lock_pairs_without_cycles(pairs: list[tuple[str, str, int]]) -> set[tuple[str, str]]:
    """Lock pairs in order, avoiding cycles."""
    locked = set()
    for winner, loser, margin in pairs:
        if not _creates_cycle(locked, winner, loser):
            locked.add((winner, loser))
    return locked


def ranked_pairs(rankings: list[Ranking], candidates: list[str]) -> dict[str, float]:
    """
    Ranked Pairs (Tideman method).
    
    1. Calculate margin of victory for each pairwise comparison
    2. Sort pairs by margin (strongest to weakest)
    3. Lock in pairs in order, skipping any that would create a cycle
    4. Winner is the candidate who is not defeated by anyone in locked pairs
    
    Pros: Condorcet method that handles cycles gracefully
    Cons: More complex to explain and implement
    
    For simplicity, we return a score based on the final ordering.
    """
    # Build pairwise preference matrix
    pref = _build_preference_matrix(rankings, candidates)
    
    # Calculate margins and create pairs
    pairs = _calculate_margin_pairs(candidates, pref)
    
    # Sort by margin (strongest first)
    pairs.sort(key=lambda x: x[2], reverse=True)
    
    # Lock pairs, avoiding cycles
    locked = _lock_pairs_without_cycles(pairs)
    
    # Score based on locked victories
    scores = dict.fromkeys(candidates, 0.0)
    for winner, loser in locked:
        scores[winner] += 1
    
    return scores


def get_winner(scores: dict[str, float]) -> str:
    """Get the candidate with the highest score."""
    if not scores:
        return ""
    return max(scores.keys(), key=lambda k: scores[k])


def get_ranking(scores: dict[str, float]) -> list[str]:
    """Get candidates sorted by score (best first)."""
    return sorted(scores.keys(), key=lambda k: scores[k], reverse=True)


# === Analysis Utilities ===


def _calculate_pairwise_agreement(r1: Ranking, r2: Ranking, candidates: list[str]) -> float:
    """Calculate Kendall tau-like agreement between two rankings."""
    agreements = 0
    comparisons = 0
    
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i+1:]:
            if (c1 in r1.rankings and c2 in r1.rankings and 
                c1 in r2.rankings and c2 in r2.rankings):
                pos1_r1 = r1.rankings.index(c1)
                pos2_r1 = r1.rankings.index(c2)
                pos1_r2 = r2.rankings.index(c1)
                pos2_r2 = r2.rankings.index(c2)
                
                # Do they agree on relative ordering?
                r1_prefers_c1 = pos1_r1 < pos2_r1
                r2_prefers_c1 = pos1_r2 < pos2_r2
                
                if r1_prefers_c1 == r2_prefers_c1:
                    agreements += 1
                comparisons += 1
    
    return agreements / comparisons if comparisons > 0 else 0.0


def agreement_matrix(rankings: list[Ranking], candidates: list[str]) -> dict[str, dict[str, float]]:
    """
    Calculate pairwise agreement between judges.
    
    Returns a matrix where [judge1][judge2] = correlation of their rankings.
    """
    judges = [r.judge for r in rankings]
    matrix = {j1: dict.fromkeys(judges, 0.0) for j1 in judges}
    
    for r1 in rankings:
        for r2 in rankings:
            if r1.id == r2.id:
                matrix[r1.judge][r2.judge] = 1.0
            else:
                matrix[r1.judge][r2.judge] = _calculate_pairwise_agreement(r1, r2, candidates)
    
    return matrix


def method_agreement(rankings: list[Ranking], candidates: list[str]) -> dict[str, str]:
    """
    Check which aggregation methods agree on the winner.
    
    Returns dict mapping method name to winner.
    """
    methods = {
        "plurality": plurality,
        "borda": borda_count,
        "weighted_borda": weighted_borda,
        "copeland": copeland_score,
        "ranked_pairs": ranked_pairs
    }
    
    return {
        name: get_winner(method(rankings, candidates))
        for name, method in methods.items()
    }


def diversity_score(rankings: list[Ranking], candidates: list[str]) -> float:
    """
    Calculate how diverse the rankings are.
    
    Returns 0-1 where 0 = perfect agreement, 1 = maximum disagreement.
    
    This is important for wisdom of crowds - too much agreement
    might indicate herding/consensus bias.
    """
    if len(rankings) < 2:
        return 0.0
    
    matrix = agreement_matrix(rankings, candidates)
    judges = list(matrix.keys())
    
    total_agreement = 0
    count = 0
    
    for i, j1 in enumerate(judges):
        for j2 in judges[i+1:]:
            total_agreement += matrix[j1][j2]
            count += 1
    
    if count == 0:
        return 0.0
    
    avg_agreement = total_agreement / count
    return 1.0 - avg_agreement  # Invert so higher = more diverse

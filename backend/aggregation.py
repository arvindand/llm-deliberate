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

from .models import Ranking


@dataclass
class AggregationResult:
    """Result of an aggregation computation."""

    method: str
    scores: dict[str, float]
    winner: str
    details: dict | None = None


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
                scores[candidate_id] += n - 1 - position

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


def _award_pairwise_points(
    wins: dict[str, float], c1: str, c2: str, c1_preferred: int, c2_preferred: int
) -> None:
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
        for c2 in candidates[i + 1 :]:
            c1_preferred, c2_preferred = _count_pairwise_preferences(rankings, c1, c2)
            _award_pairwise_points(wins, c1, c2, c1_preferred, c2_preferred)

    return wins


def _build_preference_matrix(
    rankings: list[Ranking], candidates: list[str]
) -> dict[str, dict[str, int]]:
    """Build pairwise preference matrix from rankings."""
    pref = {c1: dict.fromkeys(candidates, 0) for c1 in candidates}

    for ranking in rankings:
        for i, c1 in enumerate(ranking.rankings):
            for c2 in ranking.rankings[i + 1 :]:
                if c1 in pref and c2 in pref[c1]:
                    pref[c1][c2] += 1

    return pref


def _calculate_margin_pairs(
    candidates: list[str], pref: dict[str, dict[str, int]]
) -> list[tuple[str, str, int]]:
    """Calculate margins and create pairs for ranked pairs method."""
    pairs: list[tuple[str, str, int]] = []
    for i, c1 in enumerate(candidates):
        for c2 in candidates[i + 1 :]:
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
        for locked_winner, locked_loser in locked:
            if locked_winner == current:
                queue.append(locked_loser)
    return False


def _lock_pairs_without_cycles(pairs: list[tuple[str, str, int]]) -> set[tuple[str, str]]:
    """Lock pairs in order, avoiding cycles."""
    locked: set[tuple[str, str]] = set()
    for winner, loser, _margin in pairs:
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
    for winner, _loser in locked:
        scores[winner] += 1

    return scores


def schulze_method(rankings: list[Ranking], candidates: list[str]) -> dict[str, float]:
    """Schulze method (Schwartz Sequential Dropping).

    A Condorcet method that finds the candidate who beats all others via
    the strongest path of pairwise victories. Handles cycles by finding
    the strongest paths through the preference graph.

    Algorithm:
    1. Build pairwise preference matrix
    2. Calculate strongest paths between all pairs using Floyd-Warshall
    3. Score candidates by counting how many they beat via strongest path

    Pros: Always elects Condorcet winner if exists, handles cycles well
    Cons: Complex to explain to non-technical users
    """
    n = len(candidates)
    if n == 0:
        return {}

    pref = _build_preference_matrix(rankings, candidates)

    idx_to_cand = dict(enumerate(candidates))
    strength = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                ci, cj = idx_to_cand[i], idx_to_cand[j]
                if pref[ci][cj] > pref[cj][ci]:
                    strength[i][j] = pref[ci][cj]

    for i in range(n):
        for j in range(n):
            if i != j:
                for k in range(n):
                    if i != k and j != k:
                        strength[j][k] = max(strength[j][k], min(strength[j][i], strength[i][k]))

    scores = dict.fromkeys(candidates, 0.0)
    for i in range(n):
        for j in range(n):
            if i != j:
                ci, cj = idx_to_cand[i], idx_to_cand[j]
                if strength[i][j] > strength[j][i]:
                    scores[ci] += 1

    return scores


def stv_instant_runoff(rankings: list[Ranking], candidates: list[str]) -> dict[str, float]:
    """Single Transferable Vote / Instant Runoff Voting.

    Simulates multiple rounds of voting where the candidate with fewest
    first-place votes is eliminated, and their votes transfer to voters'
    next choices. Continues until one candidate has majority.

    Algorithm:
    1. Count first-place votes
    2. If candidate has majority, they win
    3. Else eliminate candidate with fewest first-place votes
    4. Transfer their votes to next preference
    5. Repeat until winner found

    Pros: Reduces strategic voting, ensures majority support
    Cons: Non-monotonic (getting more votes can hurt), complex tallying
    """
    if not candidates or not rankings:
        return dict.fromkeys(candidates, 0.0)

    active_candidates = set(candidates)
    round_num = 0
    max_rounds = len(candidates)

    while len(active_candidates) > 1 and round_num < max_rounds:
        round_num += 1
        first_place_counts = dict.fromkeys(active_candidates, 0)

        for ranking in rankings:
            for candidate_id in ranking.rankings:
                if candidate_id in active_candidates:
                    first_place_counts[candidate_id] += 1
                    break

        total_votes = sum(first_place_counts.values())
        if total_votes == 0:
            break

        for cand, count in first_place_counts.items():
            if count > total_votes / 2:
                scores = dict.fromkeys(candidates, 0.0)
                scores[cand] = float(len(candidates))
                for other_c in active_candidates:
                    if other_c != cand:
                        scores[other_c] = float(len(candidates) - round_num)
                for elim_c in set(candidates) - active_candidates:
                    scores[elim_c] = 0.0
                return scores

        min_votes = min(first_place_counts.values())
        to_eliminate = [c for c, count in first_place_counts.items() if count == min_votes]
        for c in to_eliminate:
            active_candidates.remove(c)

    if len(active_candidates) == 1:
        winner = list(active_candidates)[0]
        return {c: (float(len(candidates)) if c == winner else 0.0) for c in candidates}

    return dict.fromkeys(candidates, 0.0)


def approval_voting(
    rankings: list[Ranking], candidates: list[str], threshold: int = 2
) -> dict[str, float]:
    """Approval voting - voters approve/disapprove each candidate.

    Since we have ranked ballots, we approximate approval by treating
    the top N candidates in each ranking as "approved" (default N=2).

    Args:
        rankings: List of Ranking objects
        candidates: List of candidate IDs
        threshold: Number of top candidates to count as "approved" per ballot

    Algorithm:
    1. For each ballot, count top N candidates as approved
    2. Sum approval counts across all ballots
    3. Candidate with most approvals wins

    Pros: Simple, reduces strategic voting, allows expressing support for multiple
    Cons: Doesn't capture preference intensity beyond approve/disapprove
    """
    approvals = dict.fromkeys(candidates, 0.0)

    for ranking in rankings:
        approved_count = 0
        for candidate_id in ranking.rankings:
            if candidate_id in candidates and approved_count < threshold:
                approvals[candidate_id] += 1
                approved_count += 1
            if approved_count >= threshold:
                break

    return approvals


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
        for c2 in candidates[i + 1 :]:
            if c1 in r1.rankings and c2 in r1.rankings and c1 in r2.rankings and c2 in r2.rankings:
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
    judge_counts = Counter(ranking.judge for ranking in rankings)
    judge_seen: Counter[str] = Counter()
    labels: list[str] = []
    for ranking in rankings:
        judge_seen[ranking.judge] += 1
        label = ranking.judge
        if judge_counts[ranking.judge] > 1:
            label = f"{ranking.judge} #{judge_seen[ranking.judge]}"
        labels.append(label)

    matrix = {label: dict.fromkeys(labels, 0.0) for label in labels}

    for index1, r1 in enumerate(rankings):
        for index2, r2 in enumerate(rankings):
            label1 = labels[index1]
            label2 = labels[index2]
            if r1.id == r2.id:
                matrix[label1][label2] = 1.0
            else:
                matrix[label1][label2] = _calculate_pairwise_agreement(r1, r2, candidates)

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
        "ranked_pairs": ranked_pairs,
        "schulze": schulze_method,
        "stv": stv_instant_runoff,
        "approval": approval_voting,
    }

    return {name: get_winner(method(rankings, candidates)) for name, method in methods.items()}


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
        for j2 in judges[i + 1 :]:
            total_agreement += matrix[j1][j2]
            count += 1

    if count == 0:
        return 0.0

    avg_agreement = total_agreement / count
    return 1.0 - avg_agreement  # Invert so higher = more diverse

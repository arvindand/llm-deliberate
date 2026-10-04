import pytest

from backend.aggregation import (
    AGGREGATION_METHODS,
    aggregate_rankings,
    get_winner,
    get_winners,
    ranked_pairs,
)
from backend.models import Ranking


def ballots(*orders):
    return [Ranking(judge=f"judge-{i}", rankings=list(order)) for i, order in enumerate(orders)]


def test_ranked_pairs_selects_locked_graph_root_not_most_outgoing_wins():
    # Locked edges: D->A, A->B, A->C, B->C. D is undefeated even though
    # A has two outgoing victories and D has only one.
    rankings = ballots("DACB", "BCDA", "DACB", "ABCD", "BCDA")

    for candidates in [list("ABCD"), list("DCBA")]:
        scores = ranked_pairs(rankings, candidates)
        assert get_winners(scores) == ["D"]
        assert scores == {"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0}


@pytest.mark.parametrize("method", AGGREGATION_METHODS)
@pytest.mark.parametrize("candidates", [["a", "b"], ["b", "a"]])
def test_opposed_ballots_have_no_unique_winner(method, candidates):
    result = aggregate_rankings(method, ballots(["a", "b"], ["b", "a"]), candidates)

    assert result.winner is None
    if method == "stv":
        assert result.winner_ids == []
        assert result.details["status"] == "unresolved"
        assert result.details["reason"] == "elimination_tie"
        assert set(result.details["remaining_ids"]) == {"a", "b"}
    else:
        assert set(result.winner_ids) == {"a", "b"}
        assert result.details["status"] == "tie"


def test_irv_preserves_only_remaining_candidates_on_later_round_tie():
    # C is eliminated first. Its ballot transfers to A, leaving a 3-3 tie.
    result = aggregate_rankings(
        "stv", ballots("ABC", "ABC", "BAC", "BAC", "BAC", "CAB"), list("ABC")
    )

    assert result.winner is None
    assert result.winner_ids == []
    assert result.details["round"] == 2
    assert result.details["remaining_ids"] == ["A", "B"]
    assert result.details["first_place_votes"] == {"A": 3, "B": 3}
    assert result.scores["C"] == 0


def test_irv_stops_at_tied_minimum_instead_of_eliminating_both_candidates():
    # A has 3 votes, B and C each have 2. Eliminating B or C first can
    # transfer those votes to the other contender; dropping both elects A.
    result = aggregate_rankings(
        "stv", ballots("ABC", "ABC", "ABC", "BCA", "BCA", "CBA", "CBA"), list("ABC")
    )

    assert result.winner_ids == []
    assert result.details["tied_for_elimination"] == ["B", "C"]
    assert result.details["remaining_ids"] == ["A", "B", "C"]


def test_irv_can_remove_zero_vote_candidates_and_finish_normally():
    result = aggregate_rankings("stv", ballots("ABC", "ABC", "BAC", "BAC", "CAB"), list("ABCD"))
    assert result.winner_ids == ["A"]
    assert result.details["status"] == "winner"


def test_ranked_pairs_equal_strength_cycle_has_documented_stable_edge_policy():
    rankings = ballots("ABC", "BCA", "CAB")
    first = aggregate_rankings("ranked_pairs", rankings, list("ABC"))
    reversed_order = aggregate_rankings("ranked_pairs", rankings, list("CBA"))

    assert first.winner_ids == reversed_order.winner_ids == ["A"]
    assert first.details["edge_tiebreak"] == "equal margins ordered by winner ID, then loser ID"


def test_ranked_pairs_does_not_elect_a_response_added_after_judging():
    result = aggregate_rankings("ranked_pairs", ballots("ABC"), list("ABCD"))

    assert result.winner_ids == ["A"]
    assert result.winner == "A"
    assert result.scores["D"] == 0
    assert result.details["status"] == "winner"


def test_ranked_pairs_without_any_ranked_candidate_elects_nobody():
    result = aggregate_rankings("ranked_pairs", ballots("AB"), list("CD"))

    assert result.scores == {"C": 0.0, "D": 0.0}
    assert result.winner_ids == []
    assert result.winner is None
    assert result.details["status"] == "no_ballots"


def test_weighted_scores_do_not_break_mathematical_ties_due_to_float_roundoff():
    assert get_winners({"a": 0.1 + 0.2, "b": 0.3}) == ["a", "b"]
    assert get_winner({"a": 0.1 + 0.2, "b": 0.3}) is None
    assert get_winner({}) is None


def test_no_ballots_does_not_report_elected_candidates():
    for method in AGGREGATION_METHODS:
        result = aggregate_rankings(method, [], ["a", "b"])
        assert result.winner_ids == []
        assert result.winner is None
        assert result.details["status"] == "no_ballots"

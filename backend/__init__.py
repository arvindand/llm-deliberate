"""LLM Deliberate - Research tool for multi-model deliberation."""

from .aggregation import (
    agreement_matrix,
    approval_voting,
    borda_count,
    copeland_score,
    diversity_score,
    get_ranking,
    get_winner,
    method_agreement,
    plurality,
    ranked_pairs,
    schulze_method,
    stv_instant_runoff,
    weighted_borda,
)
from .models import AggregationMethod, Experiment, Question, QuestionType, Ranking, Response

__version__ = "0.1.0"
__all__ = [
    "Response",
    "Ranking",
    "Question",
    "Experiment",
    "QuestionType",
    "AggregationMethod",
    "plurality",
    "borda_count",
    "weighted_borda",
    "copeland_score",
    "ranked_pairs",
    "schulze_method",
    "stv_instant_runoff",
    "approval_voting",
    "get_winner",
    "get_ranking",
    "agreement_matrix",
    "method_agreement",
    "diversity_score",
]

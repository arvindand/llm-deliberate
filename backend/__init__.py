"""LLM Deliberate - Research tool for multi-model deliberation."""
from .models import (
    Response,
    Ranking,
    Question,
    Experiment,
    QuestionType,
    AggregationMethod
)
from .aggregation import (
    plurality,
    borda_count,
    weighted_borda,
    copeland_score,
    ranked_pairs,
    get_winner,
    get_ranking,
    agreement_matrix,
    method_agreement,
    diversity_score
)

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
    "get_winner",
    "get_ranking",
    "agreement_matrix",
    "method_agreement",
    "diversity_score"
]

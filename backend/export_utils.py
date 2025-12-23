"""Export utilities for experiments, questions, and aggregation results."""
import csv
import io
from itertools import islice

from .models import Experiment


def _get_metadata_value(metadata: dict | None, key: str) -> str:
    if not metadata:
        return ""
    value = metadata.get(key)
    return "" if value is None else str(value)


def _truncate_text(value: str, limit: int = 500) -> str:
    if len(value) <= limit:
        return value
    return value[:limit] + "..."


def _write_experiment_response_rows(writer: csv.writer, experiment: Experiment) -> None:
    for question in experiment.questions:
        for response in question.responses:
            writer.writerow([
                experiment.id,
                experiment.name,
                question.id,
                question.text,
                question.question_type,
                response.id,
                response.model,
                response.round,
                _truncate_text(response.content),
                _get_metadata_value(response.metadata, "tokens_input"),
                _get_metadata_value(response.metadata, "tokens_output"),
                _get_metadata_value(response.metadata, "cost_usd"),
                _get_metadata_value(response.metadata, "latency_ms"),
                response.source,
                response.created_at.isoformat() if response.created_at else "",
            ])


def _build_rank_cells(rankings: list[str], limit: int = 10) -> list[str]:
    cells = list(islice(rankings, limit))
    cells.extend([""] * max(0, limit - len(cells)))
    return cells


def _build_ranking_row(experiment: Experiment, question, ranking) -> list[str]:
    row: list[str] = [
        experiment.id,
        experiment.name,
        question.id,
        question.text,
        ranking.id,
        ranking.judge,
    ]
    row.extend(_build_rank_cells(ranking.rankings, limit=10))
    row.extend([
        ranking.confidence if ranking.confidence is not None else "",
        ranking.reasoning or "",
        ranking.source,
        ranking.created_at.isoformat() if ranking.created_at else "",
    ])
    return row


def _write_experiment_ranking_rows(writer: csv.writer, experiment: Experiment) -> None:
    for question in experiment.questions:
        for ranking in question.rankings:
            writer.writerow(_build_ranking_row(experiment, question, ranking))


def export_experiment_to_json(experiment: Experiment) -> dict:
    """
    Export full experiment with all nested data.

    Args:
        experiment: Experiment to export

    Returns:
        Dictionary with complete experiment data
    """
    return experiment.model_dump()


def export_experiment_to_csv(experiment: Experiment) -> str:
    """
    Export experiment responses as flattened CSV.

    Args:
        experiment: Experiment to export

    Returns:
        CSV string with one row per response
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        'experiment_id', 'experiment_name', 'question_id', 'question_text',
        'question_type', 'response_id', 'model', 'round', 'content',
        'tokens_input', 'tokens_output', 'cost_usd', 'latency_ms', 'source',
        'created_at'
    ])

    _write_experiment_response_rows(writer, experiment)

    return output.getvalue()


def export_rankings_to_csv(experiment: Experiment) -> str:
    """
    Export rankings as separate CSV.

    Args:
        experiment: Experiment to export

    Returns:
        CSV string with one row per ranking
    """
    output = io.StringIO()
    writer = csv.writer(output)

    # Header row
    writer.writerow([
        'experiment_id', 'experiment_name', 'question_id', 'question_text',
        'ranking_id', 'judge', 'rank_1', 'rank_2', 'rank_3', 'rank_4', 'rank_5',
        'rank_6', 'rank_7', 'rank_8', 'rank_9', 'rank_10',
        'confidence', 'reasoning', 'source', 'created_at'
    ])

    _write_experiment_ranking_rows(writer, experiment)

    return output.getvalue()

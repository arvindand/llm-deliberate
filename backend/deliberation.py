"""
Multi-round deliberation logic for iterative model refinement.

Enables models to see each other's responses and refine their own answers.
"""
import asyncio
import math
from typing import Optional, Callable

from .llm_client import create_client
from .models import Response
from . import prompts


async def run_deliberation_round(
    question_text: str,
    models: list[str],
    previous_responses: list[Response],
    round_number: int,
    on_progress: Optional[Callable] = None,
) -> tuple[list[Response], list[dict]]:
    """Run one round of deliberation where models see and respond to others' answers.

    Args:
        question_text: The original question
        models: List of model names
        previous_responses: List of Response objects from previous round
        round_number: Current round number (for tracking)
        on_progress: Optional callback for progress updates

    Returns:
        Tuple of (new_responses, errors)
    """
    client = create_client()
    new_responses = []
    errors = []

    # Group previous responses by model for easier lookup
    prev_by_model = {r.model: r for r in previous_responses}

    # Only attempt refinement for models that actually have a previous response.
    # This keeps progress totals accurate when some models failed in earlier rounds.
    eligible_models = [model for model in models if model in prev_by_model]

    async def refine_response(model_name: str) -> Optional[Response]:
        try:
            # Get this model's previous response
            prev_response = prev_by_model.get(model_name)
            if not prev_response:
                return None

            # Get other models' responses
            other_responses = [r for r in previous_responses if r.model != model_name]

            # Format the deliberation prompt
            prompt = prompts.format_deliberation_prompt(
                question_text,
                prev_response.content,
                [{"model": r.model, "content": r.content} for r in other_responses],
            )

            # Get refined response
            llm_response = await client.generate(prompt, model_name)

            response = Response(
                model=model_name,
                content=llm_response.content,
                source="automated",
                round=round_number,
                metadata={
                    "tokens_input": llm_response.tokens_input,
                    "tokens_output": llm_response.tokens_output,
                    "latency_ms": llm_response.latency_ms,
                    "cost_usd": llm_response.cost_usd,
                    "provider": llm_response.provider,
                    "api_model": llm_response.model_id,
                    "previous_response": prev_response.id,
                },
            )
            return response
        except Exception as e:
            errors.append(
                {
                    "model": model_name,
                    "error_type": "api_error",
                    "message": str(e),
                    "round": round_number,
                }
            )
            return None

    # Get refined responses in parallel with incremental progress updates
    tasks = [refine_response(model) for model in eligible_models]
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            new_responses.append(result)

        completed += 1
        if on_progress:
            on_progress({"completed": completed, "total": len(eligible_models)})

    return new_responses, errors


def check_convergence(
    current_responses: list[Response],
    previous_responses: list[Response],
    threshold: float = 0.95,
) -> bool:
    """Check if responses have converged (stopped changing significantly).

    Args:
        current_responses: Responses from current round
        previous_responses: Responses from previous round
        threshold: Similarity threshold (0-1) to consider converged

    Returns:
        True if converged, False otherwise
    """
    if not previous_responses or not current_responses:
        return False

    # Group by model for comparison
    prev_by_model = {r.model: r for r in previous_responses}
    curr_by_model = {r.model: r for r in current_responses}

    # Calculate similarity for each model
    similarities = []
    for model_name, curr_resp in curr_by_model.items():
        prev_resp = prev_by_model.get(model_name)
        if not prev_resp:
            continue

        # Simple similarity: exact match = 1.0, else 0.0
        # In production, you might use semantic similarity
        similarity = 1.0 if curr_resp.content == prev_resp.content else 0.0
        similarities.append(similarity)

    # If all models' responses are the same as before, consider converged
    if similarities and all(math.isclose(s, 1.0, rel_tol=0.0, abs_tol=1e-9) for s in similarities):
        return True

    return False


async def run_full_deliberation(
    question_text: str,
    models: list[str],
    max_rounds: int = 3,
    on_progress: Optional[Callable] = None,
) -> tuple[list[Response], list[list[dict]]]:
    """Run full multi-round deliberation.

    Args:
        question_text: The question
        models: List of model names to deliberate
        max_rounds: Maximum number of rounds
        on_progress: Optional callback for progress updates

    Returns:
        Tuple of (all_responses_from_all_rounds, all_errors_by_round)
        Returns ALL responses from ALL rounds, not just the final round
    """
    from .automation import collect_responses_automated

    all_responses_by_round = []
    all_errors = []

    def report_progress(progress: dict, *, round_num: int) -> None:
        if not on_progress:
            return
        payload = dict(progress or {})
        payload["round"] = round_num
        payload["max_rounds"] = max_rounds
        on_progress(payload)

    # Round 1: Initial responses
    report_progress({"completed": 0, "total": len(models)}, round_num=1)
    initial_responses, errors = await collect_responses_automated(
        question_text, models, lambda p: report_progress(p, round_num=1)
    )
    all_responses_by_round.append(initial_responses)
    all_errors.append(errors)

    if len(initial_responses) < 2:
        # Not enough responses to deliberate
        return initial_responses, [[e.model_dump() for e in errors]]

    current_responses = initial_responses

    # Subsequent rounds: deliberation
    for round_num in range(2, max_rounds + 1):
        # Announce round start so UI can switch rounds immediately.
        eligible_models = [r.model for r in current_responses]
        report_progress({"completed": 0, "total": len(eligible_models)}, round_num=round_num)

        # Run deliberation round
        refined_responses, errors = await run_deliberation_round(
            question_text,
            eligible_models,
            current_responses,
            round_num,
            lambda p, rn=round_num: report_progress(p, round_num=rn),
        )

        all_responses_by_round.append(refined_responses)
        all_errors.append(errors)

        if not refined_responses:
            # All models failed this round
            break

        # Check for convergence after running the round
        if check_convergence(refined_responses, current_responses):
            if on_progress:
                report_progress({"converged": True}, round_num=round_num)
            break

        current_responses = refined_responses

    # Flatten all responses from all rounds into a single list
    all_responses_flat = []
    for round_responses in all_responses_by_round:
        all_responses_flat.extend(round_responses)

    # Convert errors to dicts for serialization
    errors_dict = [
        [e.model_dump() if hasattr(e, "model_dump") else e for e in round_errors]
        for round_errors in all_errors
    ]

    return all_responses_flat, errors_dict

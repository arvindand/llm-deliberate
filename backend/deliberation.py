"""
Multi-round deliberation logic for iterative model refinement.

Enables models to see each other's responses and refine their own answers.
"""

import asyncio
import difflib
import math
from collections.abc import Callable

from . import prompts
from .llm_client import create_client
from .models import Response


def cosine_similarity(embedding1: list[float], embedding2: list[float]) -> float:
    """Calculate cosine similarity between two embedding vectors."""
    if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
        return 0.0

    dot_product = sum(a * b for a, b in zip(embedding1, embedding2, strict=True))
    norm1 = math.sqrt(sum(a * a for a in embedding1))
    norm2 = math.sqrt(sum(b * b for b in embedding2))

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)


async def run_deliberation_round(
    question_text: str,
    models: list[str],
    previous_responses: list[Response],
    round_number: int,
    on_progress: Callable | None = None,
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

    async def refine_response(model_name: str) -> Response | None:
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
    for completed, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        if result is not None:
            new_responses.append(result)

        if on_progress:
            on_progress({"completed": completed, "total": len(eligible_models)})

    return new_responses, errors


async def check_convergence(
    current_responses: list[Response],
    previous_responses: list[Response],
    threshold: float = 0.95,
    use_embeddings: bool = True,
) -> bool:
    """Check whether the current models have converged on semantically similar answers.

    Args:
        current_responses: Responses from current round
        previous_responses: Previous round responses, retained for API compatibility
        threshold: Similarity threshold (0-1) to consider converged
        use_embeddings: If True, use embedding-based similarity; if False, use exact string match

    Returns:
        True if converged, False otherwise
    """
    if len(current_responses) < 2:
        return False

    similarities: list[float] = []

    if use_embeddings:
        try:
            client = create_client()
            embedding_results = await asyncio.gather(
                *(client.get_embedding(response.content) for response in current_responses)
            )
            embeddings = [result.embedding for result in embedding_results]
            for index, embedding in enumerate(embeddings):
                for other_embedding in embeddings[index + 1 :]:
                    similarities.append(cosine_similarity(embedding, other_embedding))
        except Exception:
            similarities.clear()

    if not similarities:
        normalized = [" ".join(response.content.lower().split()) for response in current_responses]
        for index, content in enumerate(normalized):
            for other_content in normalized[index + 1 :]:
                similarities.append(difflib.SequenceMatcher(None, content, other_content).ratio())

    # Check if average similarity meets threshold
    if similarities:
        avg_similarity = sum(similarities) / len(similarities)
        return avg_similarity >= threshold

    return False


async def run_full_deliberation(
    question_text: str,
    models: list[str],
    max_rounds: int = 3,
    on_progress: Callable | None = None,
    use_embeddings: bool = True,
) -> tuple[list[Response], list[list[dict]]]:
    """Run full multi-round deliberation.

    Args:
        question_text: The question
        models: List of model names to deliberate
        max_rounds: Maximum number of rounds
        on_progress: Optional callback for progress updates
        use_embeddings: If True, use semantic similarity for convergence detection

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
        if await check_convergence(
            refined_responses, current_responses, use_embeddings=use_embeddings
        ):
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


async def chairman_synthesis(
    question_text: str,
    responses: list[Response],
    rankings: list[dict],
    chairman_model: str,
) -> str:
    """Synthesize final answer using chairman model based on all responses and rankings.

    Args:
        question_text: The original question
        responses: List of Response objects from all models
        rankings: List of ranking dicts with 'judge', 'rankings', 'confidence', 'reasoning'
        chairman_model: Model to use for synthesis

    Returns:
        Synthesized final answer from chairman model
    """
    client = create_client()

    chairman_prompt = prompts.format_chairman_prompt(
        question_text,
        [{"model": r.model, "content": r.content} for r in responses],
        rankings,
    )

    llm_response = await client.generate(chairman_prompt, chairman_model)
    return llm_response.content


async def run_debate(
    question_text: str,
    debaters: list[str],
    judge_model: str,
    num_rounds: int = 3,
    on_progress: Callable | None = None,
) -> tuple[list[Response], dict, list[dict]]:
    """Run debate between models with external judge evaluation.

    Implements adversarial debate format (Irving et al., 2018) where models
    argue for different positions and a judge determines the winner.

    Args:
        question_text: The question to debate
        debaters: List of 2+ model names (each takes opposing stance)
        judge_model: Model to evaluate debate and pick winner
        num_rounds: Number of back-and-forth debate rounds
        on_progress: Optional callback for progress updates

    Returns:
        Tuple of (all_debate_responses, final_judgment, errors)
        - all_debate_responses: List of Response objects from all debate rounds
        - final_judgment: Dict with 'winner', 'reasoning', 'confidence'
        - errors: List of error dicts
    """

    client = create_client()
    all_responses = []
    errors = []

    if len(debaters) < 2:
        raise ValueError("Debate requires at least 2 models")

    for round_num in range(1, num_rounds + 1):
        if on_progress:
            on_progress(
                {
                    "round": round_num,
                    "max_rounds": num_rounds,
                    "phase": "debate",
                    "completed": (round_num - 1) * len(debaters),
                    "total": num_rounds * len(debaters),
                }
            )

        for debater_idx, debater_model in enumerate(debaters):
            try:
                if round_num == 1:
                    prompt = prompts.format_debate_initial_prompt(
                        question_text, debater_idx, len(debaters)
                    )
                else:
                    other_responses = [
                        r
                        for r in all_responses
                        if r.round == round_num - 1 and r.model != debater_model
                    ]
                    own_prev = next(
                        (
                            r
                            for r in all_responses
                            if r.round == round_num - 1 and r.model == debater_model
                        ),
                        None,
                    )
                    prompt = prompts.format_debate_rebuttal_prompt(
                        question_text,
                        own_prev.content if own_prev else "",
                        [{"model": r.model, "content": r.content} for r in other_responses],
                    )

                llm_response = await client.generate(prompt, debater_model)

                response = Response(
                    model=debater_model,
                    content=llm_response.content,
                    source="automated",
                    round=round_num,
                    metadata={
                        "tokens_input": llm_response.tokens_input,
                        "tokens_output": llm_response.tokens_output,
                        "latency_ms": llm_response.latency_ms,
                        "cost_usd": llm_response.cost_usd,
                        "provider": llm_response.provider,
                        "api_model": llm_response.model_id,
                        "debate_role": f"debater_{debater_idx}",
                    },
                )
                all_responses.append(response)

            except Exception as e:
                errors.append(
                    {
                        "model": debater_model,
                        "error_type": "api_error",
                        "message": str(e),
                        "round": round_num,
                    }
                )

    if on_progress:
        on_progress(
            {
                "round": num_rounds,
                "max_rounds": num_rounds,
                "phase": "judgment",
                "completed": num_rounds * len(debaters),
                "total": num_rounds * len(debaters) + 1,
            }
        )

    try:
        final_round_responses = [r for r in all_responses if r.round == num_rounds]
        judge_prompt = prompts.format_debate_judge_prompt(
            question_text,
            [{"model": r.model, "content": r.content} for r in final_round_responses],
        )

        judge_response = await client.generate(judge_prompt, judge_model)
        winner, confidence, reasoning = prompts.parse_debate_judgment(judge_response.content)

        judgment = {
            "winner": winner,
            "confidence": confidence,
            "reasoning": reasoning,
            "judge_model": judge_model,
            "metadata": {
                "tokens_input": judge_response.tokens_input,
                "tokens_output": judge_response.tokens_output,
                "cost_usd": judge_response.cost_usd,
            },
        }
    except Exception as e:
        errors.append(
            {
                "model": judge_model,
                "error_type": "judgment_error",
                "message": str(e),
            }
        )
        judgment = {
            "winner": None,
            "confidence": 0.0,
            "reasoning": f"Judge failed: {str(e)}",
            "judge_model": judge_model,
        }

    if on_progress:
        on_progress(
            {
                "round": num_rounds,
                "max_rounds": num_rounds,
                "phase": "complete",
                "completed": num_rounds * len(debaters) + 1,
                "total": num_rounds * len(debaters) + 1,
            }
        )

    return all_responses, judgment, errors

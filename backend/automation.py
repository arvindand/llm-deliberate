"""
Job orchestration for automated response and ranking collection.

Manages in-memory job queue for async API calls with progress tracking.
"""

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel

from . import deliberation, prompts
from .llm_client import create_client
from .models import ChairmanSynthesis, Ranking, Response


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def rank_letter_to_index(rank_letter: str) -> int | None:
    """Convert a ranking token like 'A', 'AA', or 'Response A' to a 0-based index."""
    rank_str = rank_letter.strip()
    if rank_str.lower().startswith("response "):
        rank_str = rank_str[9:].strip()
    label = rank_str.upper()

    if not label or not label.isalpha() or not label.isascii():
        return None

    index = 0
    for character in label:
        index = index * 26 + (ord(character) - ord("A") + 1)
    return index - 1


def ranking_letters_to_response_ids(
    ranking_list: list[str], responses: list[Response]
) -> list[str]:
    response_ids: list[str] = []
    seen: set[str] = set()
    for rank_letter in ranking_list:
        idx = rank_letter_to_index(rank_letter)
        if idx is None:
            continue
        if 0 <= idx < len(responses):
            response_id = responses[idx].id
            if response_id not in seen:
                response_ids.append(response_id)
                seen.add(response_id)
    return response_ids


class JobStatus(str, Enum):
    """Status of an automation job."""

    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class AutomationError(BaseModel):
    """Error during automation."""

    model: str
    error_type: str  # "api_error", "timeout", "rate_limit", "validation_error"
    message: str
    retry_count: int = 0
    timestamp: datetime | None = None

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = now_utc()
        super().__init__(**data)


class JobProgress(BaseModel):
    """Progress information for a job."""

    job_id: str
    status: JobStatus
    experiment_id: str | None = None
    progress: dict[str, Any] | None = None  # {"completed": 3, "total": 5}
    results: list[dict[str, Any]] | None = None  # Responses or Rankings
    errors: list[AutomationError | dict[str, Any]] | None = None
    started_at: datetime | None = None
    completed_at: datetime | None = None
    completed_count: int = 0
    total_count: int = 0


# Global job tracking (in-memory)
jobs: dict[str, JobProgress] = {}

_background_tasks: set[asyncio.Task[Any]] = set()


def track_task(coro: Awaitable[Any]) -> asyncio.Task[Any]:
    task = asyncio.create_task(coro)
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)
    return task


def create_job_id() -> str:
    """Generate a unique job ID."""
    return str(uuid.uuid4())[:12]


async def collect_responses_automated(
    question_text: str,
    models: list[str],
    on_progress: Callable | None = None,
    on_result: Callable[[Response], None] | None = None,
    temperature: float | None = None,
) -> tuple[list[Response], list[AutomationError]]:
    """Collect responses from multiple models in parallel.

    Args:
        question_text: The question to ask
        models: List of model names to query
        on_progress: Optional callback for progress updates
        temperature: Optional temperature override for self-consistency sampling

    Returns:
        Tuple of (successful_responses, errors)
    """
    client = create_client()
    responses = []
    errors = []

    prompt = prompts.format_response_prompt(question_text)

    async def get_response(model_name: str) -> Response | None:
        try:
            llm_response = await client.generate(
                prompt, model_name, temperature=temperature if temperature is not None else 0.7
            )

            response = Response(
                model=model_name,
                content=llm_response.content,
                source="automated",
                round=1,
                metadata={
                    "tokens_input": llm_response.tokens_input,
                    "tokens_output": llm_response.tokens_output,
                    "latency_ms": llm_response.latency_ms,
                    "cost_usd": llm_response.cost_usd,
                    "provider": llm_response.provider,
                    "api_model": llm_response.model_id,
                    "temperature": temperature if temperature is not None else 0.7,
                },
            )
            return response
        except Exception as e:
            error = AutomationError(
                model=model_name,
                error_type="api_error",
                message=str(e),
            )
            errors.append(error)
            return None

    # Collect responses in parallel with incremental progress updates
    tasks = [get_response(model) for model in models]
    for completed, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        if result is not None:
            responses.append(result)
            if on_result:
                on_result(result)

        # Report progress after each completion
        if on_progress:
            on_progress({"completed": completed, "total": len(models)})

    return responses, errors


async def collect_rankings_automated(
    question_text: str,
    responses: list[Response],
    judges: list[str],
    on_progress: Callable | None = None,
    on_result: Callable[[Ranking], None] | None = None,
    anonymize: bool = True,
) -> tuple[list[Ranking], list[AutomationError]]:
    """Collect rankings from multiple judges in parallel.

    Args:
        question_text: The question being ranked
        responses: List of Response objects to rank
        judges: List of model names to use as judges
        on_progress: Optional callback for progress updates
        anonymize: If True, anonymize model names to prevent bias

    Returns:
        Tuple of (successful_rankings, errors)
    """
    client = create_client()
    rankings = []
    errors = []

    if anonymize:
        prompt = prompts.format_ranking_prompt_anonymized(
            question_text, [{"content": r.content} for r in responses]
        )
    else:
        prompt = prompts.format_ranking_prompt(
            question_text, [{"model": r.model, "content": r.content} for r in responses]
        )

    async def get_ranking(judge_name: str) -> Ranking | None:
        try:
            llm_response = await client.generate(prompt, judge_name)

            ranking_list, confidence, reasoning = prompts.parse_ranking_response(
                llm_response.content
            )
            response_ids = ranking_letters_to_response_ids(ranking_list, responses)
            if len(response_ids) != len(responses):
                raise ValueError(
                    "Judge returned an incomplete or invalid ranking "
                    f"({len(response_ids)} of {len(responses)} responses)"
                )

            ranking = Ranking(
                judge=judge_name,
                rankings=response_ids,
                confidence=confidence,
                reasoning=reasoning,
                source="automated",
            )
            ranking.metadata = {
                "tokens_input": llm_response.tokens_input,
                "tokens_output": llm_response.tokens_output,
                "latency_ms": llm_response.latency_ms,
                "cost_usd": llm_response.cost_usd,
                "provider": llm_response.provider,
                "api_model": llm_response.model_id,
            }
            return ranking
        except Exception as e:
            error = AutomationError(
                model=judge_name,
                error_type="api_error",
                message=str(e),
            )
            errors.append(error)
            return None

    # Collect rankings in parallel with incremental progress updates
    tasks = [get_ranking(judge) for judge in judges]
    for completed, coro in enumerate(asyncio.as_completed(tasks), start=1):
        result = await coro
        if result is not None:
            rankings.append(result)
            if on_result:
                on_result(result)

        # Report progress after each completion
        if on_progress:
            on_progress({"completed": completed, "total": len(judges)})

    return rankings, errors


async def process_job_responses(
    job_id: str,
    question_text: str,
    models: list[str],
    result_callback: Callable | None = None,
) -> None:
    """Background task to process response collection job."""
    job = jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.started_at = now_utc()
    job.results = []
    job.errors = []
    job.total_count = len(models)

    def on_progress(progress):
        job.progress = progress
        job.completed_count = progress.get("completed", 0)

    def on_result(response: Response) -> None:
        if job.results is None:
            job.results = []
        job.results.append(response.model_dump())

    try:
        responses, errors = await collect_responses_automated(
            question_text, models, on_progress, on_result
        )
        job.results = [r.model_dump() for r in responses]
        job.errors = [e.model_dump() for e in errors]
        job.completed_count = len(responses)

        if len(responses) == 0 and len(errors) > 0:
            job.status = JobStatus.FAILED
            job.completed_at = now_utc()
            return

        if result_callback:
            result_callback(job_id, responses, errors)

        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.errors = [
            {
                "model": "system",
                "error_type": "system_error",
                "message": str(e),
            }
        ]

    job.completed_at = now_utc()


async def process_job_rankings(
    job_id: str,
    question_text: str,
    responses: list[Response],
    judges: list[str],
    result_callback: Callable | None = None,
) -> None:
    """Background task to process ranking collection job."""
    job = jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.started_at = now_utc()
    job.results = []
    job.errors = []
    job.total_count = len(judges)

    def on_progress(progress):
        job.progress = progress
        job.completed_count = progress.get("completed", 0)

    def on_result(ranking: Ranking) -> None:
        if job.results is None:
            job.results = []
        job.results.append(ranking.model_dump())

    try:
        rankings, errors = await collect_rankings_automated(
            question_text, responses, judges, on_progress, on_result
        )
        job.results = [r.model_dump() for r in rankings]
        job.errors = [e.model_dump() for e in errors]
        job.completed_count = len(rankings)

        if len(rankings) == 0 and len(errors) > 0:
            job.status = JobStatus.FAILED
            job.completed_at = now_utc()
            return

        if result_callback:
            result_callback(job_id, rankings, errors)

        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.errors = [
            {
                "model": "system",
                "error_type": "system_error",
                "message": str(e),
            }
        ]

    job.completed_at = now_utc()


def start_response_job(
    question_text: str,
    models: list[str],
    experiment_id: str | None = None,
    question_id: str | None = None,
    result_callback: Callable | None = None,
) -> str:
    """Start a job to collect responses from models.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
        experiment_id=experiment_id,
        progress={"completed": 0, "total": len(models)},
        errors=[],
        total_count=len(models),
        completed_count=0,
    )
    jobs[job_id] = job

    # Create result callback that saves to experiment
    def save_callback(job_id, responses, errors):
        if experiment_id and question_id:
            from . import main

            exp = main.load_experiment(experiment_id)
            main.merge_responses_into_question(exp, question_id, responses)
            main.save_experiment(exp)

        if result_callback:
            result_callback(job_id, responses, errors)

    # Schedule the job
    track_task(process_job_responses(job_id, question_text, models, save_callback))

    return job_id


def start_ranking_job(
    question_text: str,
    responses: list[Response],
    judges: list[str],
    experiment_id: str | None = None,
    question_id: str | None = None,
    result_callback: Callable | None = None,
) -> str:
    """Start a job to collect rankings from judges.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
        experiment_id=experiment_id,
        progress={"completed": 0, "total": len(judges)},
        errors=[],
        total_count=len(judges),
        completed_count=0,
    )
    jobs[job_id] = job

    # Create result callback that saves to experiment
    def save_callback(job_id, rankings, errors):
        if experiment_id and question_id:
            from . import main

            exp = main.load_experiment(experiment_id)
            main.merge_rankings_into_question(exp, question_id, rankings)
            main.save_experiment(exp)

        if result_callback:
            result_callback(job_id, rankings, errors)

    # Schedule the job
    track_task(process_job_rankings(job_id, question_text, responses, judges, save_callback))

    return job_id


def get_job_status(job_id: str) -> JobProgress | None:
    """Get the status of a job."""
    return jobs.get(job_id)


async def process_job_deliberation(
    job_id: str,
    question_text: str,
    models: list[str],
    max_rounds: int,
    result_callback: Callable | None = None,
) -> None:
    """Background task to process deliberation job."""
    job = jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.started_at = now_utc()
    job.results = []
    job.errors = []
    job.total_count = max_rounds

    # Ensure progress has a stable shape for the UI.
    job.progress = {
        "round": 1,
        "max_rounds": max_rounds,
        "completed": 0,
        "total": len(models),
    }

    def on_progress(progress):
        if job.progress is None:
            job.progress = {}

        # Merge updates so we don't lose keys like max_rounds/round.
        job.progress.update(progress or {})

        round_num = job.progress.get("round")
        converged = bool(job.progress.get("converged"))
        if isinstance(round_num, int):
            # Represent "rounds completed" for any legacy consumers.
            job.completed_count = min(
                job.total_count,
                max(0, round_num if converged else round_num - 1),
            )

    try:
        current_responses, all_errors = await deliberation.run_full_deliberation(
            question_text, models, max_rounds, on_progress
        )
        job.results = [r.model_dump() for r in current_responses]
        job.errors = [e for errors in all_errors for e in errors] if all_errors else []
        completed_rounds = max((response.round for response in current_responses), default=0)
        job.completed_count = completed_rounds

        if not current_responses:
            job.status = JobStatus.FAILED
            if not job.errors:
                job.errors = [
                    {
                        "model": "system",
                        "error_type": "empty_result",
                        "message": "Deliberation produced no responses",
                    }
                ]
        else:
            job.status = JobStatus.COMPLETED

        if result_callback and current_responses:
            result_callback(job_id, current_responses, job.errors)
    except Exception as e:
        job.status = JobStatus.FAILED
        job.errors = [
            {
                "model": "system",
                "error_type": "system_error",
                "message": str(e),
            }
        ]

    job.completed_at = now_utc()


def start_deliberation_job(
    question_text: str,
    models: list[str],
    max_rounds: int = 3,
    experiment_id: str | None = None,
    question_id: str | None = None,
    result_callback: Callable | None = None,
) -> str:
    """Start a job to run multi-round deliberation.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
        experiment_id=experiment_id,
        progress={"round": 1, "max_rounds": max_rounds, "completed": 0, "total": len(models)},
        errors=[],
        total_count=max_rounds,
        completed_count=0,
    )
    jobs[job_id] = job

    # Create result callback that saves to experiment
    def save_callback(job_id, responses, errors):
        if experiment_id and question_id:
            from . import main

            try:
                exp = main.load_experiment(experiment_id)
                main.merge_responses_into_question(exp, question_id, responses)
                main.save_experiment(exp)
            except Exception:
                pass  # Error already tracked in job

        if result_callback:
            result_callback(job_id, responses, errors)

    # Schedule the job
    track_task(process_job_deliberation(job_id, question_text, models, max_rounds, save_callback))

    return job_id


async def process_job_chairman(
    job_id: str,
    question_text: str,
    responses: list[Response],
    rankings: list[dict],
    chairman_model: str,
    result_callback: Callable | None = None,
) -> None:
    """Background task to process chairman synthesis job."""
    job = jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.started_at = now_utc()
    job.results = []
    job.errors = []
    job.total_count = 1

    try:
        synthesis = await deliberation.chairman_synthesis(
            question_text, responses, rankings, chairman_model
        )
        job.results = [{"content": synthesis, "chairman_model": chairman_model}]
        job.completed_count = 1
        job.progress = {"completed": 1, "total": 1}

        if result_callback:
            result_callback(job_id, synthesis)

        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.errors = [
            {
                "model": chairman_model,
                "error_type": "api_error",
                "message": str(e),
            }
        ]

    job.completed_at = now_utc()


def start_chairman_job(
    question_text: str,
    responses: list[Response],
    rankings: list[dict],
    chairman_model: str,
    experiment_id: str | None = None,
    question_id: str | None = None,
    result_callback: Callable | None = None,
) -> str:
    """Start a job to synthesize final answer using chairman model.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
        experiment_id=experiment_id,
        progress={"completed": 0, "total": 1},
        errors=[],
        total_count=1,
        completed_count=0,
    )
    jobs[job_id] = job

    def save_callback(completed_job_id, synthesis):
        if experiment_id and question_id:
            from . import main

            experiment = main.load_experiment(experiment_id)
            question = experiment.get_question_by_id(question_id)
            if question is None:
                raise ValueError(f"Question not found: {question_id}")
            question.chairman_synthesis = ChairmanSynthesis(
                content=synthesis,
                chairman_model=chairman_model,
                job_id=completed_job_id,
            )
            main.save_experiment(experiment)

        if result_callback:
            result_callback(completed_job_id, synthesis)

    # Schedule the job
    track_task(
        process_job_chairman(
            job_id, question_text, responses, rankings, chairman_model, save_callback
        )
    )

    return job_id


async def collect_self_consistency_responses(
    question_text: str,
    model: str,
    num_samples: int = 5,
    temperature: float = 0.8,
    on_progress: Callable | None = None,
    on_result: Callable[[Response], None] | None = None,
) -> tuple[list[Response], list[AutomationError]]:
    """Collect multiple responses from same model for self-consistency.

    Self-consistency (Wang et al., 2022) samples diverse reasoning paths
    by running the same model multiple times with higher temperature.

    Args:
        question_text: The question to ask
        model: Single model name to query multiple times
        num_samples: Number of independent samples to generate
        temperature: Sampling temperature (higher = more diversity)
        on_progress: Optional callback for progress updates
        on_result: Optional callback for each completed response

    Returns:
        Tuple of (successful_responses, errors)
    """
    models_list = [model] * num_samples
    return await collect_responses_automated(
        question_text, models_list, on_progress, on_result, temperature=temperature
    )


async def process_job_self_consistency(
    job_id: str,
    question_text: str,
    model: str,
    num_samples: int,
    temperature: float,
    result_callback: Callable | None = None,
) -> None:
    """Background task to process self-consistency job."""
    job = jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.started_at = now_utc()
    job.results = []
    job.errors = []
    job.total_count = num_samples

    def on_progress(progress):
        job.progress = progress
        job.completed_count = progress.get("completed", 0)

    def on_result(response: Response) -> None:
        if job.results is None:
            job.results = []
        job.results.append(response.model_dump())

    try:
        responses, errors = await collect_self_consistency_responses(
            question_text, model, num_samples, temperature, on_progress, on_result
        )
        job.results = [r.model_dump() for r in responses]
        job.errors = [e.model_dump() for e in errors]
        job.completed_count = len(responses)

        if len(responses) == 0 and len(errors) > 0:
            job.status = JobStatus.FAILED
            job.completed_at = now_utc()
            return

        if result_callback:
            result_callback(job_id, responses, errors)

        job.status = JobStatus.COMPLETED
    except Exception as e:
        job.status = JobStatus.FAILED
        job.errors = [
            {
                "model": model,
                "error_type": "system_error",
                "message": str(e),
            }
        ]

    job.completed_at = now_utc()


def start_self_consistency_job(
    question_text: str,
    model: str,
    num_samples: int = 5,
    temperature: float = 0.8,
    experiment_id: str | None = None,
    question_id: str | None = None,
    result_callback: Callable | None = None,
) -> str:
    """Start a job to collect self-consistency samples from a single model.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
        experiment_id=experiment_id,
        progress={"completed": 0, "total": num_samples},
        errors=[],
        total_count=num_samples,
        completed_count=0,
    )
    jobs[job_id] = job

    def save_callback(job_id, responses, errors):
        if experiment_id and question_id:
            from . import main

            exp = main.load_experiment(experiment_id)
            main.merge_responses_into_question(exp, question_id, responses)
            main.save_experiment(exp)

        if result_callback:
            result_callback(job_id, responses, errors)

    track_task(
        process_job_self_consistency(
            job_id, question_text, model, num_samples, temperature, save_callback
        )
    )

    return job_id


async def process_job_debate(
    job_id: str,
    question_text: str,
    debaters: list[str],
    judge_model: str,
    num_rounds: int,
    result_callback: Callable | None = None,
) -> None:
    """Background task to process debate job."""
    job = jobs[job_id]
    job.status = JobStatus.PROCESSING
    job.started_at = now_utc()
    job.results = []
    job.errors = []
    job.total_count = num_rounds * len(debaters) + 1

    def on_progress(progress):
        job.progress = progress
        phase = progress.get("phase", "debate")
        if phase == "complete":
            job.completed_count = job.total_count

    try:
        responses, judgment, errors = await deliberation.run_debate(
            question_text, debaters, judge_model, num_rounds, on_progress
        )
        job.results = [r.model_dump() for r in responses]
        job.errors = list(errors) if errors else []
        job.completed_count = job.total_count

        if job.progress is None:
            job.progress = {}
        job.progress["judgment"] = judgment

        if not responses:
            job.status = JobStatus.FAILED
            if not job.errors:
                job.errors = [
                    {
                        "model": "system",
                        "error_type": "empty_result",
                        "message": "Debate produced no arguments",
                    }
                ]
        else:
            job.status = JobStatus.COMPLETED

        if result_callback and responses:
            result_callback(job_id, responses, judgment, errors)
    except Exception as e:
        job.status = JobStatus.FAILED
        job.errors = [
            {
                "model": "system",
                "error_type": "system_error",
                "message": str(e),
            }
        ]

    job.completed_at = now_utc()


def start_debate_job(
    question_text: str,
    debaters: list[str],
    judge_model: str,
    num_rounds: int = 3,
    experiment_id: str | None = None,
    question_id: str | None = None,
    result_callback: Callable | None = None,
) -> str:
    """Start a job to run adversarial debate between models.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
        experiment_id=experiment_id,
        progress={
            "round": 1,
            "max_rounds": num_rounds,
            "phase": "debate",
            "completed": 0,
            "total": num_rounds * len(debaters) + 1,
        },
        errors=[],
        total_count=num_rounds * len(debaters) + 1,
        completed_count=0,
    )
    jobs[job_id] = job

    def save_callback(job_id, responses, judgment, errors):
        if experiment_id and question_id:
            from . import main

            try:
                exp = main.load_experiment(experiment_id)
                main.merge_responses_into_question(exp, question_id, responses)
                main.save_experiment(exp)
            except Exception:
                pass

        if result_callback:
            result_callback(job_id, responses, judgment, errors)

    track_task(
        process_job_debate(job_id, question_text, debaters, judge_model, num_rounds, save_callback)
    )

    return job_id


def cleanup_old_jobs(max_age_seconds: int = 3600) -> None:
    """Clean up completed jobs older than max_age_seconds."""
    now = now_utc()
    to_delete = []

    for job_id, job in jobs.items():
        if (
            job.status in (JobStatus.COMPLETED, JobStatus.FAILED)
            and job.completed_at
            and (now - job.completed_at).total_seconds() > max_age_seconds
        ):
            to_delete.append(job_id)

    for job_id in to_delete:
        del jobs[job_id]

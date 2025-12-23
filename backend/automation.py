"""
Job orchestration for automated response and ranking collection.

Manages in-memory job queue for async API calls with progress tracking.
"""
import asyncio
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable
from pydantic import BaseModel

from .llm_client import create_client, LLMResponse
from .models import Response, Ranking
from . import prompts, config, deliberation


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def rank_letter_to_index(rank_letter: str) -> Optional[int]:
    """Convert a ranking token like 'A' or 'Response A' to a 0-based index."""
    rank_str = rank_letter.strip()
    if rank_str.startswith("Response "):
        letter = rank_str.replace("Response ", "").strip().upper()
    else:
        letter = rank_str.upper()

    if len(letter) != 1 or not ("A" <= letter <= "Z"):
        return None

    return ord(letter) - ord("A")


def ranking_letters_to_response_ids(
    ranking_list: list[str], responses: list[Response]
) -> list[str]:
    response_ids: list[str] = []
    for rank_letter in ranking_list:
        idx = rank_letter_to_index(rank_letter)
        if idx is None:
            continue
        if 0 <= idx < len(responses):
            response_ids.append(responses[idx].id)
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
    timestamp: Optional[datetime] = None

    def __init__(self, **data):
        if "timestamp" not in data:
            data["timestamp"] = now_utc()
        super().__init__(**data)


class JobProgress(BaseModel):
    """Progress information for a job."""

    job_id: str
    status: JobStatus
    progress: Optional[dict] = None  # {"completed": 3, "total": 5}
    results: Optional[list[dict]] = None  # Responses or Rankings
    errors: Optional[list[AutomationError]] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    completed_count: int = 0
    total_count: int = 0


# Global job tracking (in-memory)
jobs: dict[str, JobProgress] = {}

_background_tasks: set[asyncio.Task] = set()


def track_task(coro) -> asyncio.Task:
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
    on_progress: Optional[Callable] = None,
    on_result: Optional[Callable[[Response], None]] = None,
) -> tuple[list[Response], list[AutomationError]]:
    """Collect responses from multiple models in parallel.

    Args:
        question_text: The question to ask
        models: List of model names to query
        on_progress: Optional callback for progress updates

    Returns:
        Tuple of (successful_responses, errors)
    """
    client = create_client()
    responses = []
    errors = []

    prompt = prompts.format_response_prompt(question_text)

    async def get_response(model_name: str) -> Optional[Response]:
        try:
            llm_response = await client.generate(prompt, model_name)

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
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            responses.append(result)
            if on_result:
                on_result(result)
        completed += 1

        # Report progress after each completion
        if on_progress:
            on_progress({"completed": completed, "total": len(models)})

    return responses, errors


async def collect_rankings_automated(
    question_text: str,
    responses: list[Response],
    judges: list[str],
    on_progress: Optional[Callable] = None,
    on_result: Optional[Callable[[Ranking], None]] = None,
) -> tuple[list[Ranking], list[AutomationError]]:
    """Collect rankings from multiple judges in parallel.

    Args:
        question_text: The question being ranked
        responses: List of Response objects to rank
        judges: List of model names to use as judges
        on_progress: Optional callback for progress updates

    Returns:
        Tuple of (successful_rankings, errors)
    """
    client = create_client()
    rankings = []
    errors = []

    prompt = prompts.format_ranking_prompt(
        question_text, [{"model": r.model, "content": r.content} for r in responses]
    )

    async def get_ranking(judge_name: str) -> Optional[Ranking]:
        try:
            llm_response = await client.generate(prompt, judge_name)

            # Parse the ranking response
            ranking_list, confidence, reasoning = prompts.parse_ranking_response(
                llm_response.content
            )

            response_ids = ranking_letters_to_response_ids(ranking_list, responses)

            ranking = Ranking(
                judge=judge_name,
                rankings=response_ids,
                confidence=confidence,
                reasoning=reasoning,
                source="automated",
                metadata={
                    "tokens_input": llm_response.tokens_input,
                    "tokens_output": llm_response.tokens_output,
                    "latency_ms": llm_response.latency_ms,
                    "cost_usd": llm_response.cost_usd,
                    "provider": llm_response.provider,
                    "api_model": llm_response.model_id,
                },
            )
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
    completed = 0

    for coro in asyncio.as_completed(tasks):
        result = await coro
        if result is not None:
            rankings.append(result)
            if on_result:
                on_result(result)
        completed += 1

        # Report progress after each completion
        if on_progress:
            on_progress({"completed": completed, "total": len(judges)})

    return rankings, errors


async def process_job_responses(
    job_id: str,
    question_text: str,
    models: list[str],
    result_callback: Optional[Callable] = None,
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
    result_callback: Optional[Callable] = None,
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
    experiment_id: str = None,
    question_id: str = None,
    result_callback: Optional[Callable] = None,
) -> str:
    """Start a job to collect responses from models.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
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
    experiment_id: str = None,
    question_id: str = None,
    result_callback: Optional[Callable] = None,
) -> str:
    """Start a job to collect rankings from judges.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
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


def get_job_status(job_id: str) -> Optional[JobProgress]:
    """Get the status of a job."""
    return jobs.get(job_id)


async def process_job_deliberation(
    job_id: str,
    question_text: str,
    models: list[str],
    max_rounds: int,
    result_callback: Optional[Callable] = None,
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
        job.completed_count = max_rounds
        job.status = JobStatus.COMPLETED

        if result_callback:
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
    experiment_id: str = None,
    question_id: str = None,
    result_callback: Optional[Callable] = None,
) -> str:
    """Start a job to run multi-round deliberation.

    Returns:
        job_id
    """
    job_id = create_job_id()
    job = JobProgress(
        job_id=job_id,
        status=JobStatus.QUEUED,
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


def cleanup_old_jobs(max_age_seconds: int = 3600) -> None:
    """Clean up completed jobs older than max_age_seconds."""
    now = now_utc()
    to_delete = []

    for job_id, job in jobs.items():
        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED):
            if job.completed_at and (now - job.completed_at).total_seconds() > max_age_seconds:
                to_delete.append(job_id)

    for job_id in to_delete:
        del jobs[job_id]

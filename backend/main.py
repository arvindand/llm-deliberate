"""
LLM Deliberate - A research tool for exploring multi-model deliberation and aggregation methods.
"""

import asyncio
import io
import json
import os
import re
import tempfile
from contextlib import asynccontextmanager, suppress
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from . import automation, config
from .aggregation import (
    approval_voting,
    borda_count,
    copeland_score,
    get_winner,
    plurality,
    ranked_pairs,
    schulze_method,
    stv_instant_runoff,
    weighted_borda,
)
from .models import (
    Experiment,
    Question,
    QuestionType,
    Ranking,
    Response,
)

# === Error Message Constants ===
ERR_QUESTION_NOT_FOUND = "Question not found"
ERR_EXPERIMENT_NOT_FOUND = "Experiment not found"
ERR_AUTOMATION_UNAVAILABLE = "Automation not available - API key not configured"

CSV_MEDIA_TYPE = "text/csv"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Run periodic cleanup and reliably stop it during application shutdown."""

    async def cleanup_loop():
        while True:
            await asyncio.sleep(3600)
            automation.cleanup_old_jobs(max_age_seconds=3600)

    cleanup_task = asyncio.create_task(cleanup_loop())
    app.state.cleanup_task = cleanup_task
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(
    title="LLM Deliberate",
    description="Research tool for multi-model deliberation and consensus algorithms",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = Path(__file__).parent.parent / "data" / "experiments"
DATA_DIR.mkdir(parents=True, exist_ok=True)


# === API Models ===


class CreateExperimentRequest(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Experiment name cannot be empty")
        return value


class AddQuestionRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    question_type: QuestionType
    ground_truth: str | None = Field(default=None, max_length=10000)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question text cannot be empty")
        return value


class AddResponseRequest(BaseModel):
    question_id: str
    model: str = Field(min_length=1, max_length=300)
    content: str = Field(min_length=1, max_length=50000)

    @field_validator("model", "content")
    @classmethod
    def validate_nonempty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Value cannot be empty")
        return value


class AddRankingRequest(BaseModel):
    question_id: str
    judge: str = Field(min_length=1, max_length=300)
    rankings: list[str] = Field(min_length=1)  # Response IDs, best to worst
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str | None = Field(default=None, max_length=10000)


class ComputeResultsRequest(BaseModel):
    question_id: str
    method: str = "borda"  # borda, weighted_borda, copeland, plurality, ranked_pairs


class AutomateResponsesRequest(BaseModel):
    question_id: str
    models: list[str] = Field(min_length=1, max_length=20)


class AutomateRankingsRequest(BaseModel):
    question_id: str
    judges: list[str] = Field(min_length=1, max_length=20)


class DeliberateRequest(BaseModel):
    question_id: str
    models: list[str] = Field(min_length=2, max_length=20)
    max_rounds: int = Field(default=3, ge=1, le=10)


class ChairmanSynthesisRequest(BaseModel):
    question_id: str
    chairman_model: str


class SelfConsistencyRequest(BaseModel):
    question_id: str
    model: str
    num_samples: int = Field(default=5, ge=2, le=50)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)


class DebateRequest(BaseModel):
    question_id: str
    debaters: list[str] = Field(min_length=2, max_length=8)
    judge_model: str
    num_rounds: int = Field(default=3, ge=1, le=10)


# === Storage Helpers ===


def get_experiment_path(experiment_id: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9_-]+", experiment_id):
        raise HTTPException(status_code=400, detail="Invalid experiment ID")
    return DATA_DIR / f"{experiment_id}.json"


@lru_cache(maxsize=50)
def _load_experiment_cached(experiment_id: str, mtime: float) -> Experiment:
    """Load experiment with caching based on file modification time."""
    path = get_experiment_path(experiment_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    with open(path) as f:
        data = json.load(f)
    return Experiment(**data)


def load_experiment(experiment_id: str) -> Experiment:
    """Load experiment with automatic cache invalidation on file changes."""
    path = get_experiment_path(experiment_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Experiment {experiment_id} not found")
    # Use file mtime as cache key - when file changes, mtime changes, cache invalidates
    mtime = os.path.getmtime(path)
    return _load_experiment_cached(experiment_id, mtime)


def save_experiment(experiment: Experiment):
    path = get_experiment_path(experiment.id)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=DATA_DIR,
        prefix=f".{experiment.id}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        json.dump(experiment.model_dump(), temporary_file, indent=2, default=str)
        temporary_path = Path(temporary_file.name)
    os.replace(temporary_path, path)
    # Clear cache for this experiment after save
    _load_experiment_cached.cache_clear()


def list_experiment_files() -> list[str]:
    return [f.stem for f in DATA_DIR.glob("*.json")]


def merge_responses_into_question(
    experiment: Experiment, question_id: str, responses: list[Response]
):
    """Merge automated responses into a question, avoiding duplicates."""
    question = experiment.get_question_by_id(question_id)
    if not question:
        return

    # Get existing response IDs to avoid duplicates
    existing_ids = {r.id for r in question.responses}

    # Add new responses that don't already exist (by ID)
    # This allows multiple responses from the same model (different rounds)
    for response in responses:
        if response.id not in existing_ids:
            question.responses.append(response)
            existing_ids.add(response.id)


def merge_rankings_into_question(experiment: Experiment, question_id: str, rankings: list[Ranking]):
    """Merge automated rankings into a question, avoiding duplicates."""
    question = experiment.get_question_by_id(question_id)
    if not question:
        return

    # Get existing judge names
    existing_judges = {r.judge for r in question.rankings}

    # Add new rankings from judges that haven't ranked yet
    for ranking in rankings:
        if ranking.judge not in existing_judges:
            question.rankings.append(ranking)
            existing_judges.add(ranking.judge)


# === API Endpoints ===


@app.get("/")
def root():
    return {
        "name": "LLM Deliberate",
        "version": "0.1.0",
        "description": "Research tool for multi-model deliberation",
    }


@app.get("/experiments")
def list_experiments():
    """List all experiments."""
    experiments = []
    for exp_id in list_experiment_files():
        try:
            exp = load_experiment(exp_id)
            experiments.append(
                {
                    "id": exp.id,
                    "name": exp.name,
                    "description": exp.description,
                    "created_at": exp.created_at,
                    "question_count": len(exp.questions),
                }
            )
        except Exception:
            continue
    return {"experiments": experiments}


@app.post("/experiments")
def create_experiment(req: CreateExperimentRequest):
    """Create a new experiment.

    Args:
        req: CreateExperimentRequest with name and optional description

    Returns:
        dict with experiment ID and success message

    Raises:
        HTTPException: If validation fails
    """
    experiment = Experiment(
        name=req.name, description=req.description.strip() if req.description else None
    )
    save_experiment(experiment)
    return {"id": experiment.id, "message": "Experiment created"}


@app.get("/experiments/{experiment_id}")
def get_experiment(experiment_id: str):
    """Get full experiment details."""
    experiment = load_experiment(experiment_id)
    return experiment.model_dump()


@app.delete("/experiments/{experiment_id}")
def delete_experiment(experiment_id: str):
    """Delete an experiment."""
    path = get_experiment_path(experiment_id)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Experiment not found")
    path.unlink()
    return {"message": "Experiment deleted"}


@app.post("/experiments/{experiment_id}/questions")
def add_question(experiment_id: str, req: AddQuestionRequest):
    """Add a question to an experiment.

    Args:
        experiment_id: ID of the experiment
        req: AddQuestionRequest with question text, type, and optional ground truth

    Returns:
        dict with question ID and success message

    Raises:
        HTTPException: If experiment not found or validation fails
    """
    experiment = load_experiment(experiment_id)
    question = Question(
        text=req.text,
        question_type=req.question_type,
        ground_truth=req.ground_truth.strip() if req.ground_truth else None,
    )
    experiment.questions.append(question)
    save_experiment(experiment)
    return {"id": question.id, "message": "Question added"}


@app.post("/experiments/{experiment_id}/responses")
def add_response(experiment_id: str, req: AddResponseRequest):
    """Add a model response to a question.

    Args:
        experiment_id: ID of the experiment
        req: AddResponseRequest with question_id, model name, and content

    Returns:
        dict with response ID and success message

    Raises:
        HTTPException: If experiment or question not found, or validation fails
    """
    experiment = load_experiment(experiment_id)

    # Find the question
    question = None
    for q in experiment.questions:
        if q.id == req.question_id:
            question = q
            break

    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    response = Response(model=req.model, content=req.content)
    question.responses.append(response)
    save_experiment(experiment)
    return {"id": response.id, "message": "Response added"}


@app.post("/experiments/{experiment_id}/rankings")
def add_ranking(experiment_id: str, req: AddRankingRequest):
    """Add a ranking (one model judging others' responses)."""
    experiment = load_experiment(experiment_id)

    # Find the question
    question = None
    for q in experiment.questions:
        if q.id == req.question_id:
            question = q
            break

    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    # Validate that all ranking response IDs exist in the question's responses
    response_ids = {r.id for r in question.responses}
    invalid_ids = [rid for rid in req.rankings if rid not in response_ids]
    if invalid_ids:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid response IDs in ranking: {', '.join(invalid_ids)}. Available response IDs: {', '.join(response_ids)}",
        )

    if len(req.rankings) != len(response_ids) or len(set(req.rankings)) != len(req.rankings):
        raise HTTPException(
            status_code=400,
            detail="A ranking must include every response exactly once",
        )

    ranking = Ranking(
        judge=req.judge, rankings=req.rankings, confidence=req.confidence, reasoning=req.reasoning
    )
    question.rankings.append(ranking)
    save_experiment(experiment)
    return {"id": ranking.id, "message": "Ranking added"}


@app.post("/experiments/{experiment_id}/compute")
def compute_results(experiment_id: str, req: ComputeResultsRequest):
    """Compute aggregation results for a question using specified method."""
    experiment = load_experiment(experiment_id)

    # Find the question
    question = None
    for q in experiment.questions:
        if q.id == req.question_id:
            question = q
            break

    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    if len(question.rankings) == 0:
        raise HTTPException(status_code=400, detail="No rankings available")

    # Get candidate IDs (response IDs)
    candidates = [r.id for r in question.responses]

    # Map method name to function
    methods = {
        "borda": borda_count,
        "weighted_borda": weighted_borda,
        "copeland": copeland_score,
        "plurality": plurality,
        "ranked_pairs": ranked_pairs,
        "schulze": schulze_method,
        "stv": stv_instant_runoff,
        "approval": approval_voting,
    }

    if req.method not in methods:
        raise HTTPException(status_code=400, detail=f"Unknown method: {req.method}")

    # Compute scores
    scores = methods[req.method](question.rankings, candidates)
    winner_id = get_winner(scores)

    # Find the winning response
    winner_response = None
    for r in question.responses:
        if r.id == winner_id:
            winner_response = r
            break

    # Build response ID to model mapping
    id_to_model = {r.id: r.model for r in question.responses}

    return {
        "method": req.method,
        "scores": {id_to_model.get(k, k): v for k, v in scores.items()},
        "winner": {
            "id": winner_id,
            "model": winner_response.model if winner_response else None,
            "content": winner_response.content if winner_response else None,
        },
        "raw_scores": scores,
    }


@app.get("/experiments/{experiment_id}/compare")
def compare_all_methods(experiment_id: str, question_id: str):
    """Compare all aggregation methods for a question."""
    experiment = load_experiment(experiment_id)

    # Find the question
    question = None
    for q in experiment.questions:
        if q.id == question_id:
            question = q
            break

    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    if len(question.rankings) == 0:
        raise HTTPException(status_code=400, detail="No rankings available")

    candidates = [r.id for r in question.responses]
    id_to_model = {r.id: r.model for r in question.responses}

    methods = {
        "borda": borda_count,
        "weighted_borda": weighted_borda,
        "copeland": copeland_score,
        "plurality": plurality,
        "ranked_pairs": ranked_pairs,
        "schulze": schulze_method,
        "stv": stv_instant_runoff,
        "approval": approval_voting,
    }

    results = {}
    for method_name, method_fn in methods.items():
        scores = method_fn(question.rankings, candidates)
        winner_id = get_winner(scores)
        results[method_name] = {
            "scores": {id_to_model.get(k, k): v for k, v in scores.items()},
            "winner": id_to_model.get(winner_id, winner_id),
        }

    # Check agreement
    winners = [r["winner"] for r in results.values()]
    unanimous = len(set(winners)) == 1

    return {
        "question_id": question_id,
        "question_text": question.text,
        "methods": results,
        "unanimous": unanimous,
        "ground_truth": question.ground_truth,
    }


# === Export Endpoints ===


@app.get("/experiments/{experiment_id}/export")
def export_experiment(experiment_id: str, format: str = "json"):
    """
    Export full experiment data.

    Query params:
        format: 'json' or 'csv'
    """
    from . import export_utils

    experiment = load_experiment(experiment_id)

    if format == "csv":
        csv_data = export_utils.export_experiment_to_csv(experiment)
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type=CSV_MEDIA_TYPE,
            headers={
                "Content-Disposition": f"attachment; filename={experiment.name.replace(' ', '_')}_{experiment_id}.csv"
            },
        )
    else:  # JSON
        json_data = export_utils.export_experiment_to_json(experiment)
        return StreamingResponse(
            io.BytesIO(json.dumps(json_data, indent=2, default=str).encode()),
            media_type="application/json",
            headers={
                "Content-Disposition": f"attachment; filename={experiment.name.replace(' ', '_')}_{experiment_id}.json"
            },
        )


@app.get("/experiments/{experiment_id}/export/rankings")
def export_experiment_rankings(experiment_id: str):
    """Export rankings as CSV."""
    from . import export_utils

    experiment = load_experiment(experiment_id)
    csv_data = export_utils.export_rankings_to_csv(experiment)

    return StreamingResponse(
        io.StringIO(csv_data),
        media_type=CSV_MEDIA_TYPE,
        headers={
            "Content-Disposition": f"attachment; filename={experiment.name.replace(' ', '_')}_{experiment_id}_rankings.csv"
        },
    )


@app.get("/experiments/{experiment_id}/questions/{question_id}/export")
def export_question(experiment_id: str, question_id: str, format: str = "json"):
    """Export single question with responses and rankings."""
    from . import export_utils
    from .models import Experiment

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(question_id)

    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    if format == "csv":
        # Create mini-experiment with just this question
        mini_exp = Experiment(id=experiment.id, name=experiment.name, questions=[question])
        csv_data = export_utils.export_experiment_to_csv(mini_exp)
        return StreamingResponse(
            io.StringIO(csv_data),
            media_type=CSV_MEDIA_TYPE,
            headers={"Content-Disposition": f"attachment; filename=question_{question_id}.csv"},
        )
    else:
        return {
            "question": question.model_dump(),
            "experiment_id": experiment_id,
            "experiment_name": experiment.name,
        }


# === Agreement Matrix Endpoint ===


@app.get("/experiments/{experiment_id}/questions/{question_id}/agreement")
def get_agreement_matrix(experiment_id: str, question_id: str):
    """
    Calculate agreement matrix between judges.

    Returns:
        - judges: List of judge names
        - matrix: 2D array of agreement scores (0-1)
        - diversity_score: Overall diversity metric
    """
    from .aggregation import agreement_matrix, diversity_score

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(question_id)

    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    if len(question.rankings) < 2:
        raise HTTPException(
            status_code=400, detail="Need at least 2 rankings for agreement analysis"
        )

    candidates = [r.id for r in question.responses]
    matrix_data = agreement_matrix(question.rankings, candidates)
    diversity = diversity_score(question.rankings, candidates)

    # Convert dict of dicts to array format for frontend
    judges = list(matrix_data.keys())
    matrix_array = [[matrix_data[j1][j2] for j2 in judges] for j1 in judges]

    return {"judges": judges, "matrix": matrix_array, "diversity_score": diversity}


# === Automation Endpoints ===


@app.get("/config/models")
async def get_available_models():
    """Get list of available models for automation."""
    if not config.is_automation_available():
        return {"available": False, "message": "OpenRouter API key not configured"}

    models = await config.get_available_models_async()
    return {"available": True, "models": models}


@app.post("/experiments/{experiment_id}/automate/responses")
async def start_response_collection(experiment_id: str, req: AutomateResponsesRequest):
    """Start automated response collection job."""
    if not config.is_automation_available():
        raise HTTPException(status_code=503, detail=ERR_AUTOMATION_UNAVAILABLE)

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(req.question_id)
    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    # Validate models
    available = await config.get_available_models_async()
    available_ids = {m["id"] for m in available}
    for model_id in req.models:
        if model_id not in available_ids:
            raise HTTPException(status_code=400, detail=f"Model not found: {model_id}")

    # Start background job
    job_id = automation.start_response_job(
        question_text=question.text,
        models=req.models,
        experiment_id=experiment_id,
        question_id=req.question_id,
    )

    return {"job_id": job_id, "status": "queued", "message": "Response collection started"}


@app.post("/experiments/{experiment_id}/automate/rankings")
async def start_ranking_collection(experiment_id: str, req: AutomateRankingsRequest):
    """Start automated ranking collection job."""
    if not config.is_automation_available():
        raise HTTPException(status_code=503, detail=ERR_AUTOMATION_UNAVAILABLE)

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(req.question_id)
    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    if len(question.responses) == 0:
        raise HTTPException(status_code=400, detail="Question has no responses to rank")

    # Validate judges
    available = await config.get_available_models_async()
    available_ids = {m["id"] for m in available}
    for judge_id in req.judges:
        if judge_id not in available_ids:
            raise HTTPException(status_code=400, detail=f"Judge model not found: {judge_id}")

    # Start background job
    job_id = automation.start_ranking_job(
        question_text=question.text,
        responses=question.responses,
        judges=req.judges,
        experiment_id=experiment_id,
        question_id=req.question_id,
    )

    return {"job_id": job_id, "status": "queued", "message": "Ranking collection started"}


@app.post("/experiments/{experiment_id}/automate/deliberate")
async def start_deliberation(experiment_id: str, req: DeliberateRequest):
    """Start multi-round deliberation job."""
    if not config.is_automation_available():
        raise HTTPException(status_code=503, detail=ERR_AUTOMATION_UNAVAILABLE)

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(req.question_id)
    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    # Validate models
    available = await config.get_available_models_async()
    available_ids = {m["id"] for m in available}
    for model_id in req.models:
        if model_id not in available_ids:
            raise HTTPException(status_code=400, detail=f"Model not found: {model_id}")

    # Start background job
    job_id = automation.start_deliberation_job(
        question_text=question.text,
        models=req.models,
        max_rounds=req.max_rounds,
        experiment_id=experiment_id,
        question_id=req.question_id,
    )

    return {"job_id": job_id, "status": "queued", "message": "Deliberation started"}


@app.post("/experiments/{experiment_id}/questions/{question_id}/chairman-synthesis")
async def start_chairman_synthesis(
    experiment_id: str, question_id: str, req: ChairmanSynthesisRequest
):
    """Start chairman synthesis job to create final answer from responses and rankings."""
    if not config.is_automation_available():
        raise HTTPException(status_code=503, detail=ERR_AUTOMATION_UNAVAILABLE)

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(question_id)
    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    if not question.responses:
        raise HTTPException(status_code=400, detail="No responses available for synthesis")

    if not question.rankings:
        raise HTTPException(status_code=400, detail="No rankings available for synthesis")

    # Validate chairman model
    available = await config.get_available_models_async()
    available_ids = {m["id"] for m in available}
    if req.chairman_model not in available_ids:
        raise HTTPException(status_code=400, detail=f"Model not found: {req.chairman_model}")

    # Convert rankings to dict format for chairman_synthesis
    rankings_dict = [
        {
            "judge": r.judge,
            "rankings": [
                f"{response.model} (response {response.id}, round {response.round})"
                for response_id in r.rankings
                if (response := question.get_response_by_id(response_id)) is not None
            ],
            "confidence": r.confidence,
            "reasoning": r.reasoning,
        }
        for r in question.rankings
    ]

    # Start background job
    job_id = automation.start_chairman_job(
        question_text=question.text,
        responses=question.responses,
        rankings=rankings_dict,
        chairman_model=req.chairman_model,
        experiment_id=experiment_id,
        question_id=question_id,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": "Chairman synthesis started",
    }


@app.post("/experiments/{experiment_id}/automate/self-consistency")
async def start_self_consistency(experiment_id: str, req: SelfConsistencyRequest):
    """Start self-consistency sampling job for a single model."""
    if not config.is_automation_available():
        raise HTTPException(status_code=503, detail=ERR_AUTOMATION_UNAVAILABLE)

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(req.question_id)
    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    available = await config.get_available_models_async()
    available_ids = {m["id"] for m in available}
    if req.model not in available_ids:
        raise HTTPException(status_code=400, detail=f"Model not found: {req.model}")

    job_id = automation.start_self_consistency_job(
        question_text=question.text,
        model=req.model,
        num_samples=req.num_samples,
        temperature=req.temperature,
        experiment_id=experiment_id,
        question_id=req.question_id,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Self-consistency sampling started ({req.num_samples} samples)",
    }


@app.post("/experiments/{experiment_id}/automate/debate")
async def start_debate(experiment_id: str, req: DebateRequest):
    """Start adversarial debate job between models with judge."""
    if not config.is_automation_available():
        raise HTTPException(status_code=503, detail=ERR_AUTOMATION_UNAVAILABLE)

    experiment = load_experiment(experiment_id)
    question = experiment.get_question_by_id(req.question_id)
    if not question:
        raise HTTPException(status_code=404, detail=ERR_QUESTION_NOT_FOUND)

    available = await config.get_available_models_async()
    available_ids = {m["id"] for m in available}

    if len(set(req.debaters)) != len(req.debaters):
        raise HTTPException(status_code=400, detail="Debater models must be unique")

    for model_id in req.debaters + [req.judge_model]:
        if model_id not in available_ids:
            raise HTTPException(status_code=400, detail=f"Model not found: {model_id}")

    if req.judge_model in req.debaters:
        raise HTTPException(status_code=400, detail="Judge model cannot also be a debater")

    job_id = automation.start_debate_job(
        question_text=question.text,
        debaters=req.debaters,
        judge_model=req.judge_model,
        num_rounds=req.num_rounds,
        experiment_id=experiment_id,
        question_id=req.question_id,
    )

    return {
        "job_id": job_id,
        "status": "queued",
        "message": f"Debate started ({len(req.debaters)} debaters, {req.num_rounds} rounds)",
    }


@app.get("/experiments/{experiment_id}/automation/status/{job_id}")
def get_job_status(experiment_id: str, job_id: str):
    """Get status of an automation job."""
    job_progress = automation.get_job_status(job_id)
    if not job_progress or (
        job_progress.experiment_id is not None and job_progress.experiment_id != experiment_id
    ):
        raise HTTPException(status_code=404, detail="Job not found")

    progress = job_progress.progress or {}

    return {
        "job_id": job_id,
        "status": job_progress.status.value,
        "progress": progress,
        "started_at": job_progress.started_at,
        "completed_at": job_progress.completed_at,
        "errors": job_progress.errors,
        "results": job_progress.results or [],
    }


def sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, default=str)}\n\n"


def build_job_snapshot(job_id: str, job_progress) -> dict[str, Any]:
    return {
        "job_id": job_id,
        "status": job_progress.status.value,
        # Copy mutable structures so in-place updates (e.g., progress.update(...))
        # don't mutate the previous snapshot and suppress SSE change detection.
        "progress": dict(job_progress.progress or {}),
        "started_at": job_progress.started_at,
        "completed_at": job_progress.completed_at,
        "errors": list(job_progress.errors or []),
    }


def compute_sse_updates(job_id: str, job_progress, last_snapshot, last_results_len: int):
    if not job_progress:
        return (
            [
                sse_event(
                    "status",
                    {
                        "job_id": job_id,
                        "status": "failed",
                        "errors": [{"message": "Job not found"}],
                    },
                )
            ],
            last_snapshot,
            last_results_len,
            True,
        )

    events: list[str] = []

    results = job_progress.results or []
    if len(results) > last_results_len:
        new_items = results[last_results_len:]
        last_results_len = len(results)
        events.append(sse_event("partial_results", {"job_id": job_id, "items": new_items}))

    snapshot = build_job_snapshot(job_id, job_progress)
    if snapshot != last_snapshot:
        last_snapshot = snapshot
        events.append(sse_event("status", snapshot))

    done = job_progress.status.value in ("completed", "failed")
    return events, last_snapshot, last_results_len, done


async def automation_event_generator(job_id: str):
    last_snapshot = None
    last_results_len = 0

    while True:
        job_progress = automation.get_job_status(job_id)
        events, last_snapshot, last_results_len, done = compute_sse_updates(
            job_id, job_progress, last_snapshot, last_results_len
        )

        for event in events:
            yield event

        if done:
            return

        # Keep the connection alive and avoid tight-looping.
        yield ": ping\n\n"
        await asyncio.sleep(0.5)


@app.get("/experiments/{experiment_id}/automation/stream/{job_id}")
async def stream_job_status(experiment_id: str, job_id: str):
    """Stream automation job status updates via Server-Sent Events (SSE)."""
    job_progress = automation.get_job_status(job_id)
    if not job_progress or (
        job_progress.experiment_id is not None and job_progress.experiment_id != experiment_id
    ):
        raise HTTPException(status_code=404, detail="Job not found")

    return StreamingResponse(
        automation_event_generator(job_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # nginx: don't buffer SSE
            "X-Accel-Buffering": "no",
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

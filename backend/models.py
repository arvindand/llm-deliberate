"""Data models for LLM Deliberate."""

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


def generate_id() -> str:
    """Generate a short unique identifier."""
    return str(uuid.uuid4())[:8]


def utc_now() -> datetime:
    """Return current UTC time (timezone-aware)."""
    return datetime.now(timezone.utc)


class QuestionType(str, Enum):
    FACTUAL = "factual"
    REASONING = "reasoning"
    SUBJECTIVE = "subjective"
    CREATIVE = "creative"


class Response(BaseModel):
    """A single model's response to a question."""

    id: str = Field(default_factory=generate_id)
    model: str  # e.g., "gpt-4o", "claude-sonnet", "gemini-pro"
    content: str = Field(min_length=1)
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)  # Tokens, latency, cost, etc.
    source: Literal["manual", "automated"] = "manual"  # Whether manually entered or API-generated
    round: int = Field(default=1, ge=1)  # Deliberation round (1 = initial, 2+ = refined)


class Ranking(BaseModel):
    """A single judge's ranking of responses."""

    id: str = Field(default_factory=generate_id)
    judge: str
    rankings: list[str]
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    reasoning: str | None = None
    created_at: datetime = Field(default_factory=utc_now)
    source: Literal["manual", "automated"] = "manual"
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChairmanSynthesis(BaseModel):
    """A persisted final answer produced by a chairman model."""

    content: str = Field(min_length=1)
    chairman_model: str
    job_id: str | None = None
    created_at: datetime = Field(default_factory=utc_now)


class Question(BaseModel):
    """A question in an experiment with its responses and rankings."""

    id: str = Field(default_factory=generate_id)
    text: str
    question_type: QuestionType
    ground_truth: str | None = None  # For factual/reasoning questions
    responses: list[Response] = Field(default_factory=list)
    rankings: list[Ranking] = Field(default_factory=list)
    chairman_synthesis: ChairmanSynthesis | None = None
    created_at: datetime = Field(default_factory=utc_now)
    max_rounds: int = Field(default=1, ge=1)  # Maximum deliberation rounds for this question
    current_round: int = Field(default=1, ge=1)  # Current deliberation round number

    def get_response_by_id(self, response_id: str) -> Response | None:
        for r in self.responses:
            if r.id == response_id:
                return r
        return None

    def get_response_by_model(self, model: str) -> Response | None:
        for r in self.responses:
            if r.model == model:
                return r
        return None


class Experiment(BaseModel):
    """A collection of questions for a deliberation experiment."""

    id: str = Field(default_factory=generate_id)
    name: str
    description: str | None = None
    questions: list[Question] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

    # Experiment configuration
    models: list[str] = Field(
        default_factory=lambda: ["gpt-4o", "claude-sonnet", "gemini-pro", "llama-3"]
    )

    def get_question_by_id(self, question_id: str) -> Question | None:
        for q in self.questions:
            if q.id == question_id:
                return q
        return None


class AggregationMethod(str, Enum):
    """Available aggregation methods."""

    PLURALITY = "plurality"
    BORDA = "borda"
    WEIGHTED_BORDA = "weighted_borda"
    COPELAND = "copeland"
    RANKED_PAIRS = "ranked_pairs"
    SCHULZE = "schulze"
    STV = "stv"
    APPROVAL = "approval"

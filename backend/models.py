"""
Data models for LLM Deliberate.
"""
from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime, timezone
from enum import Enum
import uuid


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
    content: str
    created_at: datetime = Field(default_factory=utc_now)
    metadata: dict = Field(default_factory=dict)  # For tokens, latency, cost, etc.
    source: Literal["manual", "automated"] = "manual"  # Whether manually entered or API-generated
    round: int = 1  # Deliberation round (1 = initial, 2+ = refined)


class Ranking(BaseModel):
    """A single judge's ranking of responses."""
    id: str = Field(default_factory=generate_id)
    judge: str  # The model doing the judging
    rankings: list[str]  # Response IDs in order, best to worst
    confidence: float = 1.0  # 0-1, for weighted methods
    reasoning: Optional[str] = None  # The judge's explanation
    created_at: datetime = Field(default_factory=utc_now)
    source: Literal["manual", "automated"] = "manual"  # Whether manually entered or API-generated


class Question(BaseModel):
    """A question in an experiment with its responses and rankings."""
    id: str = Field(default_factory=generate_id)
    text: str
    question_type: QuestionType
    ground_truth: Optional[str] = None  # For factual/reasoning questions
    responses: list[Response] = Field(default_factory=list)
    rankings: list[Ranking] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    max_rounds: int = 1  # Maximum deliberation rounds for this question
    current_round: int = 1  # Current deliberation round number
    
    def get_response_by_id(self, response_id: str) -> Optional[Response]:
        for r in self.responses:
            if r.id == response_id:
                return r
        return None
    
    def get_response_by_model(self, model: str) -> Optional[Response]:
        for r in self.responses:
            if r.model == model:
                return r
        return None


class Experiment(BaseModel):
    """A collection of questions for a deliberation experiment."""
    id: str = Field(default_factory=generate_id)
    name: str
    description: Optional[str] = None
    questions: list[Question] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)
    
    # Experiment configuration
    models: list[str] = Field(default_factory=lambda: [
        "gpt-4o",
        "claude-sonnet", 
        "gemini-pro",
        "llama-3"
    ])
    
    def get_question_by_id(self, question_id: str) -> Optional[Question]:
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

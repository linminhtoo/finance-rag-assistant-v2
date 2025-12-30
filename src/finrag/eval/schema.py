from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


EvalKind = Literal["factual", "open_ended", "refusal", "distractor", "comparison"]


class EvidenceChunk(BaseModel):
    """
    A concrete, reviewable reference to a document chunk.

    This is used as "gold evidence" for factual questions (ground-truth answers)
    and as audit context for human labeling and judge analysis.
    """

    doc_id: str
    chunk_id: str
    source: str | None = None
    headings: list[str] = Field(default_factory=list)
    page_no: int | None = None
    section_path: str | None = None
    snippet: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


ScaleUnits = Literal["units", "thousands", "millions", "billions"]
class NumericAnswer(BaseModel):
    """
    Numeric ground truth for factual questions.
    """

    value: float
    unit: str = "USD"
    scale: ScaleUnits | None = None
    raw: str | None = None


class FactualSpec(BaseModel):
    metric: str
    expected_numeric: NumericAnswer
    golden_evidence: EvidenceChunk


class OpenEndedSpec(BaseModel):
    rubric_id: str = "faithfulness_v1"
    target_ticker: str | None = None
    target_year: int | None = None


RefusalReason = Literal[
    "non_investment",
    "unknown_company",
    "prompt_injection",
    "harmful_or_toxic",
    "other",
]


class RefusalSpec(BaseModel):
    """
    Refusal / out-of-scope questions.

    These are intended to test whether the system appropriately refuses (or
    defers) when the question is unrelated, malicious, or lacks in-database
    context.
    """

    reason: RefusalReason
    rubric_id: str = "refusal_v1"

    # For "unknown_company" cases.
    target_company: str | None = None
    target_ticker: str | None = None

    # Optional dataset signature (useful when the corpus evolves over time).
    known_tickers_sample: list[str] = Field(default_factory=list)


DistractorKind = Literal["emotion", "portfolio_story", "rambling", "off_tangent_finance", "other"]


class DistractorSpec(BaseModel):
    """
    Questions that contain extra distracting user-provided context.

    The system should answer the main question and not over-index on the distractor.
    """

    main_question: str
    distractor_text: str
    distractor_kind: DistractorKind
    rubric_id: str = "focus_v1"

    target_tickers: list[str] = Field(default_factory=list)
    target_year: int | None = None


class ComparisonSpec(BaseModel):
    """
    Questions that compare multiple companies.
    """

    target_tickers: list[str] = Field(default_factory=list)
    target_companies: list[str] = Field(default_factory=list)
    target_year: int | None = None
    rubric_id: str = "comparison_v1"


class EvalQuery(BaseModel):
    id: str
    question: str
    kind: EvalKind
    tags: list[str] = Field(default_factory=list)
    created_at: datetime | None = None

    # Exactly one of these should be present depending on `kind`.
    factual: FactualSpec | None = None
    open_ended: OpenEndedSpec | None = None
    refusal: RefusalSpec | None = None
    distractor: DistractorSpec | None = None
    comparison: ComparisonSpec | None = None

    generator: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_kind(self) -> "EvalQuery":
        spec_fields = {
            "factual": self.factual,
            "open_ended": self.open_ended,
            "refusal": self.refusal,
            "distractor": self.distractor,
            "comparison": self.comparison,
        }
        present = [k for k, v in spec_fields.items() if v is not None]

        if self.kind == "factual":
            if self.factual is None:
                raise ValueError("kind='factual' requires `factual`")
            if len(present) != 1:
                raise ValueError("kind='factual' forbids non-factual specs")
        elif self.kind == "open_ended":
            if self.open_ended is None:
                raise ValueError("kind='open_ended' requires `open_ended`")
            if len(present) != 1:
                raise ValueError("kind='open_ended' forbids non-open_ended specs")
        elif self.kind == "refusal":
            if self.refusal is None:
                raise ValueError("kind='refusal' requires `refusal`")
            if len(present) != 1:
                raise ValueError("kind='refusal' forbids non-refusal specs")
        elif self.kind == "distractor":
            if self.distractor is None:
                raise ValueError("kind='distractor' requires `distractor`")
            if len(present) != 1:
                raise ValueError("kind='distractor' forbids non-distractor specs")
        elif self.kind == "comparison":
            if self.comparison is None:
                raise ValueError("kind='comparison' requires `comparison`")
            if len(present) != 1:
                raise ValueError("kind='comparison' forbids non-comparison specs")
        else:  # pragma: no cover
            raise ValueError(f"Unsupported kind: {self.kind}")
        return self


class RetrievedChunk(BaseModel):
    chunk_id: str
    doc_id: str
    page_no: int | None = None
    headings: list[str] = Field(default_factory=list)
    score: float
    source: str | None = None
    preview: str | None = None
    text: str | None = None
    context: str | None = None
    metadata: dict[str, Any] | None = None


class EvalGeneration(BaseModel):
    """
    A single model run over a single `EvalQuery`.
    """

    query_id: str
    kind: EvalKind
    question: str

    created_at: datetime | None = None
    settings: dict[str, Any] = Field(default_factory=dict)

    draft_answer: str | None = None
    final_answer: str | None = None
    top_chunks: list[RetrievedChunk] = Field(default_factory=list)

    timing_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None


class JudgeResult(BaseModel):
    judge_id: str
    prediction: Literal[0, 1]
    explanation: str | None = None
    raw: str | None = None


class EvalScore(BaseModel):
    query_id: str
    kind: EvalKind
    created_at: datetime | None = None

    retrieval: dict[str, Any] = Field(default_factory=dict)
    answer: dict[str, Any] = Field(default_factory=dict)
    judges: list[JudgeResult] = Field(default_factory=list)

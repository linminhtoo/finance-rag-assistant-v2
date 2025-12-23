from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field


class Evidence(BaseModel):
    doc_id: str
    chunk_id: str | None = None
    source: str | None = None
    locator: dict[str, Any] = Field(default_factory=dict)
    snippet: str | None = None


class NumericAnswer(BaseModel):
    value: float | None = None
    unit: str = "USD"
    scale: Literal["units", "thousands", "millions", "billions"] | None = None
    raw: str | None = None


class Verification(BaseModel):
    status: Literal["unverified", "verified", "rejected"] = "unverified"
    verified_by: str | None = None
    verified_at: datetime | None = None
    notes: str | None = None


class EvalItem(BaseModel):
    id: str
    question: str
    kind: Literal["quantitative", "qualitative", "mixed"]
    tags: list[str] = Field(default_factory=list)

    expected_numeric: NumericAnswer | None = None
    expected_key_points: list[str] | None = None
    expected_series: list[dict[str, Any]] | None = None

    evidences: list[Evidence] = Field(default_factory=list)

    generation: dict[str, Any] = Field(default_factory=dict)
    verification: Verification = Field(default_factory=Verification)

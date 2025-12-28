from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Literal

AnswerStyle = Literal["concise", "normal", "detailed"]


@dataclass(frozen=True)
class GenerationPreset:
    key: str
    label: str
    description: str
    top_k_retrieve: int
    top_k_rerank: int
    draft_max_tokens: int
    final_max_tokens: int
    enable_rerank: bool
    enable_refine: bool
    answer_style: AnswerStyle
    draft_temperature: float = 0.1

    def to_public_dict(self) -> dict[str, object]:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "top_k_retrieve": self.top_k_retrieve,
            "top_k_rerank": self.top_k_rerank,
            "draft_max_tokens": self.draft_max_tokens,
            "final_max_tokens": self.final_max_tokens,
            "enable_rerank": self.enable_rerank,
            "enable_refine": self.enable_refine,
            "answer_style": self.answer_style,
            "draft_temperature": self.draft_temperature,
        }


@dataclass(frozen=True)
class GenerationSettings:
    mode: str
    top_k_retrieve: int
    top_k_rerank: int
    draft_max_tokens: int
    final_max_tokens: int
    enable_rerank: bool
    enable_refine: bool
    answer_style: AnswerStyle
    draft_temperature: float


_PRESETS: dict[str, GenerationPreset] = {
    "quick": GenerationPreset(
        key="quick",
        label="Quick",
        description="Fast + concise. Uses fewer chunks; skips reranking and verification.",
        top_k_retrieve=12,
        top_k_rerank=6,
        draft_max_tokens=16_384,
        final_max_tokens=16_384,
        enable_rerank=False,
        enable_refine=False,
        answer_style="concise",
        draft_temperature=0.1,
    ),
    "normal": GenerationPreset(
        key="normal",
        label="Normal",
        description="Balanced quality/speed. Uses reranking + verification.",
        top_k_retrieve=30,
        top_k_rerank=8,
        draft_max_tokens=65_536,
        final_max_tokens=32_768,
        enable_rerank=True,
        enable_refine=True,
        answer_style="normal",
        draft_temperature=0.1,
    ),
    "thinking": GenerationPreset(
        key="thinking",
        label="Thinking",
        description="Higher recall + deeper report. Retrieves/reranks more and verifies.",
        top_k_retrieve=60,
        top_k_rerank=20,
        draft_max_tokens=65_536,
        final_max_tokens=45_000,
        enable_rerank=True,
        enable_refine=True,
        answer_style="detailed",
        draft_temperature=0.1,
    ),
}


def list_generation_presets() -> list[GenerationPreset]:
    return list(_PRESETS.values())


def default_mode() -> str:
    return (os.getenv("FINRAG_DEFAULT_MODE") or "normal").strip().lower() or "normal"


def get_preset(mode: str | None) -> GenerationPreset:
    key = (mode or "").strip().lower()
    if key in _PRESETS:
        return _PRESETS[key]
    return _PRESETS[default_mode()]


def _pos_int(value: int | None, fallback: int) -> int:
    if value is None:
        return fallback
    try:
        v = int(value)
    except Exception:
        return fallback
    return v if v > 0 else fallback


def resolve_generation_settings(
    *,
    mode: str | None,
    top_k_retrieve: int | None = None,
    top_k_rerank: int | None = None,
    draft_max_tokens: int | None = None,
    final_max_tokens: int | None = None,
    enable_rerank: bool | None = None,
    enable_refine: bool | None = None,
    answer_style: AnswerStyle | None = None,
    draft_temperature: float | None = None,
) -> GenerationSettings:
    preset = get_preset(mode)

    resolved_top_k_retrieve = _pos_int(top_k_retrieve, preset.top_k_retrieve)
    resolved_top_k_rerank = _pos_int(top_k_rerank, preset.top_k_rerank)
    resolved_top_k_rerank = min(resolved_top_k_rerank, resolved_top_k_retrieve)

    resolved_draft_max_tokens = _pos_int(draft_max_tokens, preset.draft_max_tokens)
    resolved_final_max_tokens = _pos_int(final_max_tokens, preset.final_max_tokens)

    resolved_enable_rerank = preset.enable_rerank if enable_rerank is None else bool(enable_rerank)
    resolved_enable_refine = preset.enable_refine if enable_refine is None else bool(enable_refine)

    resolved_style: AnswerStyle = preset.answer_style if answer_style is None else answer_style
    resolved_temp = preset.draft_temperature if draft_temperature is None else float(draft_temperature)

    return GenerationSettings(
        mode=preset.key,
        top_k_retrieve=resolved_top_k_retrieve,
        top_k_rerank=resolved_top_k_rerank,
        draft_max_tokens=resolved_draft_max_tokens,
        final_max_tokens=resolved_final_max_tokens,
        enable_rerank=resolved_enable_rerank,
        enable_refine=resolved_enable_refine,
        answer_style=resolved_style,
        draft_temperature=resolved_temp,
    )


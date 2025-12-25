from __future__ import annotations

import math
import re
from typing import Iterable


def recall_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return math.nan
    return 1.0 if any(rid in relevant_ids for rid in retrieved_ids[:k]) else 0.0


def mrr(retrieved_ids: list[str], relevant_ids: set[str]) -> float:
    if not relevant_ids:
        return math.nan
    for idx, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / idx
    return 0.0


def coverage_at_k(retrieved_ids: list[str], relevant_ids: set[str], k: int) -> float:
    if not relevant_ids:
        return math.nan
    got = set(retrieved_ids[:k]) & relevant_ids
    return len(got) / len(relevant_ids)


_NUM_RE = re.compile(r"(?P<num>\(?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?)")


def extract_numbers(text: str) -> list[float]:
    out: list[float] = []
    for m in _NUM_RE.finditer(text):
        raw = m.group("num")
        s = raw.strip()
        negative = False
        if s.startswith("(") and s.endswith(")"):
            negative = True
            s = s[1:-1].strip()
        s = s.replace("$", "").replace(",", "").strip()
        try:
            val = float(s)
        except ValueError:
            continue
        out.append(-val if negative else val)
    return out


def _scale_factor_from_text(text: str) -> float | None:
    t = text.lower()
    if "billion" in t:
        return 1e9
    if "million" in t:
        return 1e6
    if "thousand" in t:
        return 1e3
    return None


def _scale_factor(scale: str | None) -> float:
    if scale is None or scale == "units":
        return 1.0
    if scale == "thousands":
        return 1e3
    if scale == "millions":
        return 1e6
    if scale == "billions":
        return 1e9
    return 1.0


def best_numeric_match(
    predicted_text: str,
    expected_value: float,
    *,
    expected_scale: str | None = None,
    rel_tol: float = 0.01,
    abs_tol: float = 1e-6,
) -> dict[str, float | bool | None]:
    """
    Try to match an expected numeric value against numbers in a model response.

    We search for candidate numbers and compare after applying scale heuristics
    (e.g. "in millions" vs raw table value).
    """
    nums = extract_numbers(predicted_text)
    if not nums:
        return {"matched": False, "best_rel_error": math.inf, "best_pred": None}

    expected = expected_value * _scale_factor(expected_scale)
    hinted = _scale_factor_from_text(predicted_text)
    candidate_factors: list[float] = [1.0]
    if hinted:
        candidate_factors.append(hinted)
    candidate_factors.extend([1e3, 1e6, 1e9])

    best_rel = math.inf
    best_pred = None
    for n in nums:
        for f in candidate_factors:
            pred = n * f
            denom = max(abs(expected), 1.0)
            rel = abs(pred - expected) / denom
            if rel < best_rel:
                best_rel = rel
                best_pred = pred

    matched = best_pred is not None and (best_rel <= rel_tol or abs(best_pred - expected) <= abs_tol)
    return {"matched": matched, "best_rel_error": best_rel, "best_pred": best_pred}


_CITE_RE = re.compile(r"\[doc=([^\s\]]+)")


def cited_doc_ids(text: str) -> set[str]:
    return {m.group(1) for m in _CITE_RE.finditer(text)}


def keyword_coverage(text: str, key_points: Iterable[str]) -> float:
    """
    Very lightweight proxy for qualitative coverage: fraction of key point
    sentences with at least one non-trivial token present in the answer.
    """
    points = [p.strip() for p in key_points if p and p.strip()]
    if not points:
        return math.nan

    t = text.lower()
    covered = 0
    for p in points:
        tokens = [w for w in re.findall(r"[a-zA-Z]{4,}", p.lower())][:8]
        if tokens and any(tok in t for tok in tokens):
            covered += 1
    return covered / len(points)

from __future__ import annotations

import random
import re
import uuid
from datetime import datetime, timezone
from typing import Any, Iterable

from finrag.dataclasses import DocChunk
from finrag.eval.schema import Evidence, EvalItem, NumericAnswer
from finrag.llm_clients import LLMClient


_SCALE_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"(?i)\b(in\s+millions)\b"), "millions"),
    (re.compile(r"(?i)\b(in\s+thousands)\b"), "thousands"),
    (re.compile(r"(?i)\b(in\s+billions)\b"), "billions"),
]


_METRIC_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("net income", re.compile(r"(?i)\bnet\s+income\b")),
    ("total revenue", re.compile(r"(?i)\btotal\s+revenue\b|\bnet\s+revenue\b|\btotal\s+net\s+revenue\b")),
    ("operating income", re.compile(r"(?i)\boperating\s+income\b")),
    ("gross profit", re.compile(r"(?i)\bgross\s+profit\b")),
    ("research and development", re.compile(r"(?i)\bresearch\s+and\s+development\b|\bR&D\b")),
    ("earnings per share", re.compile(r"(?i)\bearnings\s+per\s+share\b|\bEPS\b")),
]


_NUMBER_RE = re.compile(
    r"(?P<num>\(?\s*\$?\s*\d{1,3}(?:,\d{3})*(?:\.\d+)?\s*\)?)"
)

_PERIOD_END_RE = re.compile(
    r"(?i)\b(?:quarter|three\s+months)\s+ended\s+"
    r"(?P<month>january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+(?P<day>\d{1,2}),\s+(?P<year>\d{4})"
)

_MONTH_TO_NUM = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}


def _extract_period_end(text: str) -> str | None:
    m = _PERIOD_END_RE.search(text)
    if not m:
        return None
    month = _MONTH_TO_NUM[m.group("month").lower()]
    day = int(m.group("day"))
    year = int(m.group("year"))
    return f"{year:04d}-{month:02d}-{day:02d}"


def _detect_scale(text: str) -> str | None:
    for pattern, scale in _SCALE_HINTS:
        if pattern.search(text):
            return scale
    return None


def _parse_number(raw: str) -> float | None:
    s = raw.strip()
    negative = False
    if s.startswith("(") and s.endswith(")"):
        negative = True
        s = s[1:-1].strip()
    s = s.replace("$", "").replace(",", "").strip()
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def extract_metric_number_pairs(text: str, *, max_pairs: int = 3) -> list[tuple[str, NumericAnswer]]:
    """
    Heuristic numeric fact extraction for SEC filing text.

    Returns (metric_name, numeric_answer) pairs. This is intentionally "silver"
    ground truth meant for humans to verify later.
    """
    out: list[tuple[str, NumericAnswer]] = []
    scale = _detect_scale(text)
    for metric, metric_re in _METRIC_PATTERNS:
        m = metric_re.search(text)
        if not m:
            continue

        window = text[m.end() : m.end() + 400]
        num_m = _NUMBER_RE.search(window)
        if not num_m:
            continue

        raw = num_m.group("num")
        value = _parse_number(raw)
        out.append((metric, NumericAnswer(value=value, raw=raw, scale=scale)))
        if len(out) >= max_pairs:
            break
    return out


def _sentences(text: str, *, max_sentences: int = 8) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[\.\?\!])\s+", text)
    return [p.strip() for p in parts if p.strip()][:max_sentences]


_INVESTING_ANGLES: list[dict[str, Any]] = [
    {
        "tag": "rnd",
        "keywords": ["research", "development", "R&D", "innovation"],
        "templates": [
            "What does {ticker} say about its R&D investments and priorities in its {form} filed on {filing_date}?",
            "Summarize {ticker}'s R&D strategy and investment focus based on the {form} filed on {filing_date}.",
        ],
    },
    {
        "tag": "long_term_vision",
        "keywords": ["strategy", "long-term", "vision", "roadmap", "growth"],
        "templates": [
            "What is {ticker}'s long-term vision and strategic priorities according to its {form} filed on {filing_date}?",
            "Based on the {form} filed on {filing_date}, what are {ticker}'s stated long-term goals?",
        ],
    },
    {
        "tag": "market_uncertainty",
        "keywords": ["uncertain", "uncertainty", "macro", "inflation", "recession", "demand", "geopolitical"],
        "templates": [
            "What market uncertainties or macro risks does {ticker} highlight in its {form} filed on {filing_date}?",
            "According to the {form} filed on {filing_date}, what external uncertainties could affect {ticker}'s results?",
        ],
    },
    {
        "tag": "competitive_pressures",
        "keywords": ["competition", "competitive", "pricing", "rivals", "market share"],
        "templates": [
            "What competitive pressures does {ticker} discuss in its {form} filed on {filing_date}, and why do they matter?",
            "Summarize the competitive landscape described by {ticker} in the {form} filed on {filing_date}.",
        ],
    },
    {
        "tag": "competitive_advantage",
        "keywords": ["advantage", "differentiation", "proprietary", "patent", "brand", "moat", "scale"],
        "templates": [
            "What competitive advantages or disadvantages does {ticker} claim in its {form} filed on {filing_date}?",
            "Based on the {form} filed on {filing_date}, how does {ticker} describe its competitive positioning?",
        ],
    },
    {
        "tag": "differentiation_strategy",
        "keywords": ["differentiate", "differentiation", "unique", "platform", "ecosystem", "technology"],
        "templates": [
            "How does {ticker} describe its differentiation strategy in its {form} filed on {filing_date}?",
            "What factors does {ticker} cite as differentiators in the {form} filed on {filing_date}?",
        ],
    },
]


def _pick_key_points(text: str, keywords: list[str], *, max_points: int = 3) -> list[str]:
    sents = _sentences(text, max_sentences=12)
    if not sents:
        return []
    keyword_re = re.compile("|".join(re.escape(k) for k in keywords), re.IGNORECASE) if keywords else None
    picked: list[str] = []
    for s in sents:
        if keyword_re and keyword_re.search(s):
            picked.append(s)
        if len(picked) >= max_points:
            return picked
    return sents[:max_points]


def _maybe_paraphrase(llm: LLMClient | None, question: str) -> str:
    if llm is None:
        return question
    prompt = [
        {"role": "system", "content": "Rewrite questions to be clear, specific, and unambiguous. Do not add new facts."},
        {"role": "user", "content": f"Rewrite this question:\n{question}"},
    ]
    try:
        out = llm.chat(prompt, temperature=0.3).strip()
    except Exception:
        return question
    return out or question


def generate_eval_items_from_chunks(
    chunks: list[DocChunk],
    *,
    doc_meta_by_id: dict[str, dict[str, Any]] | None = None,
    n_quantitative: int = 20,
    n_qualitative: int = 20,
    n_mixed: int = 10,
    n_series: int = 0,
    seed: int = 0,
    llm_for_paraphrase: LLMClient | None = None,
) -> list[EvalItem]:
    rng = random.Random(seed)
    doc_meta_by_id = doc_meta_by_id or {}

    quant_candidates: list[EvalItem] = []
    qual_candidates: list[EvalItem] = []
    mixed_candidates: list[EvalItem] = []
    series_candidates: list[EvalItem] = []

    now = datetime.now(timezone.utc).isoformat()

    # For multi-doc quantitative questions.
    net_income_by_doc: dict[str, dict[str, Any]] = {}

    for chunk in chunks:
        meta = doc_meta_by_id.get(chunk.doc_id) or chunk.metadata or {}
        ticker = str(meta.get("ticker", "UNKNOWN")).upper()
        form = str(meta.get("form", "filing")).upper()
        filing_date = str(meta.get("filing_date", "unknown date"))

        # --- Quantitative candidates ---
        for metric, numeric in extract_metric_number_pairs(chunk.text, max_pairs=2):
            if metric == "net income" and chunk.doc_id not in net_income_by_doc and numeric.value is not None:
                net_income_by_doc[chunk.doc_id] = {
                    "ticker": ticker,
                    "form": form,
                    "filing_date": filing_date,
                    "period_end": _extract_period_end(chunk.text),
                    "value": numeric.value,
                    "scale": numeric.scale,
                    "evidence": chunk,
                }

            q = f"In {ticker}'s {form} filed on {filing_date}, what was {metric}?"
            q = _maybe_paraphrase(llm_for_paraphrase, q)
            quant_candidates.append(
                EvalItem(
                    id=str(uuid.uuid4()),
                    question=q,
                    kind="quantitative",
                    tags=["sec", "synthetic", "quantitative", metric, ticker, form],
                    expected_numeric=numeric,
                    evidences=[
                        Evidence(
                            doc_id=chunk.doc_id,
                            chunk_id=chunk.id,
                            source=chunk.source,
                            locator={"headings": chunk.headings, "page_no": chunk.page_no},
                            snippet=chunk.text[:800],
                        )
                    ],
                    generation={"generator": "metric_regex", "generated_at": now, "seed": seed},
                )
            )

            if metric == "research and development":
                q2 = (
                    f"In {ticker}'s {form} filed on {filing_date}, what was research and development expense, "
                    "and what does management cite as key drivers of the spending?"
                )
                q2 = _maybe_paraphrase(llm_for_paraphrase, q2)
                mixed_candidates.append(
                    EvalItem(
                        id=str(uuid.uuid4()),
                        question=q2,
                        kind="mixed",
                        tags=["sec", "synthetic", "mixed", "rnd", ticker, form],
                        expected_numeric=numeric,
                        expected_key_points=_pick_key_points(chunk.text, ["research", "development", "R&D", "innovation"]),
                        evidences=[
                            Evidence(
                                doc_id=chunk.doc_id,
                                chunk_id=chunk.id,
                                source=chunk.source,
                                locator={"headings": chunk.headings, "page_no": chunk.page_no},
                                snippet=chunk.text[:800],
                            )
                        ],
                        generation={"generator": "metric_regex+rnd", "generated_at": now, "seed": seed},
                    )
                )

        # --- Qualitative candidates ---
        combined = " ".join([*chunk.headings, chunk.text])
        for angle in _INVESTING_ANGLES:
            kw = angle["keywords"]
            if not any(re.search(re.escape(k), combined, flags=re.IGNORECASE) for k in kw):
                continue
            template = rng.choice(angle["templates"])
            q = template.format(ticker=ticker, form=form, filing_date=filing_date)
            q = _maybe_paraphrase(llm_for_paraphrase, q)
            qual_candidates.append(
                EvalItem(
                    id=str(uuid.uuid4()),
                    question=q,
                    kind="qualitative",
                    tags=["sec", "synthetic", "qualitative", angle["tag"], ticker, form],
                    expected_key_points=_pick_key_points(chunk.text, kw),
                    evidences=[
                        Evidence(
                            doc_id=chunk.doc_id,
                            chunk_id=chunk.id,
                            source=chunk.source,
                            locator={"headings": chunk.headings, "page_no": chunk.page_no},
                            snippet=chunk.text[:800],
                        )
                    ],
                    generation={"generator": "angle_keywords", "generated_at": now, "seed": seed, "angle": angle["tag"]},
                )
            )

    # --- Series questions (e.g. QoQ net income growth) ---
    if n_series > 0 and net_income_by_doc:
        by_ticker: dict[str, list[dict[str, Any]]] = {}
        for rec in net_income_by_doc.values():
            by_ticker.setdefault(rec["ticker"], []).append(rec)

        for ticker, recs in by_ticker.items():
            # Only attempt "QoQ" if we have quarterly filings.
            q_recs = [r for r in recs if str(r.get("form", "")).upper() == "10-Q"]
            if len(q_recs) < 4:
                continue

            def sort_key(r: dict[str, Any]) -> str:
                return str(r.get("period_end") or r.get("filing_date") or "")

            q_recs.sort(key=sort_key)

            series: list[dict[str, Any]] = []
            evidences: list[Evidence] = []
            prev = None
            for r in q_recs:
                value = float(r["value"])
                if prev is None:
                    growth = None
                else:
                    denom = abs(prev) if abs(prev) > 1e-9 else 1.0
                    growth = (value - prev) / denom
                prev = value

                period = r.get("period_end") or r.get("filing_date")
                series.append(
                    {
                        "period_end": period,
                        "net_income": value,
                        "net_income_scale": r.get("scale"),
                        "qoq_net_income_growth": growth,
                    }
                )
                ev_chunk = r["evidence"]
                evidences.append(
                    Evidence(
                        doc_id=ev_chunk.doc_id,
                        chunk_id=ev_chunk.id,
                        source=ev_chunk.source,
                        locator={"headings": ev_chunk.headings, "page_no": ev_chunk.page_no},
                        snippet=ev_chunk.text[:800],
                    )
                )

            start = series[0]["period_end"]
            end = series[-1]["period_end"]
            q = (
                f"For {ticker}, list the quarter-over-quarter net income growth rate for each quarter from {start} to {end}. "
                "Return a table with period_end and QoQ growth."
            )
            q = _maybe_paraphrase(llm_for_paraphrase, q)
            series_candidates.append(
                EvalItem(
                    id=str(uuid.uuid4()),
                    question=q,
                    kind="quantitative",
                    tags=["sec", "synthetic", "quantitative", "series", "qoq_net_income_growth", ticker, "10-Q"],
                    expected_series=series,
                    evidences=evidences,
                    generation={"generator": "series_net_income_qoq", "generated_at": now, "seed": seed},
                )
            )

    def _sample(items: list[EvalItem], n: int) -> list[EvalItem]:
        if n <= 0 or not items:
            return []
        rng.shuffle(items)
        return items[: min(n, len(items))]

    # De-dup questions (keep first)
    def _dedup(items: list[EvalItem]) -> list[EvalItem]:
        seen: set[str] = set()
        out: list[EvalItem] = []
        for it in items:
            key = it.question.strip().lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(it)
        return out

    quant = _sample(_dedup(quant_candidates), n_quantitative)
    qual = _sample(_dedup(qual_candidates), n_qualitative)
    mixed = _sample(_dedup(mixed_candidates), n_mixed)
    series_items = _sample(_dedup(series_candidates), n_series)
    return [*quant, *qual, *mixed, *series_items]

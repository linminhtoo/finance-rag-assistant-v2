from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from typing import Any

from finrag.eval.judges import FACTUAL_CORRECTNESS_V1, JudgeSpec, get_judge_spec, run_judge
from finrag.eval.metrics import best_numeric_match, cited_doc_ids
from finrag.eval.schema import EvalGeneration, EvalQuery, EvalScore, JudgeResult, RetrievedChunk
from finrag.llm_clients import LLMClient


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _rank(ids: list[str], target: str) -> int | None:
    for i, x in enumerate(ids, start=1):
        if x == target:
            return i
    return None


def build_context(chunks: list[RetrievedChunk], *, max_chars: int = 14_000) -> str:
    """
    Build a compact context string for judge prompts.
    """
    parts: list[str] = []
    used = 0
    for ch in chunks:
        head = f"[doc={ch.doc_id} chunk={ch.chunk_id} score={ch.score:.4f}]"
        body = (ch.text or ch.preview or "").strip()
        ctx = (ch.context or "").strip()
        block = head + "\n" + body
        if ctx:
            block += "\n\nContext:\n" + ctx
        block += "\n"
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts).strip()


def _expected_numeric_text(query: EvalQuery) -> str | None:
    if query.kind != "factual" or query.factual is None:
        return None
    n = query.factual.expected_numeric
    bits = [f"value={n.value}"]
    if n.scale:
        bits.append(f"scale={n.scale}")
    if n.unit:
        bits.append(f"unit={n.unit}")
    if n.raw:
        bits.append(f"raw={n.raw.strip()}")
    return ", ".join(bits)


_REFUSAL_HINT_RE = re.compile(
    r"(?i)\b(can(?:not|'t)\s+answer|cannot\s+answer|insufficient\s+information|"
    r"not\s+enough\s+information|not\s+provided\s+in\s+the\s+context|"
    r"based\s+on\s+the\s+provided\s+context|i\s+can\s+only\s+answer\s+using\s+the\s+provided\s+context)\b"
)


def _looks_like_refusal(answer: str) -> bool:
    return bool(_REFUSAL_HINT_RE.search(answer or ""))


def _chunk_tickers(chunks: list[RetrievedChunk]) -> list[str]:
    out: list[str] = []
    for ch in chunks or []:
        meta = ch.metadata if isinstance(ch.metadata, dict) else None
        doc = meta.get("doc") if meta and isinstance(meta.get("doc"), dict) else None
        t = doc.get("ticker") if doc and isinstance(doc.get("ticker"), str) else None
        if t and t.strip():
            out.append(t.strip().upper())
    return out


def _mentions_token(text: str, token: str) -> bool:
    if not token or not token.strip():
        return False
    pat = re.compile(rf"(?i)\\b{re.escape(token.strip())}\\b")
    return bool(pat.search(text or ""))


def score_one(
    query: EvalQuery,
    gen: EvalGeneration | None,
    *,
    judge_llm: LLMClient | None,
    judge_specs: list[JudgeSpec] | None = None,
    judge_context_chars: int = 14_000,
) -> EvalScore:
    judge_specs = list(judge_specs or [])

    score = EvalScore(query_id=query.id, kind=query.kind, created_at=_utcnow())

    if gen is None:
        score.answer["status"] = "missing_generation"
        return score

    if gen.error:
        score.answer["status"] = "generation_error"
        score.answer["error"] = gen.error
        return score

    final = (gen.final_answer or "").strip()
    top_chunks = list(gen.top_chunks or [])
    retrieved_chunk_ids = [c.chunk_id for c in top_chunks]
    retrieved_doc_ids = [c.doc_id for c in top_chunks]
    retrieved_tickers = _chunk_tickers(top_chunks)

    score.retrieval["retrieved_chunks"] = len(retrieved_chunk_ids)
    score.retrieval["retrieved_docs_unique"] = len(set(retrieved_doc_ids))
    if retrieved_tickers:
        score.retrieval["retrieved_tickers_unique"] = len(set(retrieved_tickers))
        score.retrieval["retrieved_tickers_top"] = retrieved_tickers[: min(12, len(retrieved_tickers))]

    if query.kind == "factual" and query.factual is not None:
        gold_chunk = query.factual.golden_evidence.chunk_id
        gold_doc = query.factual.golden_evidence.doc_id

        chunk_rank = _rank(retrieved_chunk_ids, gold_chunk)
        doc_rank = _rank(retrieved_doc_ids, gold_doc)

        score.retrieval.update(
            {
                "gold_chunk_id": gold_chunk,
                "gold_doc_id": gold_doc,
                "gold_chunk_rank": chunk_rank,
                "gold_doc_rank": doc_rank,
                "gold_chunk_mrr": (1.0 / chunk_rank) if chunk_rank else 0.0,
                "gold_doc_mrr": (1.0 / doc_rank) if doc_rank else 0.0,
            }
        )

        expected = query.factual.expected_numeric
        nm = best_numeric_match(final, expected.value, expected_scale=expected.scale)
        score.answer["numeric_matched"] = bool(nm["matched"])
        score.answer["numeric_best_rel_error"] = (
            float(nm["best_rel_error"]) if nm["best_rel_error"] is not None else None
        )
        score.answer["numeric_best_pred"] = nm["best_pred"]

        cited = cited_doc_ids(final)
        score.answer["cited_doc_ids"] = sorted(cited)
        score.answer["cited_gold_doc"] = bool(gold_doc in cited) if gold_doc else False

        if judge_llm is not None:
            ctx = build_context(top_chunks, max_chars=judge_context_chars)
            expected_s = _expected_numeric_text(query)
            evidence_s = query.factual.golden_evidence.snippet

            for spec in judge_specs or [FACTUAL_CORRECTNESS_V1]:
                out, raw = run_judge(
                    judge_llm,
                    spec,
                    question=query.question,
                    answer=final,
                    context=ctx,
                    expected=expected_s,
                    evidence=evidence_s,
                )
                score.judges.append(
                    JudgeResult(
                        judge_id=spec.judge_id,
                        prediction=out.prediction,
                        explanation=out.explanation_sketchpad,
                        raw=raw,
                    )
                )

    # Open-ended-style kinds: judge-based scoring only (plus a few lightweight heuristics).
    if query.kind == "open_ended" and query.open_ended is not None:
        if judge_llm is not None:
            ctx = build_context(top_chunks, max_chars=judge_context_chars)
            spec = get_judge_spec(query.open_ended.rubric_id or "faithfulness_v1")
            for js in judge_specs or [spec]:
                out, raw = run_judge(judge_llm, js, question=query.question, answer=final, context=ctx)
                score.judges.append(
                    JudgeResult(
                        judge_id=js.judge_id, prediction=out.prediction, explanation=out.explanation_sketchpad, raw=raw
                    )
                )

    if query.kind == "refusal" and query.refusal is not None:
        score.answer["refused_heuristic"] = _looks_like_refusal(final)
        if judge_llm is not None:
            ctx = build_context(top_chunks, max_chars=judge_context_chars)
            spec = get_judge_spec(query.refusal.rubric_id or "refusal_v1")
            notes = f"reason={query.refusal.reason}"
            if query.refusal.target_ticker:
                notes += f", target_ticker={query.refusal.target_ticker}"
            if query.refusal.target_company:
                notes += f", target_company={query.refusal.target_company}"
            for js in judge_specs or [spec]:
                out, raw = run_judge(judge_llm, js, question=query.question, answer=final, context=ctx, notes=notes)
                score.judges.append(
                    JudgeResult(
                        judge_id=js.judge_id, prediction=out.prediction, explanation=out.explanation_sketchpad, raw=raw
                    )
                )

    if query.kind == "distractor" and query.distractor is not None:
        if query.distractor.target_tickers:
            score.answer["mentions_target_ticker"] = any(
                _mentions_token(final, t) for t in query.distractor.target_tickers
            )
        if judge_llm is not None:
            ctx = build_context(top_chunks, max_chars=judge_context_chars)
            spec = get_judge_spec(query.distractor.rubric_id or "focus_v1")
            notes = (
                f"main_question={query.distractor.main_question.strip()}\n"
                f"distractor_kind={query.distractor.distractor_kind}\n"
                f"distractor_text={query.distractor.distractor_text.strip()}"
            )
            for js in judge_specs or [spec]:
                out, raw = run_judge(judge_llm, js, question=query.question, answer=final, context=ctx, notes=notes)
                score.judges.append(
                    JudgeResult(
                        judge_id=js.judge_id, prediction=out.prediction, explanation=out.explanation_sketchpad, raw=raw
                    )
                )

    if query.kind == "comparison" and query.comparison is not None:
        targets = [t.strip().upper() for t in (query.comparison.target_tickers or []) if t and t.strip()]
        if targets:
            score.retrieval["comparison_target_tickers"] = targets
            score.retrieval["comparison_retrieved_tickers_unique"] = sorted(set(retrieved_tickers))
            score.retrieval["comparison_all_targets_retrieved"] = all(t in set(retrieved_tickers) for t in targets)
            score.answer["mentions_all_target_tickers"] = all(_mentions_token(final, t) for t in targets)

        if judge_llm is not None:
            ctx = build_context(top_chunks, max_chars=judge_context_chars)
            spec = get_judge_spec(query.comparison.rubric_id or "comparison_v1")
            notes = f"target_tickers={targets}"
            for js in judge_specs or [spec]:
                out, raw = run_judge(judge_llm, js, question=query.question, answer=final, context=ctx, notes=notes)
                score.judges.append(
                    JudgeResult(
                        judge_id=js.judge_id, prediction=out.prediction, explanation=out.explanation_sketchpad, raw=raw
                    )
                )

    return score


def summarize(scores: list[EvalScore]) -> dict[str, Any]:
    """
    Small, copy-paste-friendly summary dict.
    """
    out: dict[str, Any] = {"n": len(scores)}
    if not scores:
        return out

    def _mean(vals: list[float]) -> float:
        vals = [v for v in vals if v is not None and not math.isnan(v)]
        return (sum(vals) / len(vals)) if vals else math.nan

    def _is_ok(s: EvalScore) -> bool:
        return not bool(s.answer.get("status"))

    factual = [s for s in scores if s.kind == "factual"]
    open_ended = [s for s in scores if s.kind == "open_ended"]
    refusal = [s for s in scores if s.kind == "refusal"]
    distractor = [s for s in scores if s.kind == "distractor"]
    comparison = [s for s in scores if s.kind == "comparison"]
    factual_ok = [s for s in factual if _is_ok(s)]
    open_ended_ok = [s for s in open_ended if _is_ok(s)]
    refusal_ok = [s for s in refusal if _is_ok(s)]
    distractor_ok = [s for s in distractor if _is_ok(s)]
    comparison_ok = [s for s in comparison if _is_ok(s)]

    if factual:
        out["factual_n"] = len(factual)
        out["factual_n_ok"] = len(factual_ok)
        out["factual_gold_chunk_hit_rate"] = _mean(
            [1.0 if s.retrieval.get("gold_chunk_rank") else 0.0 for s in factual_ok]
        )
        out["factual_numeric_accuracy"] = _mean([1.0 if s.answer.get("numeric_matched") else 0.0 for s in factual_ok])

        # Primary judge fail-rate (prediction=1).
        judge_preds = []
        for s in factual_ok:
            if s.judges:
                judge_preds.append(float(s.judges[0].prediction))
        if judge_preds:
            out["factual_judge_fail_rate"] = _mean(judge_preds)

    if open_ended:
        out["open_ended_n"] = len(open_ended)
        out["open_ended_n_ok"] = len(open_ended_ok)
        judge_preds = []
        for s in open_ended_ok:
            if s.judges:
                judge_preds.append(float(s.judges[0].prediction))
        if judge_preds:
            out["open_ended_judge_fail_rate"] = _mean(judge_preds)

    if refusal:
        out["refusal_n"] = len(refusal)
        out["refusal_n_ok"] = len(refusal_ok)
        judge_preds = []
        for s in refusal_ok:
            if s.judges:
                judge_preds.append(float(s.judges[0].prediction))
        if judge_preds:
            out["refusal_judge_fail_rate"] = _mean(judge_preds)

    if distractor:
        out["distractor_n"] = len(distractor)
        out["distractor_n_ok"] = len(distractor_ok)
        judge_preds = []
        for s in distractor_ok:
            if s.judges:
                judge_preds.append(float(s.judges[0].prediction))
        if judge_preds:
            out["distractor_judge_fail_rate"] = _mean(judge_preds)

    if comparison:
        out["comparison_n"] = len(comparison)
        out["comparison_n_ok"] = len(comparison_ok)
        judge_preds = []
        for s in comparison_ok:
            if s.judges:
                judge_preds.append(float(s.judges[0].prediction))
        if judge_preds:
            out["comparison_judge_fail_rate"] = _mean(judge_preds)

    return out

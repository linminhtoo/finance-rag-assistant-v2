from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from finrag.dataclasses import ScoredChunk
from finrag.eval.metrics import best_numeric_match, cited_doc_ids, coverage_at_k, keyword_coverage, mrr, recall_at_k
from finrag.eval.schema import EvalItem
from finrag.llm_clients import LLMClient
from finrag.qa import answer_question_two_stage
from finrag.retriever import CrossEncoderReranker, QdrantHybridRetriever, MilvusContextualRetriever, NoopReranker


@dataclass(frozen=True)
class EvalConfig:
    top_k_retrieve: int = 30
    top_k_rerank: int = 8
    do_answer: bool = False
    draft_max_tokens: int = 900
    final_max_tokens: int = 1500


def _answer_question(llm: LLMClient, question: str, reranked: list[ScoredChunk], cfg: EvalConfig) -> tuple[str, str]:
    return answer_question_two_stage(
        llm,
        question,
        reranked,
        draft_max_tokens=cfg.draft_max_tokens,
        final_max_tokens=cfg.final_max_tokens,
        temperature_draft=0.1,
    )


def run_eval(
    items: list[EvalItem],
    *,
    retriever: QdrantHybridRetriever | MilvusContextualRetriever,
    reranker: CrossEncoderReranker | NoopReranker,
    cfg: EvalConfig,
    llm_for_answer: LLMClient | None = None,
) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for item in items:
        relevant_chunk_ids = {e.chunk_id for e in item.evidences if e.chunk_id}
        relevant_doc_ids = {e.doc_id for e in item.evidences}

        t0 = time.perf_counter()
        hybrid = retriever.retrieve_hybrid(
            item.question,
            top_k_semantic=cfg.top_k_retrieve,
            top_k_bm25=cfg.top_k_retrieve,
            top_k_final=cfg.top_k_retrieve,
        )
        t1 = time.perf_counter()
        reranked = reranker.rerank(item.question, hybrid, top_k=cfg.top_k_rerank)
        t2 = time.perf_counter()

        hybrid_chunk_ids = [sc.chunk.id for sc in hybrid]
        rerank_chunk_ids = [sc.chunk.id for sc in reranked]
        hybrid_doc_ids = [sc.chunk.doc_id for sc in hybrid]
        rerank_doc_ids = [sc.chunk.doc_id for sc in reranked]

        res: dict[str, Any] = {
            "id": item.id,
            "question": item.question,
            "kind": item.kind,
            "tags": item.tags,
            "evidence_chunk_ids": sorted(relevant_chunk_ids),
            "evidence_doc_ids": sorted(relevant_doc_ids),
            "hybrid_chunk_ids": hybrid_chunk_ids,
            "rerank_chunk_ids": rerank_chunk_ids,
            "hybrid_doc_ids": hybrid_doc_ids,
            "rerank_doc_ids": rerank_doc_ids,
            "recall_hybrid_chunk": recall_at_k(hybrid_chunk_ids, set(relevant_chunk_ids), cfg.top_k_retrieve),
            "recall_rerank_chunk": recall_at_k(rerank_chunk_ids, set(relevant_chunk_ids), cfg.top_k_rerank),
            "recall_hybrid_doc": recall_at_k(hybrid_doc_ids, relevant_doc_ids, cfg.top_k_retrieve),
            "recall_rerank_doc": recall_at_k(rerank_doc_ids, relevant_doc_ids, cfg.top_k_rerank),
            "coverage_hybrid_chunk": coverage_at_k(hybrid_chunk_ids, set(relevant_chunk_ids), cfg.top_k_retrieve),
            "coverage_rerank_chunk": coverage_at_k(rerank_chunk_ids, set(relevant_chunk_ids), cfg.top_k_rerank),
            "coverage_hybrid_doc": coverage_at_k(hybrid_doc_ids, relevant_doc_ids, cfg.top_k_retrieve),
            "coverage_rerank_doc": coverage_at_k(rerank_doc_ids, relevant_doc_ids, cfg.top_k_rerank),
            "mrr_hybrid_chunk": mrr(hybrid_chunk_ids, set(relevant_chunk_ids)),
            "mrr_rerank_chunk": mrr(rerank_chunk_ids, set(relevant_chunk_ids)),
            "timing_ms": {"retrieve_ms": (t1 - t0) * 1000.0, "rerank_ms": (t2 - t1) * 1000.0},
        }

        if cfg.do_answer:
            if llm_for_answer is None:
                raise RuntimeError("cfg.do_answer=True requires llm_for_answer")
            t3 = time.perf_counter()
            draft, final = _answer_question(llm_for_answer, item.question, reranked, cfg)
            t4 = time.perf_counter()
            res["draft_answer"] = draft
            res["final_answer"] = final
            res["timing_ms"]["answer_ms"] = (t4 - t3) * 1000.0

            res["citation_doc_ids"] = sorted(cited_doc_ids(final))
            res["citation_hit"] = 1.0 if cited_doc_ids(final) & relevant_doc_ids else 0.0

            if item.expected_numeric and item.expected_numeric.value is not None:
                nm = best_numeric_match(final, item.expected_numeric.value, expected_scale=item.expected_numeric.scale)
                res["numeric_matched"] = nm["matched"]
                res["numeric_best_rel_error"] = nm["best_rel_error"]
                res["numeric_best_pred"] = nm["best_pred"]

            if item.expected_key_points:
                res["qual_keyword_coverage"] = keyword_coverage(final, item.expected_key_points)

        results.append(res)

    summary = summarize_results(results)
    return {"summary": summary, "results": results}


def _mean_ignore_nan(vals: list[float]) -> float:
    cleaned = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
    if not cleaned:
        return math.nan
    return sum(cleaned) / len(cleaned)


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    recall_hybrid_chunk = _mean_ignore_nan([r["recall_hybrid_chunk"] for r in results])
    recall_rerank_chunk = _mean_ignore_nan([r["recall_rerank_chunk"] for r in results])
    recall_hybrid_doc = _mean_ignore_nan([r["recall_hybrid_doc"] for r in results])
    recall_rerank_doc = _mean_ignore_nan([r["recall_rerank_doc"] for r in results])
    mrr_hybrid_chunk = _mean_ignore_nan([r["mrr_hybrid_chunk"] for r in results])
    mrr_rerank_chunk = _mean_ignore_nan([r["mrr_rerank_chunk"] for r in results])
    coverage_hybrid_doc = _mean_ignore_nan([r["coverage_hybrid_doc"] for r in results])
    coverage_rerank_doc = _mean_ignore_nan([r["coverage_rerank_doc"] for r in results])

    summary: dict[str, Any] = {
        "n": len(results),
        "recall_hybrid_chunk": recall_hybrid_chunk,
        "recall_rerank_chunk": recall_rerank_chunk,
        "recall_hybrid_doc": recall_hybrid_doc,
        "recall_rerank_doc": recall_rerank_doc,
        "coverage_hybrid_doc": coverage_hybrid_doc,
        "coverage_rerank_doc": coverage_rerank_doc,
        "mrr_hybrid_chunk": mrr_hybrid_chunk,
        "mrr_rerank_chunk": mrr_rerank_chunk,
    }

    if any("numeric_matched" in r for r in results):
        matched = [1.0 if r.get("numeric_matched") else 0.0 for r in results if "numeric_matched" in r]
        summary["numeric_accuracy"] = _mean_ignore_nan(matched)

    if any("citation_hit" in r for r in results):
        hits = [r.get("citation_hit", 0.0) for r in results if "citation_hit" in r]
        summary["citation_hit_rate"] = _mean_ignore_nan(hits)

    if any("qual_keyword_coverage" in r for r in results):
        covs = [r.get("qual_keyword_coverage") for r in results if "qual_keyword_coverage" in r]
        summary["qual_keyword_coverage"] = _mean_ignore_nan([c for c in covs if c is not None])

    return summary


def save_run(run: dict[str, Any], out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(run, ensure_ascii=False, indent=2), encoding="utf-8")

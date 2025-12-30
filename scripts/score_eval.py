#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from finrag.eval.io import dump_jsonl, load_jsonl
from finrag.eval.judges import get_judge_client
from finrag.eval.schema import EvalGeneration, EvalQuery
from finrag.eval.scoring import score_one, summarize
from finrag.eval.runner import save_json

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _compact_top_chunks(gen: EvalGeneration, *, max_chars: int = 2400, max_chunks: int = 6) -> str:
    parts: list[str] = []
    used = 0
    for ch in (gen.top_chunks or [])[:max_chunks]:
        head = f"[doc={ch.doc_id} chunk={ch.chunk_id} score={ch.score:.4f}]"
        body = (ch.preview or ch.text or "").strip().replace("\n", " ")
        body = " ".join(body.split())
        block = head + " " + body
        if used + len(block) > max_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n".join(parts).strip()


def _write_review_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> None:
    ap = argparse.ArgumentParser(description="Score a run directory produced by scripts/run_eval.py.")
    ap.add_argument("--run-dir", required=True, help="Run directory containing eval_queries.jsonl + generations.jsonl.")
    ap.add_argument("--no-judge", action="store_true", help="Skip LLM-as-judge scoring.")
    ap.add_argument("--judge-provider", default=None)
    ap.add_argument("--judge-model", default=None)
    ap.add_argument("--judge-base-url", default=None)
    ap.add_argument("--judge-context-chars", type=int, default=14_000)
    ap.add_argument("--max-items", type=int, default=None)
    ap.add_argument(
        "--kinds",
        nargs="*",
        default=None,
        choices=["factual", "open_ended", "refusal", "distractor", "comparison"],
        help="Optional filter (defaults to all).",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    eval_queries_path = run_dir / "eval_queries.jsonl"
    generations_path = run_dir / "generations.jsonl"
    if not eval_queries_path.exists():
        raise SystemExit(f"Missing: {eval_queries_path}")
    if not generations_path.exists():
        raise SystemExit(f"Missing: {generations_path}")

    queries = load_jsonl(eval_queries_path, EvalQuery)
    generations = load_jsonl(generations_path, EvalGeneration)

    gens_by_id = {g.query_id: g for g in generations}

    if args.kinds:
        wanted = set(args.kinds)
        queries = [q for q in queries if q.kind in wanted]
    if args.max_items is not None:
        queries = queries[: max(0, int(args.max_items))]
    if not queries:
        raise SystemExit("No items to score (check --kinds/--max-items).")

    judge_llm = None if args.no_judge else get_judge_client(
        provider=args.judge_provider, chat_model=args.judge_model, base_url=args.judge_base_url
    )

    scores = [
        score_one(q, gens_by_id.get(q.id), judge_llm=judge_llm, judge_context_chars=args.judge_context_chars)
        for q in queries
    ]
    summary = summarize(scores)

    dump_jsonl(scores, run_dir / "scores.jsonl")
    save_json(summary, run_dir / "score_summary.json")

    # A merged, single-record-per-case JSONL is easiest to grep through.
    cases: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    for q, s in zip(queries, scores):
        g = gens_by_id.get(q.id)
        case = {
            "query": q.model_dump(mode="json"),
            "generation": (g.model_dump(mode="json") if g is not None else None),
            "score": s.model_dump(mode="json"),
        }
        cases.append(case)

        expected = q.factual.expected_numeric if q.factual is not None else None
        gold = q.factual.golden_evidence if q.factual is not None else None
        judge0 = s.judges[0] if s.judges else None
        target_tickers: list[str] = []
        if q.open_ended is not None and q.open_ended.target_ticker:
            target_tickers.append(q.open_ended.target_ticker)
        if q.distractor is not None:
            target_tickers.extend(list(q.distractor.target_tickers or []))
        if q.comparison is not None:
            target_tickers.extend(list(q.comparison.target_tickers or []))
        if q.refusal is not None and q.refusal.target_ticker:
            target_tickers.append(q.refusal.target_ticker)
        target_tickers_s = " ".join(sorted({t.strip().upper() for t in target_tickers if t and t.strip()}))
        review_rows.append(
            {
                "query_id": q.id,
                "kind": q.kind,
                "question": q.question,
                "tags": " ".join([t for t in (q.tags or []) if t]),
                "target_tickers": target_tickers_s,
                "refusal_reason": (q.refusal.reason if q.refusal is not None else ""),
                "distractor_kind": (q.distractor.distractor_kind if q.distractor is not None else ""),
                "expected_value": (expected.value if expected is not None else ""),
                "expected_scale": (expected.scale if expected is not None else ""),
                "expected_raw": (expected.raw if expected is not None and expected.raw else ""),
                "gold_doc_id": (gold.doc_id if gold is not None else ""),
                "gold_chunk_id": (gold.chunk_id if gold is not None else ""),
                "gold_section_path": (gold.section_path if gold is not None and gold.section_path else ""),
                "gold_chunk_rank": s.retrieval.get("gold_chunk_rank", ""),
                "numeric_matched": s.answer.get("numeric_matched", ""),
                "numeric_best_pred": s.answer.get("numeric_best_pred", ""),
                "numeric_best_rel_error": s.answer.get("numeric_best_rel_error", ""),
                "cited_gold_doc": s.answer.get("cited_gold_doc", ""),
                "judge_id": (judge0.judge_id if judge0 is not None else ""),
                "judge_prediction": (judge0.prediction if judge0 is not None else ""),
                "judge_explanation": (judge0.explanation if judge0 is not None and judge0.explanation else ""),
                "final_answer": (g.final_answer if g is not None and g.final_answer else ""),
                "top_chunks_compact": (_compact_top_chunks(g) if g is not None else ""),
                "human_label": "",
                "human_notes": "",
            }
        )

    dump_jsonl(cases, run_dir / "cases.jsonl")
    _write_review_csv(review_rows, run_dir / "review.csv")

    print(f"Wrote: {run_dir / 'scores.jsonl'}")
    print(f"Wrote: {run_dir / 'cases.jsonl'}")
    print(f"Wrote: {run_dir / 'review.csv'}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

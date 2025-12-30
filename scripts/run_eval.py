#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from finrag.eval.io import load_jsonl
from finrag.eval.runner import RunConfig, run_generation, save_json
from finrag.eval.schema import EvalQuery

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an eval query set through finrag.main.RAGService.answer_question().")
    ap.add_argument("--eval-queries", required=True, help="Eval queries JSONL (from scripts/make_eval_set.py).")
    ap.add_argument("--out-dir", required=True, help="Directory to write run artifacts.")
    ap.add_argument("--run-name", default=None, help="Optional run name prefix (e.g. 'baseline').")
    ap.add_argument("--mode", default="normal", help="Generation preset (quick|normal|thinking).")
    ap.add_argument(
        "--concurrency",
        type=int,
        default=8,
        help="Max parallel questions to run (process-based; set 1 to disable).",
    )

    # Convenience: point the runner at an existing Milvus Lite + BM25 snapshot dir
    # (the same dir you pass as `--ingest-output-dir` to scripts/build_index.py).
    ap.add_argument(
        "--index-dir",
        default=None,
        help="Directory containing `milvus.db`, `bm25.pkl`, and `doc_index.jsonl` (sets env vars for this run).",
    )
    ap.add_argument("--milvus-path", default=None, help="Overrides env MILVUS_PATH for this run.")
    ap.add_argument("--bm25-path", default=None, help="Overrides env BM25_PATH for this run.")
    ap.add_argument("--doc-index-path", default=None, help="Overrides env FINRAG_DOC_INDEX_PATH for this run.")

    # Optional overrides.
    ap.add_argument("--top-k-retrieve", type=int, default=None)
    ap.add_argument("--top-k-rerank", type=int, default=None)
    ap.add_argument("--draft-max-tokens", type=int, default=None)
    ap.add_argument("--final-max-tokens", type=int, default=None)
    ap.add_argument("--enable-rerank", type=int, default=None, help="1/0 override (defaults to preset).")
    ap.add_argument("--enable-refine", type=int, default=None, help="1/0 override (defaults to preset).")
    ap.add_argument("--draft-temperature", type=float, default=None)

    # Output controls.
    ap.add_argument("--max-chunks", type=int, default=50)
    ap.add_argument("--chunk-text-chars", type=int, default=2000)
    ap.add_argument("--chunk-context-chars", type=int, default=2000)

    # Filters.
    ap.add_argument("--max-items", type=int, default=None, help="Optional cap on number of queries to run.")
    ap.add_argument(
        "--kinds",
        nargs="*",
        default=None,
        choices=["factual", "open_ended", "refusal", "distractor", "comparison"],
        help="Optional filter (defaults to all).",
    )

    args = ap.parse_args()

    if args.index_dir:
        idx = Path(args.index_dir).expanduser().resolve()
        if not idx.exists():
            raise SystemExit(f"--index-dir does not exist: {idx}")
        if args.milvus_path is None:
            os.environ["MILVUS_PATH"] = str((idx / "milvus.db").resolve())
        if args.bm25_path is None:
            os.environ["BM25_PATH"] = str((idx / "bm25.pkl").resolve())
        if args.doc_index_path is None:
            os.environ["FINRAG_DOC_INDEX_PATH"] = str((idx / "doc_index.jsonl").resolve())
    if args.milvus_path:
        os.environ["MILVUS_PATH"] = str(Path(args.milvus_path).expanduser().resolve())
    if args.bm25_path:
        os.environ["BM25_PATH"] = str(Path(args.bm25_path).expanduser().resolve())
    if args.doc_index_path:
        os.environ["FINRAG_DOC_INDEX_PATH"] = str(Path(args.doc_index_path).expanduser().resolve())

    queries = load_jsonl(args.eval_queries, EvalQuery)
    if args.kinds:
        wanted = set(args.kinds)
        queries = [q for q in queries if q.kind in wanted]
    if args.max_items is not None:
        queries = queries[: max(0, int(args.max_items))]
    if not queries:
        raise SystemExit("No eval queries to run (check --kinds/--max-items).")

    out_root = Path(args.out_dir).expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    stamp = _timestamp()
    run_name = (args.run_name.strip() + ".") if isinstance(args.run_name, str) and args.run_name.strip() else ""
    run_dir = out_root / f"eval_run.{run_name}{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Persist the exact query set used for this run.
    shutil.copyfile(args.eval_queries, run_dir / "eval_queries.jsonl")

    cfg = RunConfig(
        mode=args.mode,
        top_k_retrieve=args.top_k_retrieve,
        top_k_rerank=args.top_k_rerank,
        draft_max_tokens=args.draft_max_tokens,
        final_max_tokens=args.final_max_tokens,
        enable_rerank=(bool(args.enable_rerank) if args.enable_rerank is not None else None),
        enable_refine=(bool(args.enable_refine) if args.enable_refine is not None else None),
        draft_temperature=args.draft_temperature,
        concurrency=args.concurrency,
        max_chunks=args.max_chunks,
        chunk_text_chars=args.chunk_text_chars,
        chunk_context_chars=args.chunk_context_chars,
    )

    summary = run_generation(queries, out_jsonl=run_dir / "generations.jsonl", cfg=cfg)
    save_json(cfg.to_dict(), run_dir / "run_config.json")
    save_json(summary, run_dir / "generation_summary.json")

    print(f"Wrote run dir: {run_dir}")
    print(f"Summary: {summary}")


if __name__ == "__main__":
    main()

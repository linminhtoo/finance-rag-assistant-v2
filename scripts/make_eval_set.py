#!/usr/bin/env python3
import argparse
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from finrag.eval.generation import (
    generate_comparison_queries,
    generate_distractor_queries,
    generate_factual_queries,
    generate_open_ended_queries,
    generate_refusal_queries,
)
from finrag.eval.io import dump_jsonl
from finrag.eval.sec_corpus import iter_all_chunks, iter_chunk_export_docs
from finrag.utils import seed_everything

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate an eval query set from chunk exports (JSONL).")
    ap.add_argument(
        "--ingest-output-dir",
        required=True,
        help="Chunk export dir produced by `scripts/chunk.py` (must contain doc_index.jsonl and chunks/).",
    )
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--tickers", nargs="*", default=None, help="Optional tickers filter (e.g. AAPL MSFT)")
    ap.add_argument("--forms", nargs="*", default=None, help="Optional forms filter (e.g. 10-Q 10-K)")
    ap.add_argument("--max-docs", type=int, default=200, help="Max documents to scan")
    ap.add_argument("--max-chunks-per-doc", type=int, default=120, help="Max chunks to scan per document")
    ap.add_argument("--n-factual", type=int, default=40, help="Number of factual questions")
    ap.add_argument("--n-open-ended", type=int, default=40, help="Number of open-ended questions")
    ap.add_argument("--n-refusal", type=int, default=20, help="Number of refusal / out-of-scope questions")
    ap.add_argument("--n-distractor", type=int, default=20, help="Number of distractor questions")
    ap.add_argument("--n-comparison", type=int, default=20, help="Number of multi-company comparison questions")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument(
        "--snippet-chars",
        type=int,
        default=5000,
        help="Max characters to store from the golden chunk as evidence snippet (factual questions).",
    )

    args = ap.parse_args()

    seed_everything(args.seed)

    chunks_iter = iter_all_chunks(
        args.ingest_output_dir,
        tickers=args.tickers,
        forms=args.forms,
        max_docs=args.max_docs,
        max_chunks_per_doc=args.max_chunks_per_doc,
    )
    factual = generate_factual_queries(chunks_iter, n=args.n_factual, seed=args.seed, snippet_chars=args.snippet_chars)
    logger.success(f"Generated {len(factual)} factual questions.")

    docs_iter = list(
        iter_chunk_export_docs(args.ingest_output_dir, tickers=args.tickers, forms=args.forms, max_docs=args.max_docs)
    )
    doc_dicts = [
        {"ticker": d.ticker, "year": d.year, "company": d.company}
        for d in docs_iter
        if d.ticker is not None and d.year is not None
    ]
    logger.debug(f"Prepared {len(doc_dicts)} document dicts for query generation.")

    open_ended = generate_open_ended_queries(doc_dicts, n=args.n_open_ended, seed=args.seed)
    logger.success(f"Generated {len(open_ended)} open-ended questions.")

    refusal = generate_refusal_queries(doc_dicts, n=args.n_refusal, seed=args.seed)
    logger.success(f"Generated {len(refusal)} refusal questions.")

    distractor = generate_distractor_queries(doc_dicts, n=args.n_distractor, seed=args.seed)
    logger.success(f"Generated {len(distractor)} distractor questions.")

    comparison = generate_comparison_queries(doc_dicts, n=args.n_comparison, seed=args.seed)
    logger.success(f"Generated {len(comparison)} comparison questions.")

    items = [*factual, *open_ended, *refusal, *distractor, *comparison]

    out_path = Path(args.out)
    dump_jsonl(items, out_path)
    logger.success(f"Wrote {len(items)} total eval queries to {out_path}")


if __name__ == "__main__":
    main()

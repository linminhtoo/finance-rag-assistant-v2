#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from finrag.eval.io import load_jsonl
from finrag.eval.report import write_html_report
from finrag.eval.runner import EvalConfig, run_eval, save_run
from finrag.eval.sec_corpus import load_sec_download_dir
from finrag.llm_clients import get_llm_client
from finrag.retriever import CrossEncoderReranker, HybridRetriever, NoopReranker
from finrag.sec_chunking import SecHtmlChunker

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an end-to-end eval over an SEC-filings corpus.")
    ap.add_argument("--data-dir", required=True, help="Directory containing `10k_raw/` and `meta/`")
    ap.add_argument("--eval-set", required=True, help="Eval set JSONL created by scripts/make_eval_set.py")
    ap.add_argument("--out-dir", required=True, help="Output directory for run JSON + HTML report")
    ap.add_argument("--tickers", nargs="*", default=None, help="Optional tickers filter for indexing")
    ap.add_argument("--forms", nargs="*", default=None, help="Optional forms filter for indexing")
    ap.add_argument("--max-docs", type=int, default=200, help="Max documents to index")
    ap.add_argument("--storage-path", default=":memory:", help="Qdrant storage path (':memory:' for ephemeral)")
    ap.add_argument("--collection-name", default="rag_chunks", help="Qdrant collection name")
    ap.add_argument("--max-words", type=int, default=350, help="Chunk max words")
    ap.add_argument("--overlap-words", type=int, default=50, help="Chunk overlap words")
    ap.add_argument("--top-k-retrieve", type=int, default=30)
    ap.add_argument("--top-k-rerank", type=int, default=8)
    ap.add_argument("--do-answer", action="store_true", help="Run LLM answer generation + answer scoring")
    ap.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker (uses hybrid scores)")
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--llm-provider", default=None, help="Overrides LLM_PROVIDER (e.g. mistral, openai, fastembed)")
    ap.add_argument("--llm-kwargs", default=None, help="JSON dict passed to get_llm_client(...) (advanced)")

    args = ap.parse_args()

    items = load_jsonl(args.eval_set)
    if not items:
        raise SystemExit("Empty eval set")

    docs = load_sec_download_dir(args.data_dir, tickers=args.tickers, forms=args.forms, max_docs=args.max_docs)
    if not docs:
        raise SystemExit("No documents found to index. Check --data-dir and filters.")

    llm_kwargs = {}
    if args.llm_kwargs:
        import json

        llm_kwargs = json.loads(args.llm_kwargs)
        if not isinstance(llm_kwargs, dict):
            raise ValueError("--llm-kwargs must be a JSON object")

    llm = get_llm_client(provider=args.llm_provider, **llm_kwargs)
    retriever = HybridRetriever(
        llm_client=llm, storage_path=args.storage_path, collection_name=args.collection_name, vector_dim=None
    )

    reranker = NoopReranker() if args.no_reranker else CrossEncoderReranker(model_name=args.reranker_model)

    # TODO: not sure if this HtmlChunker is performant... we probably should use Mistral OCR...?
    # and also avoid re-chunking docs that had been ingested by the retriever.
    chunker = SecHtmlChunker(max_words=args.max_words, overlap_words=args.overlap_words)
    for doc in docs:
        chunks = chunker.chunk_html_file(doc.source_path, doc_id=doc.doc_id, metadata=doc.meta)
        retriever.index(chunks)

    cfg = EvalConfig(top_k_retrieve=args.top_k_retrieve, top_k_rerank=args.top_k_rerank, do_answer=args.do_answer)
    run = run_eval(
        items, retriever=retriever, reranker=reranker, cfg=cfg, llm_for_answer=llm if args.do_answer else None
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = out_dir / f"eval_run.{stamp}.json"
    out_html = out_dir / f"eval_run.{stamp}.html"

    save_run(run, out_json)
    write_html_report(run, out_html)

    print(f"Wrote run JSON: {out_json}")
    print(f"Wrote HTML report: {out_html}")
    print(f"Summary: {run['summary']}")


if __name__ == "__main__":
    main()

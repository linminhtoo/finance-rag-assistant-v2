#!/usr/bin/env python3
import argparse
import os
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from finrag.eval.io import load_jsonl
from finrag.eval.report import write_html_report
from finrag.eval.runner import EvalConfig, run_eval, save_run
from finrag.eval.sec_corpus import load_sec_download_dir
from finrag.llm_clients import get_llm_client
from finrag.retriever import CrossEncoderReranker, QdrantHybridRetriever, MilvusContextualRetriever, NoopReranker
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
    ap.add_argument(
        "--qdrant-storage-path",
        default=":memory:",
        help="Qdrant storage path (':memory:' for ephemeral; ignored for Milvus).",
    )
    ap.add_argument("--collection-name", default="rag_chunks", help="Collection name.")
    ap.add_argument(
        "--retriever-backend",
        default="qdrant",
        choices=["qdrant", "milvus"],
        help="Backend to store vectors (default: qdrant).",
    )
    ap.add_argument("--milvus-uri", default=None, help="Milvus URI/path (defaults to env MILVUS_URI).")
    ap.add_argument(
        "--milvus-sparse",
        default="bm25",
        choices=["bm25", "none"],
        help="Sparse embedding strategy for Milvus (default: bm25).",
    )
    ap.add_argument("--milvus-bm25-path", default=None, help="Optional path to load/save BM25 parameters.")
    ap.add_argument("--max-words", type=int, default=350, help="Chunk max words")
    ap.add_argument("--overlap-words", type=int, default=50, help="Chunk overlap words")
    ap.add_argument("--top-k-retrieve", type=int, default=30)
    ap.add_argument("--top-k-rerank", type=int, default=8)
    ap.add_argument("--do-answer", action="store_true", help="Run LLM answer generation + answer scoring")
    ap.add_argument("--no-reranker", action="store_true", help="Disable cross-encoder reranker (uses hybrid scores)")
    ap.add_argument("--reranker-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    ap.add_argument("--llm-provider", default=None, help="Overrides LLM_PROVIDER (e.g. mistral, openai, fastembed)")
    ap.add_argument("--llm-kwargs", default=None, help="JSON dict passed to get_llm_client(...) (advanced)")
    ap.add_argument(
        "--answer-llm-provider", default=None, help="Provider for answer generation (defaults to --llm-provider)"
    )
    ap.add_argument(
        "--answer-llm-kwargs",
        default=None,
        help="JSON dict passed to get_llm_client(...) for answer generation (advanced; defaults to --llm-kwargs).",
    )

    args = ap.parse_args()

    items = load_jsonl(args.eval_set)
    if not items:
        raise SystemExit("Empty eval set")

    docs = load_sec_download_dir(args.data_dir, tickers=args.tickers, forms=args.forms, max_docs=args.max_docs)
    if not docs:
        raise SystemExit("No documents found to index. Check --data-dir and filters.")

    llm_kwargs: dict = {}
    if args.llm_kwargs:
        import json

        llm_kwargs = json.loads(args.llm_kwargs)
        if not isinstance(llm_kwargs, dict):
            raise ValueError("--llm-kwargs must be a JSON object")

    llm_for_embeddings = get_llm_client(provider=args.llm_provider, **llm_kwargs)

    llm_for_answer = None
    if args.do_answer:
        answer_kwargs = llm_kwargs
        if args.answer_llm_kwargs:
            import json

            answer_kwargs = json.loads(args.answer_llm_kwargs)
            if not isinstance(answer_kwargs, dict):
                raise ValueError("--answer-llm-kwargs must be a JSON object")
        llm_for_answer = get_llm_client(provider=(args.answer_llm_provider or args.llm_provider), **answer_kwargs)

    if args.retriever_backend == "qdrant":
        retriever: QdrantHybridRetriever | MilvusContextualRetriever = QdrantHybridRetriever(
            llm_client=llm_for_embeddings,
            storage_path=args.qdrant_storage_path,
            collection_name=args.collection_name,
            vector_dim=None,
        )
        use_sparse = False
        bm25_loaded = False
    else:
        milvus_uri = args.milvus_uri or os.environ.get("MILVUS_URI") or "milvus_eval.db"
        if "://" not in milvus_uri:
            milvus_uri = os.path.expanduser(milvus_uri)
        use_sparse = args.milvus_sparse != "none"
        retriever = MilvusContextualRetriever(
            llm_client=llm_for_embeddings,
            uri=milvus_uri,
            collection_name=args.collection_name,
            use_sparse=use_sparse,
            bm25_path=args.milvus_bm25_path,
        )
        bm25_loaded = False
        if use_sparse and retriever.uses_bm25 and args.milvus_bm25_path:
            bm25_path = Path(args.milvus_bm25_path)
            if bm25_path.exists():
                retriever.load_bm25(str(bm25_path))
                bm25_loaded = True

    reranker = NoopReranker() if args.no_reranker else CrossEncoderReranker(model_name=args.reranker_model)

    # FIXME: sync to latest chunking pipeline. this is outdated.
    # and also avoid re-chunking docs that had been ingested by the retriever.
    chunker = SecHtmlChunker(max_words=args.max_words, overlap_words=args.overlap_words)
    pending_chunks: list = []
    needs_bm25_fit = args.retriever_backend == "milvus" and use_sparse and retriever.uses_bm25 and not bm25_loaded
    for doc in docs:
        chunks = chunker.chunk_html_file(doc.source_path, doc_id=doc.doc_id, metadata=doc.meta)
        if needs_bm25_fit:
            pending_chunks.extend(chunks)
        else:
            retriever.index(chunks)

    if needs_bm25_fit and pending_chunks:
        corpus = [(ch.metadata or {}).get("index_text") or ch.text for ch in pending_chunks]
        retriever.fit_bm25([str(t) for t in corpus])
        retriever.index(pending_chunks)
        # FIXME: milvus bm25 should fit once before chunking and saved immediately, not here.
        if args.milvus_bm25_path:
            retriever.save_bm25(args.milvus_bm25_path)

    cfg = EvalConfig(top_k_retrieve=args.top_k_retrieve, top_k_rerank=args.top_k_rerank, do_answer=args.do_answer)
    run = run_eval(items, retriever=retriever, reranker=reranker, cfg=cfg, llm_for_answer=llm_for_answer)

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

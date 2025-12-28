#!/usr/bin/env python3
import argparse
from pathlib import Path

from dotenv import load_dotenv
from finrag.eval.generation import generate_eval_items_from_chunks
from finrag.eval.io import dump_jsonl
from finrag.eval.sec_corpus import load_sec_download_dir
from finrag.llm_clients import get_llm_client
from finrag.sec_chunking import SecHtmlChunker

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate a synthetic eval set from downloaded SEC filings (JSONL).")
    ap.add_argument("--data-dir", required=True, help="Directory containing `10k_raw/` and `meta/`")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--tickers", nargs="*", default=None, help="Optional tickers filter (e.g. AAPL MSFT)")
    ap.add_argument("--forms", nargs="*", default=None, help="Optional forms filter (e.g. 10-K 10-Q)")
    ap.add_argument("--max-docs", type=int, default=50, help="Max documents to scan")
    ap.add_argument("--max-words", type=int, default=350, help="Chunk max words")
    ap.add_argument("--overlap-words", type=int, default=50, help="Chunk overlap words")
    ap.add_argument("--n-quant", type=int, default=20, help="Number of quantitative questions")
    ap.add_argument("--n-qual", type=int, default=20, help="Number of qualitative questions")
    ap.add_argument("--n-mixed", type=int, default=10, help="Number of mixed questions")
    ap.add_argument("--n-series", type=int, default=0, help="Number of multi-doc series questions (e.g. QoQ growth)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed")
    ap.add_argument("--paraphrase-with-llm", action="store_true", help="Use configured LLM to paraphrase questions")
    ap.add_argument("--llm-provider", default=None, help="Overrides LLM_PROVIDER (e.g. mistral, openai, fastembed)")
    ap.add_argument("--llm-kwargs", default=None, help="JSON dict passed to get_llm_client(...) (advanced)")

    args = ap.parse_args()

    docs = load_sec_download_dir(args.data_dir, tickers=args.tickers, forms=args.forms, max_docs=args.max_docs)
    if not docs:
        raise SystemExit("No documents found. Check --data-dir and filters.")

    chunker = SecHtmlChunker(max_words=args.max_words, overlap_words=args.overlap_words)
    chunks = []
    doc_meta_by_id = {}
    for doc in docs:
        doc_meta_by_id[doc.doc_id] = doc.meta
        chunks.extend(chunker.chunk_html_file(doc.source_path, doc_id=doc.doc_id, metadata=doc.meta))

    llm_kwargs = {}
    if args.llm_kwargs:
        import json

        llm_kwargs = json.loads(args.llm_kwargs)
        if not isinstance(llm_kwargs, dict):
            raise ValueError("--llm-kwargs must be a JSON object")
    llm = get_llm_client(provider=args.llm_provider, **llm_kwargs) if args.paraphrase_with_llm else None
    items = generate_eval_items_from_chunks(
        chunks,
        doc_meta_by_id=doc_meta_by_id,
        n_quantitative=args.n_quant,
        n_qualitative=args.n_qual,
        n_mixed=args.n_mixed,
        n_series=args.n_series,
        seed=args.seed,
        llm_for_paraphrase=llm,
    )

    out_path = Path(args.out)
    dump_jsonl(items, out_path)
    print(f"Wrote {len(items)} eval items to {out_path}")


if __name__ == "__main__":
    main()

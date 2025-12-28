"""
Chunk a directory of Markdown filings using `finrag.chunking.DoclingHybridChunker`.

This script focuses on chunking + exporting chunks (and metadata) to disk.
Indexing into a vector store can be done as a separate step.
"""

import argparse
import json
import os
import sys
import time
import uuid
from dataclasses import asdict, dataclass
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from finrag.chunk_postprocess import (
    DocumentContextPostprocessor,
    HeuristicSummaryPostprocessor,
    SectionLinkPostprocessor,
    YahooFinanceCompanyNameResolver,
)
from finrag.chunking import DoclingHybridChunker
from finrag.dataclasses import DocChunk

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


@dataclass
class Args:
    markdown_dir: str
    output_dir: str
    pattern: str
    recursive: bool
    max_files: int | None
    year_cutoff: int | None
    overwrite: bool
    doc_id_strategy: str
    hf_offline: bool
    tokenizer_model: str
    max_tokens: int
    overlap_tokens: int
    preprocess_markdown_tables: bool
    markdown_table_fence_lang: str
    section_neighbor_window: int
    max_summary_chars: int
    company_name_resolver: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_args() -> Args:
    parser = argparse.ArgumentParser(description="Chunk a directory of Markdown files using DoclingHybridChunker.")
    parser.add_argument(
        "--markdown-dir",
        required=True,
        help="Directory containing Markdown files to chunk (e.g. data/sec_filings/processed_markdown).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/chunks_docling",
        help="Directory to write chunk exports (creates `chunks/`, `doc_index.jsonl`, `run_info.json`).",
    )
    parser.add_argument("--pattern", default="*.md", help="Glob pattern for markdown files.")
    parser.add_argument("--recursive", action="store_true", help="Recurse into subdirectories.")
    parser.add_argument("--max-files", type=int, default=None, help="Optional cap on number of files to process.")
    parser.add_argument(
        "--year-cutoff",
        type=int,
        default=None,
        help="Only process filings from this year (YYYY) based on the filename date.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-document chunk files.")
    parser.add_argument(
        "--doc-id-strategy",
        choices=["uuid", "sha1_relpath", "stem"],
        default="uuid",
        help="How to generate `doc_id` for each markdown file.",
    )

    hf_group = parser.add_mutually_exclusive_group()
    hf_group.add_argument(
        "--hf-offline",
        dest="hf_offline",
        action="store_true",
        help="Force HuggingFace/transformers to run in offline mode (no downloads).",
    )
    hf_group.add_argument(
        "--hf-online",
        dest="hf_offline",
        action="store_false",
        help="Allow HuggingFace/transformers to download models if needed.",
    )
    parser.set_defaults(hf_online=True)

    # Chunker options
    parser.add_argument(
        "--tokenizer-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HuggingFace tokenizer model used by Docling hybrid chunker.",
    )
    parser.add_argument("--max-tokens", type=int, default=1024, help="Max tokens per chunk (approx).")
    parser.add_argument("--overlap-tokens", type=int, default=128, help="Overlap tokens between chunks.")

    md_table_group = parser.add_mutually_exclusive_group()
    md_table_group.add_argument(
        "--preprocess-markdown-tables",
        dest="preprocess_markdown_tables",
        action="store_true",
        help="Fence pipe tables before Docling parses markdown (recommended for SEC filings).",
    )
    md_table_group.add_argument(
        "--no-preprocess-markdown-tables",
        dest="preprocess_markdown_tables",
        action="store_false",
        help="Disable table fencing before Docling parses markdown.",
    )
    parser.set_defaults(preprocess_markdown_tables=True)

    parser.add_argument(
        "--markdown-table-fence-lang",
        default="table",
        help="Fence language to use when `--preprocess-markdown-tables` is enabled.",
    )

    # Postprocessors
    parser.add_argument(
        "--company-name-resolver",
        choices=["yahoo", "none"],
        default="yahoo",
        help="Resolve ticker->company name (yahoo requires `yfinance` and network access).",
    )
    parser.add_argument(
        "--section-neighbor-window", type=int, default=2, help="SectionLinkPostprocessor neighbor window size."
    )
    parser.add_argument(
        "--max-summary-chars", type=int, default=300, help="HeuristicSummaryPostprocessor max summary length."
    )

    args = parser.parse_args()
    return Args(
        markdown_dir=args.markdown_dir,
        output_dir=args.output_dir,
        pattern=args.pattern,
        recursive=bool(args.recursive),
        max_files=args.max_files,
        year_cutoff=args.year_cutoff,
        overwrite=bool(args.overwrite),
        doc_id_strategy=args.doc_id_strategy,
        hf_offline=bool(args.hf_offline),
        tokenizer_model=args.tokenizer_model,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        preprocess_markdown_tables=bool(args.preprocess_markdown_tables),
        markdown_table_fence_lang=args.markdown_table_fence_lang,
        section_neighbor_window=args.section_neighbor_window,
        max_summary_chars=args.max_summary_chars,
        company_name_resolver=args.company_name_resolver,
    )


def _extract_year_from_filename(path: Path) -> int | None:
    # Expects: *_YYYY-MM-DD(.md)
    stem = path.stem
    if len(stem) < 10:
        return None
    tail = stem[-10:]
    try:
        yyyy, mm, dd = tail.split("-")
        _ = int(mm)
        _ = int(dd)
        return int(yyyy)
    except Exception:
        return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, set):
        return list(obj)
    if hasattr(obj, "isoformat"):
        try:
            return obj.isoformat()  # type: ignore[no-any-return]
        except Exception:
            pass
    return str(obj)


def _stable_sha1(text: str, n: int = 16) -> str:
    return sha1(text.encode("utf-8")).hexdigest()[:n]


def _make_doc_id(md_path: Path, *, markdown_root: Path, strategy: str) -> str:
    if strategy == "uuid":
        return str(uuid.uuid4())
    if strategy == "stem":
        return md_path.stem
    if strategy == "sha1_relpath":
        rel = md_path.resolve().relative_to(markdown_root.resolve()).as_posix()
        return _stable_sha1(rel, n=20)
    raise ValueError(f"Unknown doc id strategy: {strategy}")


def _iter_markdown_files(root: Path, *, pattern: str, recursive: bool) -> Iterable[Path]:
    if recursive:
        yield from root.rglob(pattern)
    else:
        yield from root.glob(pattern)


def _setup_logging(project_root: Path) -> Path:
    logs_dir = project_root / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"chunk_{ts}.log"

    logger.remove()
    logger.add(sys.stderr, level="INFO")
    logger.add(str(log_path), level="DEBUG")
    return log_path


def _build_chunker(args: Args) -> DoclingHybridChunker:
    tokenizer_kwargs: dict[str, Any] = {}
    if args.hf_offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        tokenizer_kwargs["local_files_only"] = True

    resolver = YahooFinanceCompanyNameResolver() if args.company_name_resolver == "yahoo" else None
    return DoclingHybridChunker(
        tokenizer_model=args.tokenizer_model,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        preprocess_markdown_tables=args.preprocess_markdown_tables,
        markdown_table_fence_lang=args.markdown_table_fence_lang,
        tokenizer_kwargs=tokenizer_kwargs,
        chunk_postprocessors=[
            DocumentContextPostprocessor(company_name_resolver=resolver),
            SectionLinkPostprocessor(neighbor_window=args.section_neighbor_window),
            HeuristicSummaryPostprocessor(max_summary_chars=args.max_summary_chars),
        ],
    )


def _chunk_to_dict(chunk: DocChunk) -> dict[str, Any]:
    return asdict(chunk)


def main() -> int:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    markdown_root = Path(args.markdown_dir).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    output_chunks_root = output_root / "chunks"
    output_chunks_root.mkdir(parents=True, exist_ok=True)

    log_path = _setup_logging(project_root)
    logger.info(f"Project root: {project_root}")
    logger.info(f"Logging to: {log_path}")

    md_files = [
        p for p in _iter_markdown_files(markdown_root, pattern=args.pattern, recursive=args.recursive) if p.is_file()
    ]
    md_files.sort()

    if args.year_cutoff is not None:
        before = len(md_files)
        md_files = [p for p in md_files if (_extract_year_from_filename(p) or 0) >= args.year_cutoff]
        logger.info(f"Year cutoff {args.year_cutoff}: {before} -> {len(md_files)} files")

    if args.max_files is not None:
        md_files = md_files[: max(0, args.max_files)]

    logger.info(f"Found {len(md_files)} markdown files under {markdown_root}")

    chunker = _build_chunker(args)

    doc_index_path = output_root / "doc_index.jsonl"
    errors_path = output_root / "errors.jsonl"
    run_info_path = output_root / "run_info.json"

    started_at = time.time()
    processed = 0
    total_chunks = 0

    with doc_index_path.open("a", encoding="utf-8") as doc_index_f, errors_path.open("a", encoding="utf-8") as errors_f:
        for md_path in tqdm(md_files, desc="chunking markdown"):
            rel = md_path.resolve().relative_to(markdown_root).as_posix()
            out_path = (output_chunks_root / rel).with_suffix(".jsonl")
            out_path.parent.mkdir(parents=True, exist_ok=True)

            if out_path.exists() and not args.overwrite:
                logger.info(f"Skipping (exists): {out_path}")
                continue

            doc_start_time = time.time()
            doc_id = _make_doc_id(md_path, markdown_root=markdown_root, strategy=args.doc_id_strategy)
            try:
                chunks = chunker.chunk_document(str(md_path), doc_id)

                with out_path.open("w", encoding="utf-8") as out_f:
                    for ch in chunks:
                        out_f.write(json.dumps(_chunk_to_dict(ch), ensure_ascii=False, default=_json_default) + "\n")

                rec = {
                    "doc_id": doc_id,
                    "source": str(md_path),
                    "relpath": rel,
                    "chunks_path": str(out_path),
                    "num_chunks": len(chunks),
                    "chunk_time_s": round(time.time() - doc_start_time, 3),
                }
                doc_index_f.write(json.dumps(rec, ensure_ascii=False, default=_json_default) + "\n")

                processed += 1
                total_chunks += len(chunks)
                logger.success(f"Chunked {md_path} -> {out_path} | chunks={len(chunks)} | time_s={rec['chunk_time_s']}")

            except Exception as e:  # noqa: BLE001 - best-effort batch processing
                logger.exception(f"Failed to chunk {md_path}: {e}")
                err = {"source": str(md_path), "relpath": rel, "error": repr(e)}
                errors_f.write(json.dumps(err, ensure_ascii=False, default=_json_default) + "\n")

    run_info = {
        "args": args.to_dict(),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "elapsed_s": round(time.time() - started_at, 3),
        "markdown_dir": str(markdown_root),
        "output_dir": str(output_root),
        "processed_files": processed,
        "total_chunks": total_chunks,
        "doc_index_path": str(doc_index_path),
        "errors_path": str(errors_path),
    }
    run_info_path.write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8"
    )

    logger.success(f"Done. processed_files={processed} total_chunks={total_chunks}")
    logger.success(f"Wrote: {doc_index_path}")
    logger.success(f"Wrote: {output_chunks_root}")
    logger.success(f"Wrote: {run_info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

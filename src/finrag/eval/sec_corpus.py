from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from finrag.dataclasses import DocChunk

_FILENAME_RE = re.compile(
    r"^(?P<ticker>[A-Za-z0-9.]+)_(?P<accession>\d{18})_(?P<form>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})$"
)


@dataclass(frozen=True)
class ChunkExportDoc:
    doc_id: str
    source: str
    chunks_path: str
    relpath: str | None = None
    ticker: str | None = None
    filing_type: str | None = None
    filing_date: str | None = None
    year: int | None = None
    company: str | None = None


def _parse_doc_from_source(source: str | None, relpath: str | None) -> dict[str, Any]:
    path_s = (relpath or source or "").strip()
    if not path_s:
        return {}
    stem = Path(path_s).stem
    m = _FILENAME_RE.match(stem)
    if not m:
        return {}
    ticker = m.group("ticker").upper()
    filing_type = m.group("form").upper()
    filing_date = m.group("date")
    try:
        year = int(filing_date.split("-", 1)[0])
    except Exception:
        year = None
    return {"ticker": ticker, "filing_type": filing_type, "filing_date": filing_date, "year": year}


def _resolve_chunks_path(ingest_output_dir: Path, chunks_path: str) -> Path:
    p = Path(chunks_path).expanduser()
    if not p.is_absolute():
        p = (ingest_output_dir / p).resolve()
    return p


def _peek_company_from_chunks(chunks_path: Path, *, max_lines: int = 30) -> str | None:
    try:
        with chunks_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                if i > max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                meta = d.get("metadata")
                if not isinstance(meta, dict):
                    continue
                doc = meta.get("doc")
                if not isinstance(doc, dict):
                    continue
                company = doc.get("company")
                if isinstance(company, str) and company.strip():
                    return company.strip()
    except Exception:
        return None
    return None


def iter_chunk_export_docs(
    ingest_output_dir: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    forms: Iterable[str] | None = None,
    max_docs: int | None = None,
) -> Iterator[ChunkExportDoc]:
    """
    Iterate the chunk export "doc index" (produced by `scripts/chunk.py`).

    `ingest_output_dir` is expected to contain:
      - doc_index.jsonl
      - chunks/ (per-document chunk JSONL files)
    """
    root = Path(ingest_output_dir).expanduser().resolve()
    doc_index_path = root / "doc_index.jsonl"
    if not doc_index_path.exists():
        raise FileNotFoundError(f"Missing doc index: {doc_index_path}")

    ticker_set = {t.upper() for t in tickers} if tickers else None
    form_set = {f.upper() for f in forms} if forms else None

    n = 0
    with doc_index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            doc_id = str(d.get("doc_id") or "")
            source = str(d.get("source") or "")
            relpath = str(d.get("relpath") or "") or None
            chunks_path_raw = str(d.get("chunks_path") or "")
            if not doc_id or not chunks_path_raw:
                continue

            parsed = _parse_doc_from_source(source, relpath)
            ticker = parsed.get("ticker")
            filing_type = parsed.get("filing_type")

            if ticker_set and isinstance(ticker, str) and ticker not in ticker_set:
                continue
            if form_set and isinstance(filing_type, str) and filing_type not in form_set:
                continue

            chunks_path = _resolve_chunks_path(root, chunks_path_raw)
            company = _peek_company_from_chunks(chunks_path)

            yield ChunkExportDoc(
                doc_id=doc_id,
                source=source,
                relpath=relpath,
                chunks_path=str(chunks_path),
                ticker=ticker if isinstance(ticker, str) else None,
                filing_type=filing_type if isinstance(filing_type, str) else None,
                filing_date=parsed.get("filing_date") if isinstance(parsed.get("filing_date"), str) else None,
                year=parsed.get("year") if isinstance(parsed.get("year"), int) else None,
                company=company,
            )

            n += 1
            if max_docs is not None and n >= max_docs:
                break


def iter_doc_chunks(chunks_path: str | Path, *, max_chunks: int | None = None) -> Iterator[DocChunk]:
    """
    Stream chunks from a per-document chunks JSONL file.
    """
    p = Path(chunks_path).expanduser()
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            yield DocChunk(
                id=str(d["id"]),
                doc_id=str(d["doc_id"]),
                text=str(d.get("text") or ""),
                page_no=d.get("page_no"),
                headings=list(d.get("headings") or []),
                source=str(d.get("source") or ""),
                metadata=d.get("metadata") if isinstance(d.get("metadata"), dict) else None,
            )
            n += 1
            if max_chunks is not None and n >= max_chunks:
                break


def iter_all_chunks(
    ingest_output_dir: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    forms: Iterable[str] | None = None,
    max_docs: int | None = None,
    max_chunks_per_doc: int | None = None,
) -> Iterator[DocChunk]:
    """
    Stream chunks across many documents from a chunk export directory.
    """
    for doc in iter_chunk_export_docs(ingest_output_dir, tickers=tickers, forms=forms, max_docs=max_docs):
        yield from iter_doc_chunks(doc.chunks_path, max_chunks=max_chunks_per_doc)

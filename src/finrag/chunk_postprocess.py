import hashlib
import re
from datetime import date, datetime
from pathlib import Path
from collections import defaultdict
from typing import Protocol

from loguru import logger

from finrag.dataclasses import DocChunk


class ChunkPostprocessor(Protocol):
    def process(self, chunks: list[DocChunk]) -> list[DocChunk]: ...


class CompanyNameResolver(Protocol):
    def resolve(self, *, ticker: str | None, cik: str | None) -> str | None: ...


class StaticTickerCompanyNameResolver:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = {k.upper(): v for k, v in mapping.items()}

    def resolve(self, *, ticker: str | None, cik: str | None) -> str | None:
        _ = cik
        if not ticker:
            return None
        return self.mapping.get(ticker.upper())


class YahooFinanceCompanyNameResolver:
    """
    Best-effort ticker -> company name using Yahoo Finance (via `yfinance`).

    Notes:
      - Requires `yfinance` installed and network access at runtime.
      - Uses in-memory caching per process to avoid repeated lookups.
    """

    def __init__(self):
        self._cache: dict[str, str | None] = {}

    def resolve(self, *, ticker: str | None, cik: str | None) -> str | None:
        _ = cik
        if not ticker:
            return None
        key = ticker.upper()
        if key in self._cache:
            return self._cache[key]

        try:
            import yfinance as yf  # type: ignore[import-not-found]
        except Exception as e:
            logger.warning(f"yfinance not available: {e}")
            self._cache[key] = None
            return None

        try:
            info = yf.Ticker(key).get_info()
        except Exception:
            self._cache[key] = None
            return None

        name = (
            info.get("longName")
            or info.get("shortName")
            or info.get("legalName")
            or info.get("name")
            or info.get("displayName")
        )
        if isinstance(name, str):
            name = re.sub(r"\s+", " ", name).strip()
        else:
            name = None

        self._cache[key] = name
        return name


class ChunkPostprocessorPipeline:
    def __init__(self, postprocessors: list[ChunkPostprocessor]):
        self.postprocessors = list(postprocessors)

    def process(self, chunks: list[DocChunk]) -> list[DocChunk]:
        out = list(chunks)
        for pp in self.postprocessors:
            out = pp.process(out)
        return out


class SectionLinkPostprocessor:
    """
    Adds stable section IDs and lightweight related-chunk pointers.

    A "section" is defined by the full headings path (DocChunk.headings).
    """

    def __init__(
        self,
        *,
        neighbor_window: int = 2,
        include_section_chunk_ids: bool = False,
        include_section_related_ids: bool = False,
        max_related_ids: int = 50,
    ):
        self.neighbor_window = int(neighbor_window)
        self.include_section_chunk_ids = include_section_chunk_ids
        self.include_section_related_ids = include_section_related_ids
        self.max_related_ids = int(max_related_ids)

    @staticmethod
    def _ensure_meta(chunk: DocChunk) -> dict:
        if chunk.metadata is None:
            chunk.metadata = {}
        return chunk.metadata

    @staticmethod
    def _stable_id(text: str) -> str:
        return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]

    def process(self, chunks: list[DocChunk]) -> list[DocChunk]:
        if not chunks:
            return chunks

        # Global adjacency.
        for i, ch in enumerate(chunks):
            meta = self._ensure_meta(ch)
            meta["prev_chunk_id"] = chunks[i - 1].id if i > 0 else None
            meta["next_chunk_id"] = chunks[i + 1].id if i + 1 < len(chunks) else None

        # Group by headings path (section).
        groups: dict[tuple[str, ...], list[DocChunk]] = defaultdict(list)
        for ch in chunks:
            groups[tuple(ch.headings)].append(ch)

        for headings_path, group in groups.items():
            # Use doc_id from first chunk; doc_id should be uniform within a document.
            doc_id = group[0].doc_id
            section_path = " > ".join([h for h in headings_path if h])
            section_id = self._stable_id(f"{doc_id}|{section_path}")

            parent_section_id = None
            if len(headings_path) > 1:
                parent_path = " > ".join([h for h in headings_path[:-1] if h])
                if parent_path:
                    parent_section_id = self._stable_id(f"{doc_id}|{parent_path}")

            section_chunk_ids = [c.id for c in group]
            for j, ch in enumerate(group):
                meta = self._ensure_meta(ch)
                meta["section_id"] = section_id
                meta["section_path"] = section_path
                meta["parent_section_id"] = parent_section_id
                meta["section_index"] = j
                meta["section_size"] = len(group)
                meta["section_prev_chunk_id"] = group[j - 1].id if j > 0 else None
                meta["section_next_chunk_id"] = group[j + 1].id if j + 1 < len(group) else None

                if self.neighbor_window > 0:
                    lo = max(0, j - self.neighbor_window)
                    hi = min(len(group), j + self.neighbor_window + 1)
                    neighbors = [c.id for c in group[lo:hi] if c.id != ch.id]
                    meta["section_neighbor_chunk_ids"] = neighbors

                if self.include_section_chunk_ids:
                    meta["section_chunk_ids"] = section_chunk_ids

                if self.include_section_related_ids:
                    related = [cid for cid in section_chunk_ids if cid != ch.id]
                    if self.max_related_ids > 0:
                        related = related[: self.max_related_ids]
                    meta["related_chunk_ids"] = related

        return chunks


class DocumentContextPostprocessor:
    """
    Adds document-level context to each chunk under `chunk.metadata['doc']`.

    Intended fields (when available):
      - company, ticker, cik, accession
      - filing_type (e.g. '10-Q'), filing_date (YYYY-MM-DD)
      - period_end_date (YYYY-MM-DD) if detected
      - filing_quarter (e.g. '2024Q3') and `filing_quarter_basis`

    Data sources (best-effort):
      - `chunk.source` filename pattern: TICKER_ACCESSION_FORM_YYYY-MM-DD(.md)
      - `chunk.headings` (often includes the company name)
      - early text chunks (for "For the quarterly period ended ...")
    """

    _FILENAME_RE = re.compile(
        r"^(?P<ticker>[A-Za-z0-9.]+)_(?P<accession>\d{18})_(?P<form>[^_]+)_(?P<date>\d{4}-\d{2}-\d{2})$"
    )
    _PERIOD_ENDED_RE = re.compile(
        r"for the (?:quarterly period|fiscal year) ended\s+(?P<date>[A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})", re.IGNORECASE
    )

    def __init__(
        self,
        *,
        scan_chunks: int = 15,
        scan_chars_per_chunk: int = 2500,
        company_name_resolver: CompanyNameResolver | None = None,
        fallback_company_from_headings: bool = False,
    ):
        self.scan_chunks = int(scan_chunks)
        self.scan_chars_per_chunk = int(scan_chars_per_chunk)
        self.company_name_resolver = company_name_resolver
        self.fallback_company_from_headings = bool(fallback_company_from_headings)

    @staticmethod
    def _ensure_meta(chunk: DocChunk) -> dict:
        if chunk.metadata is None:
            chunk.metadata = {}
        return chunk.metadata

    @staticmethod
    def _parse_iso_date(s: str) -> date | None:
        try:
            return datetime.strptime(s, "%Y-%m-%d").date()
        except ValueError:
            return None

    @staticmethod
    def _parse_month_date(s: str) -> date | None:
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(s, fmt).date()
            except ValueError:
                continue
        return None

    @staticmethod
    def _quarter_label(d: date) -> str:
        q = (d.month - 1) // 3 + 1
        return f"{d.year}Q{q}"

    @staticmethod
    def _pick_company_from_headings(chunks: list[DocChunk]) -> str | None:
        # Best-effort fallback only: pick a company-like heading (often "ASTERA LABS, INC.").
        bad = ("united states securities and exchange commission", "table of contents")
        company_tokens = (" inc", "inc.", " corp", "corporation", " ltd", "llc", " plc", " co", "company")

        seen: set[str] = set()
        headings: list[str] = []
        for ch in chunks[:10]:
            for h in ch.headings:
                h = h.strip()
                if not h or h in seen:
                    continue
                seen.add(h)
                headings.append(h)

        for h in headings:
            h_cf = h.casefold()
            if any(b in h_cf for b in bad):
                continue
            if any(tok in h_cf for tok in company_tokens):
                return h

        for h in headings:
            h_cf = h.casefold()
            if any(b in h_cf for b in bad):
                continue
            if h.isupper() and len(h) >= 8:
                return h

        return None

    def _extract_from_filename(self, source: str) -> dict:
        stem = Path(source).stem
        m = self._FILENAME_RE.match(stem)
        if not m:
            return {}
        ticker = m.group("ticker")
        accession = m.group("accession")
        form = m.group("form")
        filing_date = self._parse_iso_date(m.group("date"))
        cik = accession[:10] if len(accession) >= 10 else None
        out = {
            "ticker": ticker,
            "accession": accession,
            "cik": cik,
            "filing_type": form,
            "filing_date": filing_date.isoformat() if filing_date else None,
        }
        return {k: v for k, v in out.items() if v is not None}

    def _extract_period_end(self, chunks: list[DocChunk]) -> date | None:
        for ch in chunks[: self.scan_chunks]:
            text = ch.text[: self.scan_chars_per_chunk]
            m = self._PERIOD_ENDED_RE.search(text)
            if not m:
                continue
            d = self._parse_month_date(m.group("date"))
            if d:
                return d
        return None

    def process(self, chunks: list[DocChunk]) -> list[DocChunk]:
        by_doc: dict[str, list[DocChunk]] = defaultdict(list)
        for ch in chunks:
            by_doc[ch.doc_id].append(ch)

        for doc_id, group in by_doc.items():
            source = group[0].source
            doc_ctx: dict = {}

            doc_ctx.update(self._extract_from_filename(source))

            existing_doc = (group[0].metadata or {}).get("doc")
            existing_doc = existing_doc if isinstance(existing_doc, dict) else {}
            ticker = existing_doc.get("ticker") or doc_ctx.get("ticker")
            cik = existing_doc.get("cik") or doc_ctx.get("cik")

            if not existing_doc.get("company") and self.company_name_resolver is not None:
                company = self.company_name_resolver.resolve(
                    ticker=str(ticker) if isinstance(ticker, str) else None,
                    cik=str(cik) if isinstance(cik, str) else None,
                )
                if company:
                    logger.info(f"Resolved company name for ticker={ticker}: {company}")
                    doc_ctx["company"] = company
            elif not existing_doc.get("company") and self.fallback_company_from_headings:
                company = self._pick_company_from_headings(group)
                if company:
                    doc_ctx["company"] = company

            period_end = self._extract_period_end(group)
            if period_end:
                doc_ctx["period_end_date"] = period_end.isoformat()

            # filing_quarter: prefer period_end_date; fall back to filing_date.
            basis = None
            quarter_date = period_end
            if quarter_date is not None:
                basis = "period_end_date"
            else:
                filing_date_s = doc_ctx.get("filing_date")
                quarter_date = self._parse_iso_date(filing_date_s) if isinstance(filing_date_s, str) else None
                basis = "filing_date" if quarter_date else None
            if quarter_date:
                doc_ctx["filing_quarter"] = self._quarter_label(quarter_date)
                if basis:
                    doc_ctx["filing_quarter_basis"] = basis

            # Attach to all chunks, merging with any existing keys.
            for ch in group:
                meta = self._ensure_meta(ch)
                existing = meta.get("doc")
                if isinstance(existing, dict):
                    merged = dict(existing)
                    for k, v in doc_ctx.items():
                        merged.setdefault(k, v)
                    meta["doc"] = merged
                else:
                    meta["doc"] = dict(doc_ctx)

        return chunks


class HeuristicSummaryPostprocessor:
    """
    Adds a short summary + an optional index/embedding text prefix.

    The summary is stored in `chunk.metadata['summary']`.
    The enriched text for embedding/BM25 is stored in `chunk.metadata['index_text']`.
    """

    _TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

    def __init__(
        self,
        *,
        max_summary_chars: int = 300,
        include_headings_in_index_text: bool = True,
        include_page_in_index_text: bool = True,
        include_summary_in_index_text: bool = True,
        include_doc_context_in_index_text: bool = True,
    ):
        self.max_summary_chars = int(max_summary_chars)
        self.include_headings_in_index_text = include_headings_in_index_text
        self.include_page_in_index_text = include_page_in_index_text
        self.include_summary_in_index_text = include_summary_in_index_text
        self.include_doc_context_in_index_text = include_doc_context_in_index_text

    @staticmethod
    def _ensure_meta(chunk: DocChunk) -> dict:
        if chunk.metadata is None:
            chunk.metadata = {}
        return chunk.metadata

    @classmethod
    def _looks_like_pipe_table(cls, text: str) -> bool:
        # FIXME: this doesn't work. it might be too rigid as well
        # eg text could be some text followed by table, so lines[0] may not have any "|"
        lines = [ln for ln in text.splitlines() if ln.strip() and ln != "```"]
        if len(lines) < 2:
            return False
        return ("|" in lines[0]) and bool(cls._TABLE_SEPARATOR_RE.match(lines[1]))

    @staticmethod
    def _truncate(text: str, limit: int) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if limit <= 0:
            return ""
        if len(text) <= limit:
            return text
        return text[: max(0, limit - 1)].rstrip() + "…"

    def _summarize_table(self, text: str, headings: list[str]) -> str:
        # TODO: tables are not being captured.
        lines = [ln for ln in text.splitlines() if ln.strip() and ln != "```"]
        header = lines[0].strip()
        cols = [c.strip() for c in header.strip("|").split("|")]
        cols = [c for c in cols if c]
        topic = headings[-1] if headings else "Table"
        if cols:
            col_str = ", ".join(cols[:8])
            if len(cols) > 8:
                col_str += ", …"
            return self._truncate(f"{topic} (table). Columns: {col_str}.", self.max_summary_chars)
        return self._truncate(f"{topic} (table).", self.max_summary_chars)

    def _summarize_text(self, text: str, headings: list[str]) -> str:
        raise RuntimeError("Current logic is not useful. Maybe we can use an LLM instead.")
        # Grab first non-empty line and then first sentence-ish.
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return ""
        first = lines[0]
        # If the first line is huge, just truncate it.
        if len(first) > self.max_summary_chars:
            return self._truncate(first, self.max_summary_chars)
        parts = re.split(r"(?<=[.!?])\s+", first, maxsplit=1)
        sent = parts[0] if parts else first
        topic = headings[-1] if headings else ""
        if topic and topic.casefold() not in sent.casefold():
            sent = f"{topic}: {sent}"
        return self._truncate(sent, self.max_summary_chars)

    def process(self, chunks: list[DocChunk]) -> list[DocChunk]:
        for ch in chunks:
            meta = self._ensure_meta(ch)

            # TODO: tables are not being captured. _looks_like_pipe_table() is not good enough.
            is_table = bool(meta.get("block_type") == "table") or self._looks_like_pipe_table(ch.text)
            if is_table:
                summary = self._summarize_table(ch.text, ch.headings)
            else:
                # summary = self._summarize_text(ch.text, ch.headings)
                summary = ""

            if summary:
                meta["summary"] = summary

            # Build index_text used for embedding + BM25, without mutating `chunk.text`.
            prefix_lines: list[str] = []
            doc = meta.get("doc") if isinstance(meta.get("doc"), dict) else {}
            if self.include_doc_context_in_index_text and doc:
                company = doc.get("company")
                ticker = doc.get("ticker")
                filing_type = doc.get("filing_type")
                filing_date = doc.get("filing_date")
                period_end_date = doc.get("period_end_date")
                filing_quarter = doc.get("filing_quarter")
                # cik = doc.get("cik")

                if company:
                    prefix_lines.append(f"Company: {company}")
                if ticker:
                    prefix_lines.append(f"Ticker: {ticker}")
                # if cik:
                #     prefix_lines.append(f"CIK: {cik}")
                filing_bits = []
                if filing_type:
                    filing_bits.append(str(filing_type))
                if filing_date:
                    filing_bits.append(f"filed {filing_date}")
                if period_end_date:
                    filing_bits.append(f"period ended {period_end_date}")
                if filing_bits:
                    prefix_lines.append("Filing: " + ", ".join(filing_bits))
                if filing_quarter:
                    prefix_lines.append(f"Filing quarter: {filing_quarter}")

            if self.include_headings_in_index_text and ch.headings:
                prefix_lines.append("Section: " + " > ".join(ch.headings))
            if self.include_page_in_index_text and ch.page_no is not None:
                prefix_lines.append(f"Page: {ch.page_no}")
            if self.include_summary_in_index_text and summary:
                prefix_lines.append("Summary: " + summary)

            if prefix_lines:
                meta["index_text"] = "\n".join(prefix_lines).strip() + "\n\n" + ch.text
            else:
                meta["index_text"] = ch.text

        return chunks

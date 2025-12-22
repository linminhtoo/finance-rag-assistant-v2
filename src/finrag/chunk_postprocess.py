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



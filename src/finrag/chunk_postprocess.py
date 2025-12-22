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

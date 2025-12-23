from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from finrag.sec_chunking import SecDocument


def load_sec_download_dir(
    data_dir: str | Path,
    *,
    tickers: Iterable[str] | None = None,
    forms: Iterable[str] | None = None,
    max_docs: int | None = None,
) -> list[SecDocument]:
    root = Path(data_dir)
    raw_dir = root / "10k_raw"
    meta_dir = root / "meta"
    if not raw_dir.exists() or not meta_dir.exists():
        raise FileNotFoundError(f"Expected {raw_dir} and {meta_dir} to exist")

    ticker_set = {t.upper() for t in tickers} if tickers else None
    form_set = {f.upper() for f in forms} if forms else None

    docs: list[SecDocument] = []
    for meta_path in sorted(meta_dir.glob("*.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        ticker = str(meta.get("ticker", "")).upper()
        form = str(meta.get("form", "")).upper()
        if ticker_set and ticker not in ticker_set:
            continue
        if form_set and form not in form_set:
            continue

        base = meta_path.stem
        html_path = raw_dir / f"{base}.html"
        if not html_path.exists():
            htm_path = raw_dir / f"{base}.htm"
            if htm_path.exists():
                html_path = htm_path
            else:
                continue

        doc_id = base
        meta = dict(meta)
        meta["doc_id"] = doc_id
        meta["meta_path"] = str(meta_path)

        docs.append(SecDocument(doc_id=doc_id, source_path=str(html_path), meta=meta))
        if max_docs is not None and len(docs) >= max_docs:
            break

    return docs


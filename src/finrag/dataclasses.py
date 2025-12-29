from dataclasses import dataclass
from typing import Any


@dataclass
class DocChunk:
    """
    A chunk of a document with associated metadata.

    TODO: describe Attributes
    """

    id: str
    doc_id: str
    text: str
    page_no: int | None
    headings: list[str]
    source: str  # original file/URL
    metadata: dict[str, Any] | None = None
    # add more metadata fields as needed (section path, captions, etc.)

    def as_payload(self) -> dict:
        payload = {
            "chunk_id": self.id,
            "doc_id": self.doc_id,
            "page_no": self.page_no,
            "headings": self.headings,
            "source": self.source,
            "text": self.text,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


@dataclass
class ScoredChunk:
    """
    A document chunk with an associated relevance score.

    TODO: describe Attributes
    """

    chunk: DocChunk
    score: float
    # TODO: define Enum ?
    source: str  # "hybrid" or "reranker"


@dataclass
class TopChunk:
    """
    A top chunk returned in a query response.
    """

    chunk_id: str
    doc_id: str
    page_no: int | None
    headings: list[str]
    score: float
    preview: str  # first N chars of text
    source: str  # original file/URL
    text: str | None = None  # optional full chunk text (can be large)
    context: str | None = None  # optional situated/enriched context text
    metadata: dict[str, Any] | None = None  # optional metadata (company, ticker, filing_date, summary, ...)

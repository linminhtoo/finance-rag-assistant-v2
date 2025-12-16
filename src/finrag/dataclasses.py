from dataclasses import dataclass


@dataclass
class DocChunk:
    """
    A chunk of a document with associated metadata.

    TODO: describe Attributes
    """

    id: str
    doc_id: str
    text: str
    # page_no: Optional[int]
    headings: list[str]
    source: str  # original file/URL
    # you can add more metadata fields as needed (section path, captions, etc.)

    def as_payload(self) -> dict:
        return {
            "doc_id": self.doc_id,
            # "page_no": self.page_no,
            "headings": self.headings,
            "source": self.source,
        }


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

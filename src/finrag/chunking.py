import os
import tempfile
from dataclasses import dataclass
from typing import Iterable, List, Optional

from docling.document_converter import DocumentConverter
# installed via docling-core[chunking]
from docling_core.types.doc.document import DoclingDocument
from docling_core.transforms.chunker.doc_chunk import DocChunk as DoclingDocChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer

from finrag.dataclasses import DocChunk
from finrag.ocr import MistralOCRClient


class DoclingHybridChunker:
    """
    End-to-end 'doc -> DoclingDocument -> Hierarchical+Hybrid chunking' pipeline.

    - Optionally uses Mistral OCR first (MistralOCRClient -> Markdown).
    - Always goes through Docling's DocumentConverter to get a DoclingDocument.
    - Uses HybridChunker (which itself builds on HierarchicalChunker) for final chunks.
    """

    def __init__(
        self,
        tokenizer_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        use_mistral_ocr: bool = False,
        mistral_ocr_client: Optional[MistralOCRClient] = None,
    ):
        # Docling converter (PDF / DOCX / Markdown -> DoclingDocument) 
        self.converter = DocumentConverter()

        # Tokenizer for HybridChunker (HuggingFace-based) 
        self.tokenizer = HuggingFaceTokenizer.from_pretrained(
            model_name=tokenizer_model,
            max_tokens=max_tokens,
        )

        # HybridChunker does the real token-aware splitting+merging
        self.hybrid_chunker = HybridChunker(
            tokenizer=self.tokenizer,
            # merge_peers=True means undersized siblings with same headings are merged. 
            merge_peers=True,
        )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens  # if you want to use overlap, you can pass it through options

        self.use_mistral_ocr = use_mistral_ocr
        self.mistral_ocr_client = mistral_ocr_client or MistralOCRClient()

    # --- Internal helpers -------------------------------------------------

    def _docling_from_pdf(self, source: str) -> DoclingDocument:
        """
        Direct PDF -> DoclingDocument using Docling's own PDF pipeline.
        """
        conv_result = self.converter.convert(source=source)
        return conv_result.document

    def _docling_from_mistral_ocr(self, source: str) -> DoclingDocument:
        """
        PDF -> Mistral OCR (Markdown) -> temp .md -> DoclingDocument.

        Note: Docling can parse Markdown as an input format and construct
        a DoclingDocument. 
        """
        markdown = self.mistral_ocr_client.pdf_to_markdown(source)

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp.write(markdown)
            tmp_path = tmp.name

        try:
            conv_result = self.converter.convert(source=tmp_path)
            return conv_result.document
        finally:
            # For a production pipeline you might want to keep these or write into
            # an object store instead; for now we clean up.
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _docling_document(self, source: str):
        if self.use_mistral_ocr:
            return self._docling_from_mistral_ocr(source)
        else:
            return self._docling_from_pdf(source)

    # --- Public API -------------------------------------------------------

    def chunk_document(self, source: str, doc_id: str) -> List[DocChunk]:
        """
        Convert a PDF / URL / Markdown file into token-bounded, hierarchy-aware chunks.

        - `source` can be a local path or URL.
        - `doc_id` is your own ID (for joining with metadata / DB).
        """
        dl_doc = self._docling_document(source)

        # NOTE: we don't directly use HierarchicalChunker but we can also do:
        #   structural_chunks = list(HierarchicalChunker().chunk(dl_doc))
        # HybridChunker internally runs a similar hierarchical pass
        # and applies sliding-window + semantic splitting + merging on top.

        # Token-aware hybrid chunking
        chunks: List[DocChunk] = []
        for i, chunk in enumerate(self.hybrid_chunker.chunk(dl_doc)):
            # cast is needed bcos HybridChunker wrongly annotates output type as BaseChunk 
            # when it actually returns DocChunk
            if not isinstance(chunk, DoclingDocChunk):
                raise TypeError(f"Expected DoclingDocChunk, got {type(chunk)}")
            headings = chunk.meta.headings or []

            # TODO: check how to retrieve page_no. it does NOT exist in origin.
            # origin = chunk.meta.origin
            # page_no = origin.page_no

            chunks.append(
                DocChunk(
                    id=f"{doc_id}_{i}",
                    doc_id=doc_id,
                    text=chunk.text,
                    # page_no=page_no,
                    headings=headings,
                    source=source,
                )
            )

        return chunks

    def iter_chunk_documents(self, sources: Iterable[str]) -> Iterable[DocChunk]:
        """
        Generator variant: stream chunks across many documents to avoid
        holding everything in memory for extremely large PDFs.
        """
        for doc_idx, src in enumerate(sources):
            doc_id = f"doc_{doc_idx}"
            for chunk in self.chunk_document(src, doc_id):
                yield chunk

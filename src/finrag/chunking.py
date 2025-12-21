import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Optional

from docling.document_converter import DocumentConverter
from docling_core.transforms.chunker.doc_chunk import DocChunk as DoclingDocChunk
from docling_core.transforms.chunker.hybrid_chunker import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import HuggingFaceTokenizer
from docling_core.types.doc.document import DoclingDocument

from finrag.dataclasses import DocChunk

if TYPE_CHECKING:
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
        mistral_ocr_client: Optional["MistralOCRClient"] = None,
    ):
        # Docling converter (PDF / DOCX / Markdown -> DoclingDocument)
        self.converter = DocumentConverter()

        # Tokenizer for HybridChunker (HuggingFace-based)
        self.tokenizer = HuggingFaceTokenizer.from_pretrained(model_name=tokenizer_model, max_tokens=max_tokens)

        # HybridChunker does the real token-aware splitting+merging
        self.hybrid_chunker = HybridChunker(
            tokenizer=self.tokenizer,
            # merge_peers=True means undersized siblings with same headings are merged.
            merge_peers=True,
        )

        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens

        self.use_mistral_ocr = use_mistral_ocr
        self.mistral_ocr_client = mistral_ocr_client

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

        Requires that `self.mistral_ocr_client` is set.
        """
        if self.mistral_ocr_client is None:
            raise ValueError("Mistral OCR client is not set for OCR-based chunking.")

        markdown = self.mistral_ocr_client.pdf_to_markdown(source)

        with tempfile.NamedTemporaryFile(suffix=".md", delete=False, mode="w", encoding="utf-8") as tmp:
            tmp.write(markdown)
            tmp_path = tmp.name

        try:
            conv_result = self.converter.convert(source=tmp_path)
            return conv_result.document
        finally:
            # For a production pipeline we might want to keep these, or
            # write into an object store instead.
            # for now we clean up.
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    def _docling_document(self, source: str):
        if self.use_mistral_ocr:
            return self._docling_from_mistral_ocr(source)
        else:
            return self._docling_from_pdf(source)

    def _extract_page_numbers(self, chunk: DoclingDocChunk) -> list[int]:
        """
        Gather the unique page numbers that contributed to this chunk.

        Docling exposes provenance information (DocItem.prov) for every
        structural element; each provenance entry carries a `page_no`.
        """
        page_numbers: set[int] = set()
        doc_items = chunk.meta.doc_items
        for doc_item in doc_items:
            provenance = doc_item.prov
            for prov in provenance:
                page_no = prov.page_no
                page_numbers.add(page_no)
        return sorted(page_numbers)

    # --- Public API -------------------------------------------------------

    def chunk_document(self, source: str, doc_id: str) -> list[DocChunk]:
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
        chunks: list[DocChunk] = []
        for i, chunk in enumerate(self.hybrid_chunker.chunk(dl_doc)):
            # cast is needed bcos HybridChunker wrongly annotates output type as BaseChunk
            # when it actually returns DocChunk
            if not isinstance(chunk, DoclingDocChunk):
                raise TypeError(f"Expected DoclingDocChunk, got {type(chunk)}")
            headings = chunk.meta.headings or []
            page_numbers = self._extract_page_numbers(chunk)
            page_no = page_numbers[0] if page_numbers else None

            chunks.append(
                DocChunk(
                    id=f"{doc_id}_{i}",
                    doc_id=doc_id,
                    text=chunk.text,
                    page_no=page_no,
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


class MarkdownTablePreservingChunker:
    """
    Chunk a Markdown file without re-parsing tables.

    Why:
      - Docling's Markdown ingestion may linearize tables into repetitive prose.
      - For RAG over SEC filings, keeping Markdown tables as-is is often better.

    Features:
      - Hierarchy-aware headings stack (`#`..`######`) stored in `DocChunk.headings`.
      - Table blocks are emitted as their own chunk(s), preserving the original pipe layout.
      - Optional page number inference from an accompanying `metadata.json` (table_of_contents.page_id).
      - Optional page markers in Markdown (`<span id="page-5-1"></span>`).
    """

    _HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)(\s+#+\s*)?$")
    _PAGE_SPAN_RE = re.compile(r'<span\s+id="page-(\d+)-(\d+)"\s*></span>', re.IGNORECASE)
    _TABLE_SEPARATOR_RE = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")

    def __init__(self, *, max_tokens: int = 512, overlap_tokens: int = 64, split_tables: bool = True):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.split_tables = split_tables

    # --- Metadata helpers -------------------------------------------------

    @staticmethod
    def _normalize_heading(text: str) -> str:
        text = re.sub(r"</?[^>]+>", "", text)  # strip HTML tags
        text = re.sub(r"[`*_]+", "", text)  # strip common markdown emphasis
        text = re.sub(r"\s+", " ", text).strip()
        text = text.strip(" .:-")
        return text.casefold()

    @classmethod
    def _load_toc_page_map(cls, metadata_json_path: str | None) -> dict[str, int]:
        if not metadata_json_path:
            return {}
        path = Path(metadata_json_path)
        if not path.exists():
            return {}
        with path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        toc = meta.get("table_of_contents") or []
        out: dict[str, int] = {}
        for entry in toc:
            if not isinstance(entry, dict):
                continue
            title = entry.get("title")
            page_id = entry.get("page_id")
            if not isinstance(title, str):
                continue
            if page_id is None:
                continue
            try:
                page_no = int(page_id)
            except (TypeError, ValueError):
                continue
            key = cls._normalize_heading(title)
            if key and key not in out:
                out[key] = page_no
        return out

    # --- Markdown parsing -------------------------------------------------

    def _count_tokens(self, text: str) -> int:
        # Conservative + dependency-free: approximate tokens by whitespace-separated terms.
        return len(re.findall(r"\S+", text))

    def _tail_overlap(self, text: str) -> str:
        if self.overlap_tokens <= 0:
            return ""
        words = re.findall(r"\S+", text)
        if not words:
            return ""
        return " ".join(words[-self.overlap_tokens :])

    @classmethod
    def _is_table_start(cls, lines: list[str], idx: int) -> bool:
        if idx + 1 >= len(lines):
            return False
        head = lines[idx]
        sep = lines[idx + 1]
        if "|" not in head:
            return False
        return bool(cls._TABLE_SEPARATOR_RE.match(sep))

    def _iter_blocks(self, markdown: str) -> Iterable[dict[str, Any]]:
        lines = markdown.splitlines()
        i = 0
        while i < len(lines):
            raw = lines[i]
            line = raw.rstrip("\n")

            # Page marker (may be inline with other text)
            m_page = self._PAGE_SPAN_RE.search(line)
            if m_page:
                page_no = int(m_page.group(1))
                rest = self._PAGE_SPAN_RE.sub("", line).strip()
                yield {"kind": "page", "page_no": page_no}
                if rest:
                    yield {"kind": "text", "text": rest}
                i += 1
                continue

            # Heading
            m_head = self._HEADING_RE.match(line.strip())
            if m_head:
                level = len(m_head.group(1))
                title = m_head.group(2).strip()
                yield {"kind": "heading", "level": level, "title": title}
                i += 1
                continue

            # Table
            if self._is_table_start(lines, i):
                start = i
                i += 2
                while i < len(lines) and lines[i].strip() and ("|" in lines[i]):
                    i += 1
                table_text = "\n".join(lines[start:i]).strip()
                yield {"kind": "table", "text": table_text}
                continue

            # Blank
            if not line.strip():
                i += 1
                continue

            # Paragraph-ish block until blank / heading / table / page marker
            start = i
            i += 1
            while i < len(lines):
                nxt = lines[i]
                if not nxt.strip():
                    break
                if self._PAGE_SPAN_RE.search(nxt):
                    break
                if self._HEADING_RE.match(nxt.strip()):
                    break
                if self._is_table_start(lines, i):
                    break
                i += 1
            text = "\n".join(lines[start:i]).strip()
            if text:
                yield {"kind": "text", "text": text}

    def _split_table_if_needed(self, table_text: str) -> list[str]:
        if not self.split_tables:
            return [table_text]
        lines = table_text.splitlines()
        if len(lines) <= 2:
            return [table_text]
        header = lines[0]
        sep = lines[1]
        rows = lines[2:]

        # If it fits, keep it whole.
        if self._count_tokens(table_text) <= self.max_tokens:
            return [table_text]

        # Otherwise: split by rows, repeating header + separator.
        chunks: list[str] = []
        cur_rows: list[str] = []
        cur_tokens = self._count_tokens("\n".join([header, sep]))
        for row in rows:
            row_tokens = self._count_tokens(row)
            if cur_rows and cur_tokens + row_tokens > self.max_tokens:
                chunks.append("\n".join([header, sep, *cur_rows]).strip())
                cur_rows = []
                cur_tokens = self._count_tokens("\n".join([header, sep]))
            cur_rows.append(row)
            cur_tokens += row_tokens
        if cur_rows:
            chunks.append("\n".join([header, sep, *cur_rows]).strip())
        return chunks

    # --- Public API -------------------------------------------------------

    def chunk_document(self, source: str, doc_id: str, *, metadata_json_path: str | None = None) -> list[DocChunk]:
        md_path = Path(source)
        markdown = md_path.read_text(encoding="utf-8")

        toc_page_map = self._load_toc_page_map(metadata_json_path)
        headings_stack: list[str] = []
        current_page: int | None = None

        chunks: list[DocChunk] = []
        buf_parts: list[str] = []
        buf_tokens = 0
        carry = ""

        def flush_buffer():
            nonlocal buf_parts, buf_tokens, carry
            if not buf_parts:
                return
            text = "\n\n".join(p.strip() for p in buf_parts if p.strip()).strip()
            if not text:
                buf_parts = []
                buf_tokens = 0
                carry = ""
                return
            chunks.append(
                DocChunk(
                    id=f"{doc_id}_{len(chunks)}",
                    doc_id=doc_id,
                    text=text,
                    page_no=current_page,
                    headings=list(headings_stack),
                    source=str(md_path),
                    metadata={"source_format": "markdown", "block_type": "text"},
                )
            )
            carry = self._tail_overlap(text)
            buf_parts = []
            buf_tokens = 0

        for block in self._iter_blocks(markdown):
            kind = block["kind"]

            if kind == "page":
                current_page = int(block["page_no"])
                continue

            if kind == "heading":
                flush_buffer()
                level = int(block["level"])
                title = str(block["title"]).strip()
                title_clean = re.sub(r"[`*_]+", "", title).strip()
                if level <= 0:
                    level = 1
                while len(headings_stack) >= level:
                    headings_stack.pop()
                headings_stack.append(title_clean)

                norm = self._normalize_heading(title_clean)
                if norm in toc_page_map:
                    current_page = toc_page_map[norm]
                continue

            if kind == "table":
                flush_buffer()
                table_text = str(block["text"]).strip()
                for table_part in self._split_table_if_needed(table_text):
                    chunks.append(
                        DocChunk(
                            id=f"{doc_id}_{len(chunks)}",
                            doc_id=doc_id,
                            text=table_part,
                            page_no=current_page,
                            headings=list(headings_stack),
                            source=str(md_path),
                            metadata={"source_format": "markdown", "block_type": "table"},
                        )
                    )
                carry = ""  # don't overlap tables into subsequent chunks
                continue

            if kind == "text":
                text = str(block["text"]).strip()
                if not text:
                    continue
                if not buf_parts and carry:
                    buf_parts.append(carry)
                    buf_tokens += self._count_tokens(carry)
                    carry = ""
                text_tokens = self._count_tokens(text)
                if buf_parts and (buf_tokens + text_tokens) > self.max_tokens:
                    flush_buffer()
                    if carry:
                        buf_parts.append(carry)
                        buf_tokens += self._count_tokens(carry)
                        carry = ""
                buf_parts.append(text)
                buf_tokens += text_tokens
                continue

        flush_buffer()
        return chunks

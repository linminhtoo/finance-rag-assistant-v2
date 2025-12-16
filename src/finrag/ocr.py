import base64
import os
from typing import List, Optional

from mistralai import Mistral
from mistralai.models.documenturlchunk import DocumentURLChunkTypedDict

DEFAULT_PAGE_SEP = "\n\n<!-- PAGE BREAK -->\n\n"


class MistralOCRClient:
    """
    Thin wrapper around Mistral OCR that turns a PDF into a single Markdown string.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "mistral-ocr-latest",
    ):
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        if not self.api_key:
            raise ValueError("MISTRAL_API_KEY is not set")
        self.client = Mistral(api_key=self.api_key)
        self.model = model

    def _document_payload_from_source(self, source: str) -> DocumentURLChunkTypedDict:
        # Heuristic: if it looks like a URL, treat it as document_url
        if source.startswith("http://") or source.startswith("https://"):
            return {"type": "document_url", "document_url": source}

        # Otherwise, assume local file path. Parse base64
        with open(source, "rb") as f:
            data = base64.b64encode(f.read()).decode("utf-8")
        data_url = f"data:application/pdf;base64,{data}"
        return {"type": "document_url", "document_url": data_url}

    def pdf_to_markdown_pages(self, source: str) -> List[str]:
        document = self._document_payload_from_source(source)

        ocr_response = self.client.ocr.process(
            model=self.model,
            document=document,
            include_image_base64=False,  # text-only for RAG
        )
        return [page.markdown for page in ocr_response.pages]

    def pdf_to_markdown(self, source: str, 
                        page_separator: str = DEFAULT_PAGE_SEP) -> str:
        pages = self.pdf_to_markdown_pages(source)
        return page_separator.join(pages)

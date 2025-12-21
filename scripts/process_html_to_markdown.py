import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import local
from typing import Annotated, List

import PIL
import weasyprint
from loguru import logger
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.logger import get_logger
from marker.models import create_model_dict
# from marker.renderers.html import HTMLRenderer
# from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownRenderer
from marker.renderers.markdown import MarkdownOutput
from marker.schema.blocks import Block
from marker.services.openai import OpenAIService as BaseOpenAIService
from openai import APITimeoutError, RateLimitError
from pydantic import BaseModel
from tqdm import tqdm

marker_logger = get_logger()


class CustomOpenAIService(BaseOpenAIService):
    """Drop-in replacement for marker's OpenAIService with additional parameters."""

    openai_temperature: Annotated[float, "The sampling temperature to use for OpenAI-like services."] = 0.7
    openai_system_prompt: Annotated[str, "Optional system prompt to prepend before the user message."] = ""

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,  # type: ignore[type-arg]
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,
        timeout: int | None = None,
    ):
        if max_retries is None:
            max_retries = self.max_retries

        if timeout is None:
            timeout = self.timeout

        client = self.get_client()
        image_data = self.format_image_for_llm(image)

        messages = []
        if self.openai_system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.openai_system_prompt}]})
        messages.append({"role": "user", "content": [*image_data, {"type": "text", "text": prompt}]})

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                response = client.beta.chat.completions.parse(
                    extra_headers={"X-Title": "Marker", "HTTP-Referer": "https://github.com/datalab-to/marker"},
                    model=self.openai_model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                    temperature=self.openai_temperature,
                )
                response_text = response.choices[0].message.content
                if response_text is None:
                    raise ValueError("LLM response missing content")
                if response.usage is None:
                    raise ValueError("LLM response missing usage information")
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                return json.loads(response_text)
            except (APITimeoutError, RateLimitError) as e:
                if tries == total_tries:
                    marker_logger.error(
                        "Rate limit error: %s. Max retries reached. Giving up. (Attempt %s/%s)", e, tries, total_tries
                    )
                    break
                wait_time = tries * self.retry_wait_time
                marker_logger.warning(
                    "Rate limit error: %s. Retrying in %s seconds... (Attempt %s/%s)", e, wait_time, tries, total_tries
                )
                time.sleep(wait_time)
            except Exception as e:  # noqa: BLE001 - best effort logging
                marker_logger.error("OpenAI inference failed: %s", e)
                break

        return {}


W_IN = 1400 / 96
H_IN = 1350 / 96

# first part sets landscape orientation
# second part handles page-breaks
CSS_STYLESHEET = weasyprint.CSS(
    string=f"""
@page {{ size: {W_IN:.3f}in {H_IN:.3f}in; margin: 4mm; }}
html, body {{ margin: 0; }}
"""
    + """
/* Turn EDGAR <hr page-break-after> into hard page boundaries */
hr[style*="page-break-after"] {{
  break-after: page;
  page-break-after: always;
  border: 0; height: 0; margin: 0; padding: 0;
}}

/* IMPORTANT: allow multi-page tables */
table {{
  break-inside: auto;          /* allow splitting */
  page-break-inside: auto;     /* legacy */
}}

/* Optional: keep individual rows together (usually OK) */
tr {{
  break-inside: avoid;
  page-break-inside: avoid;
}}

/* Optional but helpful: repeat header row on each page */
thead {{ display: table-header-group; }}
tfoot {{ display: table-footer-group; }}
"""
)


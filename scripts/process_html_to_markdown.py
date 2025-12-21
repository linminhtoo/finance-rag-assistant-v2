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


@dataclass
class Args:
    html_dir: str
    output_dir: str
    openai_base_url: str
    openai_model: str
    openai_system_prompt: str
    openai_temperature: float
    timeout: int
    max_retries: int
    max_concurrency: int
    workers: int


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files to process (recursively).")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root output directory (writes `pdf/` and `markdown/` subfolders).",
    )
    parser.add_argument(
        "--openai-base-url", required=True, help="Base URL for OpenAI-compatible API (no trailing slash)."
    )
    parser.add_argument("--openai-model", required=True, help="Model name for OpenAI-compatible API.")
    parser.add_argument("--openai-system-prompt", default="", help="Optional system prompt for the LLM.")
    parser.add_argument("--openai-temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout in seconds.")
    parser.add_argument(
        "--max-retries", type=int, default=2, help="Max retries for transient errors (rate limit/timeout)."
    )
    parser.add_argument("--max-concurrency", type=int, default=3, help="Max parallel in-flight LLM requests.")
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers over HTML files (effective LLM concurrency ~= workers * max_concurrency).",
    )
    args = parser.parse_args()
    return Args(
        html_dir=args.html_dir,
        output_dir=args.output_dir,
        openai_base_url=args.openai_base_url,
        openai_model=args.openai_model,
        openai_system_prompt=args.openai_system_prompt,
        openai_temperature=args.openai_temperature,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_concurrency=args.max_concurrency,
        workers=args.workers,
    )


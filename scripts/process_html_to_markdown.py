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


def main():
    args = parse_args()

    html_dir = Path(args.html_dir).expanduser().resolve()
    if not html_dir.exists() or not html_dir.is_dir():
        raise RuntimeError(f"--html-dir must be an existing directory: {html_dir}")

    output_root = Path(args.output_dir).expanduser().resolve()
    pdf_root = output_root / "intermediate_pdf"
    md_root = output_root / "processed_markdown"
    debug_root = output_root / "debug"
    pdf_root.mkdir(parents=True, exist_ok=True)
    md_root.mkdir(parents=True, exist_ok=True)
    debug_root.mkdir(parents=True, exist_ok=True)

    openai_api_key = os.environ.get("OPENAI_API_KEY")
    if not openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY before running so the OpenAI llm_service can authenticate.")

    config = {
        "output_format": "markdown",
        "use_llm": True,
        "llm_service": "scripts.process_html_to_markdown.CustomOpenAIService",
        "openai_api_key": openai_api_key,
        "openai_base_url": args.openai_base_url,
        "openai_model": args.openai_model,
        "CustomOpenAIService_openai_temperature": args.openai_temperature,
        "CustomOpenAIService_openai_system_prompt": args.openai_system_prompt,
        "CustomOpenAIService_timeout": args.timeout,
        "CustomOpenAIService_max_retries": args.max_retries,
        # Marker calls this "max_concurrency" (default 3). This controls parallel in-flight
        # LLM requests (not "batching" into a single request).
        "max_concurrency": args.max_concurrency,
        "LLMTableProcessor_max_concurrency": args.max_concurrency,
        "disable_image_extraction": True,
        "force_ocr": False,
    }
    config_parser = ConfigParser(config)
    renderer_config = config_parser.generate_config_dict()

    converter = PdfConverter(
        config=renderer_config,
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )

    markdown_renderer = MarkdownRenderer(renderer_config)
    # html_renderer = HTMLRenderer(renderer_config)
    # json_renderer = JSONRenderer({**renderer_config, "extract_images": False})

    html_paths = sorted(list(html_dir.rglob("*.html")) + list(html_dir.rglob("*.htm")))
    if not html_paths:
        raise RuntimeError(f"No HTML files found under: {html_dir}")
    logger.info(f"Found {len(html_paths)} HTML files to process under {html_dir}")

    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1")

    thread_state = local()

    def get_pipeline() -> tuple[PdfConverter, MarkdownRenderer]:
        pipeline = getattr(thread_state, "pipeline", None)
        if pipeline is not None:
            return pipeline

        thread_converter = PdfConverter(
            config=renderer_config,
            artifact_dict=create_model_dict(),
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=config_parser.get_llm_service(),
        )
        thread_markdown_renderer = MarkdownRenderer(renderer_config)
        thread_state.pipeline = (thread_converter, thread_markdown_renderer)
        return thread_state.pipeline

    def process_one(html_file_path: Path) -> tuple[Path, int, int]:
        thread_converter, thread_markdown_renderer = (
            (converter, markdown_renderer) if args.workers == 1 else get_pipeline()
        )

        rel_path = html_file_path.relative_to(html_dir)
        pdf_path = (pdf_root / rel_path).with_suffix(".pdf")
        md_path = (md_root / rel_path).with_suffix(".md")
        pdf_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Processing {html_file_path} -> {pdf_path}, {md_path}")

        debug_dir = debug_root / rel_path.parent / rel_path.stem
        debug_dir.mkdir(parents=True, exist_ok=True)

        # 1. convert HTML to PDF
        source_url = f"file://{html_file_path.absolute()}"
        weasyprint.HTML(source_url).write_pdf(str(pdf_path), stylesheets=[CSS_STYLESHEET])
        logger.success(f"Wrote intermediate PDF to {pdf_path}")

        # 2. convert PDF to document (so we can save multiple renderings/artifacts)
        # each 10Q usually requires ~40 LLM calls for LLMTableProcessor
        document = thread_converter.build_document(str(pdf_path))
        logger.success(f"Converted PDF to Marker document for {rel_path}")

        rendered = thread_markdown_renderer(document)
        logger.success(f"Rendered markdown for {rel_path}")
        
        llm_err_total, page_cnt = count_llm_errors(rendered)
        logger.info(f"{rel_path}: {llm_err_total=} out of {page_cnt=}")

        with open(md_path, "w") as f:
            f.write(rendered.markdown)

        with open(debug_dir / "metadata.json", "w") as f:
            json.dump(rendered.metadata, f, indent=2)

        # also save HTML and JSON renderings for debugging
        # html_out = html_renderer(document)
        # with open(debug_dir / "document.html", "w") as f:
        #     f.write(html_out.html)

        # json_out = json_renderer(document)
        # with open(debug_dir / "document.json", "w") as f:
        #     f.write(json_out.model_dump_json(indent=2))

        with open(debug_dir / "run_info.json", "w") as f:
            json.dump(
                {
                    "input_html": str(html_file_path),
                    "output_pdf": str(pdf_path),
                    "output_markdown": str(md_path),
                    "args": {
                        "html_dir": args.html_dir,
                        "output_dir": args.output_dir,
                        "openai_base_url": args.openai_base_url,
                        "openai_model": args.openai_model,
                        "openai_system_prompt": args.openai_system_prompt,
                        "openai_temperature": args.openai_temperature,
                        "timeout": args.timeout,
                        "max_retries": args.max_retries,
                        "max_concurrency": args.max_concurrency,
                        "workers": args.workers,
                    },
                    "config": renderer_config,
                    "llm_error_count": llm_err_total,
                    "page_count": page_cnt,
                    "llm_error_ratio": llm_err_total / page_cnt if page_cnt > 0 else None,
                },
                f,
                indent=2,
            )
        
        logger.success(f"Finished processing {rel_path}")
        return rel_path, llm_err_total, page_cnt

    if args.workers == 1:
        for html_file_path in tqdm(
            html_paths,
            desc="Processing HTML files",
            unit="file",
        ):
            process_one(html_file_path)
        return

    failures: list[Path] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_one, p): p for p in html_paths}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing HTML files", unit="file"):
            html_file_path = futures[future]
            try:
                rel_path, llm_err_total, page_cnt = future.result()
                logger.info(f"{rel_path}: {llm_err_total=} out of {page_cnt=}")
            except Exception:
                failures.append(html_file_path)
                logger.exception(f"Failed processing: {html_file_path}")

    if failures:
        raise RuntimeError(f"Failed to process {len(failures)} files; see logs for details.")


if __name__ == "__main__":
    main()

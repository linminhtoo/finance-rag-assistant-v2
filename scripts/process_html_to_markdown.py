import argparse
import json
import multiprocessing as mp
import os

# import random
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, List

import PIL
import weasyprint
from langsmith.wrappers import wrap_openai
from loguru import logger
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.logger import get_logger
from marker.models import create_model_dict

# from marker.renderers.html import HTMLRenderer
# from marker.renderers.json import JSONRenderer
from marker.renderers.markdown import MarkdownOutput, MarkdownRenderer
from marker.schema.blocks import Block
from marker.services.openai import OpenAIService as BaseOpenAIService
from openai import APITimeoutError, RateLimitError
from pydantic import BaseModel
from tqdm import tqdm

marker_logger = get_logger()

_worker_converter: PdfConverter | None = None
_worker_markdown_renderer: MarkdownRenderer | None = None
_worker_renderer_config: dict | None = None


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

        client = wrap_openai(self.get_client())
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
    gpu_ids: str
    year_cutoff: int | None

    def to_dict(self) -> dict:
        return asdict(self)


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files to process (recursively).")
    parser.add_argument(
        "--output-dir", default="outputs", help="Root output directory (writes `pdf/` and `markdown/` subfolders)."
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
    parser.add_argument(
        "--year-cutoff",
        type=int,
        default=None,
        help="Only process filings from this year (YYYY) based on the filename date.",
    )
    parser.add_argument(
        "--gpu-ids", default="", help="Optional comma-separated GPU ids to round-robin across workers (e.g., '0,1')."
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
        gpu_ids=args.gpu_ids,
        year_cutoff=args.year_cutoff,
    )


def count_llm_errors(rendered: MarkdownOutput) -> tuple[int, int]:
    llm_err_total = 0
    page_cnt = 0
    for page_meta in rendered.metadata["page_stats"]:
        block_meta = page_meta["block_metadata"]
        llm_err_total += block_meta["llm_error_count"]
        page_cnt += 1
    return llm_err_total, page_cnt


def get_worker_index() -> int:
    identity = mp.current_process()._identity
    if identity:
        return identity[0]
    match = re.search(r"(\d+)$", mp.current_process().name)
    if match:
        return int(match.group(1))
    return 1


def extract_year_from_filename(html_path: Path) -> int | None:
    match = re.search(r"(\d{4})-\d{2}-\d{2}$", html_path.stem)
    if not match:
        return None
    return int(match.group(1))


def init_worker(config: dict, gpu_ids: list[str]) -> None:    
    if gpu_ids:
        worker_index = get_worker_index()
        gpu_id = gpu_ids[(worker_index - 1) % len(gpu_ids)]
        # NOTE: setting env var is too late here, so we must use torch.cuda.set_device()
        # os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.set_device(int(gpu_id))
        except Exception as exc:  # noqa: BLE001 - best effort logging
            logger.warning(f"Unable to pin CUDA device for worker {worker_index}: {exc}")
        logger.info(f"Worker {worker_index} using GPU {gpu_id}")
    
    global _worker_converter, _worker_markdown_renderer, _worker_renderer_config
    config_parser = ConfigParser(config)
    renderer_config = config_parser.generate_config_dict()
    _worker_converter = PdfConverter(
        config=renderer_config,
        artifact_dict=create_model_dict(),
        processor_list=config_parser.get_processors(),
        renderer=config_parser.get_renderer(),
        llm_service=config_parser.get_llm_service(),
    )
    _worker_markdown_renderer = MarkdownRenderer(renderer_config)
    _worker_renderer_config = renderer_config


def get_worker_pipeline() -> tuple[PdfConverter, MarkdownRenderer, dict]:
    if _worker_converter is None or _worker_markdown_renderer is None or _worker_renderer_config is None:
        raise RuntimeError("Worker pipeline has not been initialized")
    return _worker_converter, _worker_markdown_renderer, _worker_renderer_config


def process_one(
    html_file_path: Path, html_dir: Path, pdf_root: Path, md_root: Path, debug_root: Path, args_dict: dict
) -> None:
    # hack: sleep a bit to stagger LLM requests
    # time.sleep(random.uniform(10, 30))

    worker_converter, worker_markdown_renderer, renderer_config = get_worker_pipeline()

    rel_path = html_file_path.relative_to(html_dir)
    pdf_path = (pdf_root / rel_path).with_suffix(".pdf")
    md_path = (md_root / rel_path).with_suffix(".md")
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Processing {html_file_path} -> {pdf_path}, {md_path}")

    debug_dir = debug_root / rel_path.parent / rel_path.stem
    debug_dir.mkdir(parents=True, exist_ok=True)

    # if all output paths already exist, skip
    if pdf_path.exists() and md_path.exists():
        logger.info(f"Outputs already exist; skipping: {rel_path}")
        return

    # 1. convert HTML to PDF
    source_url = f"file://{html_file_path.absolute()}"
    weasyprint.HTML(source_url).write_pdf(str(pdf_path), stylesheets=[CSS_STYLESHEET])
    logger.success(f"Wrote intermediate PDF to {pdf_path}")

    # 2. convert PDF to document (so we can save multiple renderings/artifacts)
    # each 10Q usually requires ~40 LLM calls for LLMTableProcessor
    document = worker_converter.build_document(str(pdf_path))
    logger.success(f"Converted PDF to Marker document for {rel_path}")

    rendered = worker_markdown_renderer(document)
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
                "args": args_dict,
                "config": renderer_config,
                "llm_error_count": llm_err_total,
                "page_count": page_cnt,
                "llm_error_ratio": llm_err_total / page_cnt if page_cnt > 0 else None,
            },
            f,
            indent=2,
        )

    logger.success(f"Finished processing {rel_path}")
    return


def main():
    args = parse_args()
    logger.info(f"Starting processing with args: {args}")

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

    html_paths = sorted(list(html_dir.rglob("*.html")) + list(html_dir.rglob("*.htm")))
    if not html_paths:
        raise RuntimeError(f"No HTML files found under: {html_dir}")
    logger.info(f"Found {len(html_paths)} HTML files to process under {html_dir}")

    if args.year_cutoff is not None:
        filtered: list[Path] = []
        skipped_missing_date: list[Path] = []
        for html_path in html_paths:
            year = extract_year_from_filename(html_path)
            if year is None:
                skipped_missing_date.append(html_path)
                continue
            if year >= args.year_cutoff:
                filtered.append(html_path)
        if skipped_missing_date:
            logger.warning(
                "Skipped %s HTML files without a YYYY-MM-DD suffix in the filename.", len(skipped_missing_date)
            )
        html_paths = filtered
        if not html_paths:
            raise RuntimeError(f"No HTML files found for year >= {args.year_cutoff} under: {html_dir}")
        logger.info(f"Filtered to {len(html_paths)} HTML files for year >= {args.year_cutoff}")

    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1")

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

    if args.workers == 1:
        gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
        init_worker(config, gpu_ids)
        logger.info("Processing HTML files sequentially with a single worker")
        args_dict = args.to_dict()
        for html_file_path in tqdm(html_paths, desc="Processing HTML files", unit="file"):
            process_one(html_file_path, html_dir, pdf_root, md_root, debug_root, args_dict)
        return

    args_dict = args.to_dict()
    ctx = mp.get_context("spawn")
    logger.info(f"Processing HTML files in parallel with {args.workers} workers")
    failures: list[Path] = []
    gpu_ids = [gpu.strip() for gpu in args.gpu_ids.split(",") if gpu.strip()]
    with ProcessPoolExecutor(
        max_workers=args.workers, mp_context=ctx, initializer=init_worker, initargs=(config, gpu_ids)
    ) as executor:
        futures = {
            executor.submit(process_one, p, html_dir, pdf_root, md_root, debug_root, args_dict): p for p in html_paths
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing HTML files", unit="file"):
            html_file_path = futures[future]
            try:
                future.result()
            except Exception:
                failures.append(html_file_path)
                logger.exception(f"Failed processing: {html_file_path}")

    if failures:
        raise RuntimeError(f"Failed to process {len(failures)} files; see logs for details.")


if __name__ == "__main__":
    main()

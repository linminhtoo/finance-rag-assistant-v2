import argparse
import json
import multiprocessing as mp
import os

# import random
import re
import time
import uuid
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, List, cast

from dotenv import load_dotenv
import openai
import PIL
import weasyprint
from langsmith.wrappers import wrap_openai
from loguru import logger
from marker.config.parser import ConfigParser
from marker.converters.pdf import PdfConverter
from marker.logger import get_logger
from marker.models import create_model_dict

from marker.renderers.markdown import MarkdownOutput, MarkdownRenderer
from marker.schema.blocks import Block
from marker.services.openai import OpenAIService as BaseOpenAIService
from openai import APITimeoutError, RateLimitError
from PIL import ImageStat
from pydantic import BaseModel
from tqdm import tqdm

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

marker_logger = get_logger()

_worker_converter: PdfConverter | None = None
_worker_markdown_renderer: MarkdownRenderer | None = None
_worker_renderer_config: dict | None = None


class CustomOpenAIService(BaseOpenAIService):
    """Drop-in replacement for marker's OpenAIService with additional parameters."""

    openai_temperature: Annotated[float, "The sampling temperature to use for OpenAI-like services."] = 0.1
    openai_system_prompt: Annotated[str, "Optional system prompt to prepend before the user message."] = ""
    openai_timeout: Annotated[int, "Request timeout in seconds."] = 240
    openai_max_retries: Annotated[int, "Max retries for transient errors (rate limit/timeout)."] = 1
    schema_routes: Annotated[
        dict[str, dict],
        "Optional per-schema overrides keyed by `<module>.<SchemaName>` (e.g. "
        "`marker.processors.llm.llm_sectionheader.SectionHeaderSchema`). Supported keys include: "
        "`openai_base_url`, `openai_model`, `openai_timeout`, `openai_max_retries`, `openai_temperature`, "
        "`max_prompt_tokens`, `hf_model_id`, `max_image_long_side`, `skip_blank_images`, "
        "`blank_image_variance_threshold`.",
    ] = {}
    hf_model_id: Annotated[
        str,
        "Optional HuggingFace model/tokenizer id used to estimate prompt token counts before sending the request. "
        "Defaults to `openai_model` when empty.",
    ] = ""
    hf_trust_remote_code: Annotated[bool, "Pass trust_remote_code=True to transformers loaders."] = True
    hf_count_image_tokens: Annotated[
        bool, "When possible, include image tokens by using an AutoProcessor and decoding data URLs to PIL Images."
    ] = True
    log_prompt_token_count: Annotated[bool, "Log estimated prompt token counts before requesting the LLM."] = False
    max_prompt_tokens: Annotated[
        int, "If > 0, skip the LLM call when the estimated prompt tokens exceed this value (best-effort)."
    ] = 0
    max_image_long_side: Annotated[int, "If >0, downscale images so max(width,height) <= this before sending."] = 0
    skip_blank_images: Annotated[bool, "If true, skip LLM calls when all images look blank/uniform."] = False
    blank_image_variance_threshold: Annotated[
        float, "Grayscale variance threshold used to treat an image as blank (lower = stricter)."
    ] = 0.25

    def get_client(self, *, base_url: str | None = None, timeout: int | None = None) -> openai.OpenAI:
        base_url = (base_url or self.openai_base_url).strip()
        timeout = timeout if timeout is not None else self.openai_timeout
        # OpenAI Python SDK defaults to `max_retries=2` (3 total attempts), which can make a single
        # LangSmith-traced request appear to run up to ~3x longer than `timeout`. We implement our
        # own retry loop below, so disable SDK retries for predictable timing.
        return openai.OpenAI(api_key=self.openai_api_key, base_url=base_url, max_retries=0, timeout=timeout)

    @staticmethod
    def _schema_key(response_schema: type[BaseModel]) -> str:
        return f"{response_schema.__module__}.{response_schema.__name__}"

    def _route_for_schema(self, response_schema: type[BaseModel]) -> dict:
        schema_key = self._schema_key(response_schema)
        routes = self.schema_routes or {}
        return routes.get(schema_key, {})

    @staticmethod
    def _downscale_image(image: PIL.Image.Image, max_long_side: int) -> PIL.Image.Image:  # type: ignore[type-arg]
        if max_long_side <= 0:
            return image
        width, height = image.size
        if max(width, height) <= max_long_side:
            return image
        img_copy = image.copy()
        img_copy.thumbnail((max_long_side, max_long_side))
        return img_copy

    @staticmethod
    def _looks_blank(image: PIL.Image.Image, variance_threshold: float) -> bool:  # type: ignore[type-arg]
        try:
            stat = ImageStat.Stat(image.convert("L"))
            var = float(stat.var[0]) if stat.var else 0.0
            return var <= variance_threshold
        except Exception:  # noqa: BLE001 - best effort
            return False

    def _prepare_images(
        self,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,  # type: ignore[type-arg]
        *,
        max_long_side: int,
        skip_blank_images: bool,
        blank_variance_threshold: float,
    ) -> PIL.Image.Image | List[PIL.Image.Image] | None:  # type: ignore[type-arg]
        if not image:
            return None

        images: list[PIL.Image.Image] = image if isinstance(image, list) else [image]  # type: ignore[type-arg]
        processed = [self._downscale_image(img, max_long_side) for img in images]

        if skip_blank_images and processed:
            if all(self._looks_blank(img, blank_variance_threshold) for img in processed):
                return None

        if isinstance(image, list):
            return processed
        return processed[0]

    def __call__(
        self,
        prompt: str,
        image: PIL.Image.Image | List[PIL.Image.Image] | None,  # type: ignore[type-arg]
        block: Block | None,
        response_schema: type[BaseModel],
        max_retries: int | None = None,  # ignored
        timeout: int | None = None,  # ignored
    ):
        request_id = uuid.uuid4().hex
        start_time = time.perf_counter()

        had_image = bool(image)
        route = self._route_for_schema(response_schema)
        model = (route.get("openai_model") or self.openai_model).strip()
        base_url = (route.get("openai_base_url") or self.openai_base_url).strip()
        temperature = float(route.get("openai_temperature", self.openai_temperature))
        max_retries = int(route.get("openai_max_retries", self.openai_max_retries))
        timeout = int(route.get("openai_timeout", self.openai_timeout))
        max_prompt_tokens = int(route.get("max_prompt_tokens", self.max_prompt_tokens))
        hf_model_id_override = (route.get("hf_model_id") or "").strip()
        max_image_long_side = int(route.get("max_image_long_side", self.max_image_long_side))
        skip_blank_images = bool(route.get("skip_blank_images", self.skip_blank_images))
        blank_variance_threshold = float(
            route.get("blank_image_variance_threshold", self.blank_image_variance_threshold)
        )

        logger.info(
            f"CustomOpenAIService[{request_id}] start timeout={timeout}s max_retries={max_retries} model={model} schema={self._schema_key(response_schema)}"
        )

        image = self._prepare_images(
            image,
            max_long_side=max_image_long_side,
            skip_blank_images=skip_blank_images,
            blank_variance_threshold=blank_variance_threshold,
        )
        if skip_blank_images and had_image and image is None:
            marker_logger.warning("CustomOpenAIService[%s] blank_image_detected; skipping LLM call", request_id)
            return {}

        client = self.get_client(base_url=base_url, timeout=timeout)
        if os.environ.get("LANGSMITH_TRACING", "false").lower() == "true":
            client = wrap_openai(client)

        image_data = self.format_image_for_llm(image)

        messages = []
        if self.openai_system_prompt:
            messages.append({"role": "system", "content": [{"type": "text", "text": self.openai_system_prompt}]})
        messages.append({"role": "user", "content": [*image_data, {"type": "text", "text": prompt}]})

        if self.log_prompt_token_count or max_prompt_tokens > 0:
            try:
                from finrag.hf_token_count import count_tokens_openai_messages

                hf_model_id = (hf_model_id_override or self.hf_model_id or model).strip()
                token_info = count_tokens_openai_messages(
                    messages,
                    hf_model_id,
                    trust_remote_code=self.hf_trust_remote_code,
                    count_images=self.hf_count_image_tokens,
                )
                marker_logger.info(
                    "CustomOpenAIService[%s] prompt_tokens_est total=%s text=%s image=%s method=%s max_length=%s warnings=%s",
                    request_id,
                    token_info.total_tokens,
                    token_info.text_tokens,
                    token_info.image_tokens,
                    token_info.method,
                    token_info.model_max_length,
                    token_info.warnings if token_info.warnings else None,
                )
                # TODO: we can't do block.update_metadata bcos BlockMetadata doesn't have these new fields
                # we will need to somehow override BlockMetadata to add these fields
                if max_prompt_tokens > 0 and token_info.total_tokens > max_prompt_tokens:
                    marker_logger.warning(
                        "CustomOpenAIService[%s] prompt_too_long total=%s max=%s; skipping LLM call",
                        request_id,
                        token_info.total_tokens,
                        max_prompt_tokens,
                    )
                    return {}
            except Exception as exc:  # noqa: BLE001 - best effort
                marker_logger.warning("CustomOpenAIService[%s] token_count_failed err=%s", request_id, exc)

        total_tries = max_retries + 1
        for tries in range(1, total_tries + 1):
            try:
                # if we use client.responses.parse(), we will get a response_id
                # that we can POST an abort to vLLM if vLLM is not cancelling on its own
                # however, based on testing, vLLM does cancel requests immediately once python process is killed
                # (not just ctrl+C but kill -9)
                response = client.chat.completions.parse(
                    model=model,
                    messages=messages,
                    timeout=timeout,
                    response_format=response_schema,
                    temperature=temperature,
                    # TODO: try guided JSON to further robustify the structured output
                    # need to double check cuz not sure if it accepts flexible list of dicts
                    # since we don't know how many corrections will be needed ahead of time
                    # extra_body={"guided_json": response_schema.model_json_schema()},
                )
                response_text = response.choices[0].message.content
                if response_text is None:
                    raise ValueError("LLM response missing content")
                if response.usage is None:
                    raise ValueError("LLM response missing usage information")
                total_tokens = response.usage.total_tokens
                if block:
                    block.update_metadata(llm_tokens_used=total_tokens, llm_request_count=1)
                elapsed_s = time.perf_counter() - start_time
                marker_logger.info(
                    "CustomOpenAIService[%s] success elapsed_s=%.3f attempt=%s/%s prompt_tokens=%s completion_tokens=%s total_tokens=%s",
                    request_id,
                    elapsed_s,
                    tries,
                    total_tries,
                    response.usage.prompt_tokens,
                    response.usage.completion_tokens,
                    total_tokens,
                )
                return json.loads(response_text)
            except (APITimeoutError, RateLimitError) as e:
                elapsed_s = time.perf_counter() - start_time
                if tries == total_tries:
                    marker_logger.error(
                        "CustomOpenAIService[%s] retryable_error elapsed_s=%.3f giving_up attempt=%s/%s err=%s",
                        request_id,
                        elapsed_s,
                        tries,
                        total_tries,
                        e,
                    )
                    break
                wait_time = tries * self.retry_wait_time
                marker_logger.warning(
                    "CustomOpenAIService[%s] retryable_error elapsed_s=%.3f retry_in_s=%s attempt=%s/%s err=%s",
                    request_id,
                    elapsed_s,
                    wait_time,
                    tries,
                    total_tries,
                    e,
                )
                time.sleep(wait_time)
            except Exception as e:  # noqa: BLE001 - best effort logging
                elapsed_s = time.perf_counter() - start_time
                marker_logger.error(
                    "CustomOpenAIService[%s] error elapsed_s=%.3f attempt=%s/%s err=%s",
                    request_id,
                    elapsed_s,
                    tries,
                    total_tries,
                    e,
                )
                break

        elapsed_s = time.perf_counter() - start_time
        marker_logger.warning(
            "CustomOpenAIService[%s] failed elapsed_s=%.3f attempts=%s/%s", request_id, elapsed_s, tries, total_tries
        )
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
    sectionheader_openai_base_url: str
    sectionheader_openai_model: str
    sectionheader_hf_model_id: str
    sectionheader_timeout: int
    sectionheader_max_prompt_tokens: int
    openai_system_prompt: str
    openai_temperature: float
    hf_model_id: str
    hf_trust_remote_code: bool
    hf_count_image_tokens: bool
    log_prompt_token_count: bool
    max_prompt_tokens: int
    max_image_long_side: int
    timeout: int
    max_retries: int
    max_concurrency: int
    workers: int
    gpu_ids: str
    year_cutoff: int | None
    disable_forms: bool
    drop_front_pages: int
    drop_back_pages: int
    strip_repeated_toc: bool
    save_cleaned_html: bool

    def to_dict(self) -> dict:
        return asdict(self)


def parse_args() -> Args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--html-dir", required=True, help="Directory containing HTML files to process (recursively).")
    parser.add_argument(
        "--output-dir", default="outputs", help="Root output directory (writes `pdf/` and `markdown/` subfolders)."
    )
    parser.add_argument("--openai-model", required=True, help="Model name for OpenAI-compatible API.")
    parser.add_argument("--openai-system-prompt", default="", help="Optional system prompt for the LLM.")
    parser.add_argument("--openai-temperature", type=float, default=0.7, help="Sampling temperature.")
    parser.add_argument(
        "--token-count-hf-model-id",
        "--hf-model-id",
        default="",
        help=(
            "Optional HuggingFace model/tokenizer id used ONLY to estimate prompt token counts "
            "(defaults to --openai-model)."
        ),
    )
    hf_trust_group = parser.add_mutually_exclusive_group()
    hf_trust_group.add_argument(
        "--hf-trust-remote-code",
        dest="hf_trust_remote_code",
        action="store_true",
        help="Enable trust_remote_code when loading the tokenizer/processor for token counting (default: enabled).",
    )
    hf_trust_group.add_argument(
        "--no-hf-trust-remote-code",
        dest="hf_trust_remote_code",
        action="store_false",
        help="Disable trust_remote_code when loading the tokenizer/processor for token counting.",
    )
    parser.set_defaults(hf_trust_remote_code=True)
    parser.add_argument(
        "--no-hf-count-image-tokens", action="store_true", help="Disable image token counting (only count text tokens)."
    )
    parser.add_argument(
        "--log-prompt-token-count",
        action="store_true",
        help="Log estimated prompt tokens for each LLM request (best-effort).",
    )
    parser.add_argument(
        "--max-prompt-tokens",
        type=int,
        default=0,
        help="If >0, skip LLM calls whose estimated prompt exceeds this many tokens (best-effort).",
    )
    parser.add_argument(
        "--max-image-long-side",
        type=int,
        default=0,
        help="If >0, downscale images so max(width,height) <= this before sending to LLM.",
    )
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
    forms_group = parser.add_mutually_exclusive_group()
    forms_group.add_argument(
        "--disable-forms",
        dest="disable_forms",
        action="store_true",
        default=True,
        help="Disable `marker.processors.llm.llm_form.LLMFormProcessor` (skip all form LLM calls) (default).",
    )
    forms_group.add_argument(
        "--enable-forms",
        dest="disable_forms",
        action="store_false",
        help="Enable `marker.processors.llm.llm_form.LLMFormProcessor` (process forms).",
    )
    parser.add_argument(
        "--drop-front-pages",
        type=int,
        default=0,
        help="Skip the first N pages of each PDF for Marker processing (0 = keep all, -1 = SEC auto-detect).",
    )
    parser.add_argument(
        "--drop-back-pages",
        type=int,
        default=0,
        help="Skip the last N pages of each PDF for Marker processing (0 = keep all, -1 = SEC auto-detect).",
    )
    toc_group = parser.add_mutually_exclusive_group()
    toc_group.add_argument(
        "--strip-repeated-toc",
        dest="strip_repeated_toc",
        action="store_true",
        default=True,
        help="Strip repeated 'Table of Contents' backlink artifacts before Marker by re-rendering the PDF (default).",
    )
    toc_group.add_argument(
        "--no-strip-repeated-toc",
        dest="strip_repeated_toc",
        action="store_false",
        help="Disable stripping repeated 'Table of Contents' backlink artifacts before Marker.",
    )
    parser.add_argument(
        "--save-cleaned-html",
        action="store_true",
        default=False,
        help="When repeated ToC artifacts are stripped, save the cleaned HTML into the per-file debug folder.",
    )
    parser.add_argument(
        "--sectionheader-openai-model",
        default="",
        help="Optional model override for `LLMSectionHeaderProcessor` (text-only) requests.",
    )
    parser.add_argument(
        "--sectionheader-token-count-hf-model-id",
        "--sectionheader-hf-model-id",
        default="",
        help=(
            "Optional HuggingFace model/tokenizer id used ONLY to estimate prompt token counts for "
            "`LLMSectionHeaderProcessor` requests (falls back to --token-count-hf-model-id, then the request model)."
        ),
    )
    parser.add_argument(
        "--sectionheader-timeout",
        type=int,
        default=0,
        help="Optional timeout override (seconds) for `LLMSectionHeaderProcessor` requests.",
    )
    parser.add_argument(
        "--sectionheader-max-prompt-tokens",
        type=int,
        default=0,
        help="If >0, skip `LLMSectionHeaderProcessor` requests whose estimated prompt exceeds this many tokens.",
    )
    args = parser.parse_args()

    openai_base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    if not openai_base_url:
        raise RuntimeError("Set OPENAI_BASE_URL in `.env` (or the environment) before running.")

    sectionheader_openai_base_url = (os.environ.get("SECTIONHEADER_OPENAI_BASE_URL") or "").strip()

    return Args(
        html_dir=args.html_dir,
        output_dir=args.output_dir,
        openai_base_url=openai_base_url,
        openai_model=args.openai_model,
        sectionheader_openai_base_url=sectionheader_openai_base_url,
        sectionheader_openai_model=args.sectionheader_openai_model,
        sectionheader_hf_model_id=args.sectionheader_token_count_hf_model_id,
        sectionheader_timeout=args.sectionheader_timeout,
        sectionheader_max_prompt_tokens=args.sectionheader_max_prompt_tokens,
        openai_system_prompt=args.openai_system_prompt,
        openai_temperature=args.openai_temperature,
        hf_model_id=args.token_count_hf_model_id,
        hf_trust_remote_code=args.hf_trust_remote_code,
        hf_count_image_tokens=not args.no_hf_count_image_tokens,
        log_prompt_token_count=args.log_prompt_token_count,
        max_prompt_tokens=args.max_prompt_tokens,
        max_image_long_side=args.max_image_long_side,
        timeout=args.timeout,
        max_retries=args.max_retries,
        max_concurrency=args.max_concurrency,
        workers=args.workers,
        gpu_ids=args.gpu_ids,
        year_cutoff=args.year_cutoff,
        disable_forms=args.disable_forms,
        drop_front_pages=args.drop_front_pages,
        drop_back_pages=args.drop_back_pages,
        strip_repeated_toc=args.strip_repeated_toc,
        save_cleaned_html=args.save_cleaned_html,
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


def get_pdf_page_count(pdf_path: Path) -> int:
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        return len(doc)
    finally:
        doc.close()


def _pdfium_extract_page_text(doc: Any, page_index: int) -> str:
    page = doc[page_index]
    try:
        textpage = page.get_textpage()
        try:
            char_count = textpage.count_chars()
            if char_count <= 0:
                return ""
            return textpage.get_text_range(0, char_count)
        finally:
            textpage.close()
    finally:
        page.close()


def _sec_toc_listing_score(page_text: str) -> int:
    text = page_text.lower()
    score = 0
    if "table of contents" in text:
        score += 2

    if re.search(r"(?im)^\s*index\s*$", page_text):
        score += 2

    # SEC filings often (unhelpfully) include a "Table of Contents" header on every page after conversion.
    # To reduce false positives, prefer ToC *listing* pages that have multiple Item/Part lines ending in
    # page numbers (e.g. "ITEM 7A. ... 55").
    tocish_lines = re.findall(r"(?im)^\s*(part\s+[ivx]+\b.*\s\d+\s*$|item\s+\d+[a-z]?\b.*\s\d+\s*$)", page_text)
    tocish_count = len(tocish_lines)
    if tocish_count >= 8:
        score += 4
    elif tocish_count >= 4:
        score += 3
    elif tocish_count >= 2:
        score += 1

    if re.search(r"(?i)\bpart\s+i\b", page_text) or "financial information" in text:
        score += 1

    if re.search(r"(?i)\bpage\b", page_text):
        score += 1

    return score


def _sec_signatures_score(page_text: str) -> int:
    text = page_text.lower()
    score = 0
    if re.search(r"(?im)^\s*signatures?\s*$", page_text):
        score += 3
    elif "signatures" in text or "signature" in text:
        score += 1

    if "pursuant to the requirements" in text:
        score += 1
    if "registrant has duly caused" in text or "duly caused this report to be signed" in text:
        score += 1
    if "thereunto duly authorized" in text:
        score += 1
    if "/s/" in text:
        score += 2
    if "chief executive officer" in text or "chief financial officer" in text:
        score += 1
    return score


def _sec_exhibits_score(page_text: str) -> int:
    text = page_text.lower()
    score = 0
    if re.search(r"(?i)\bitem\s+6\.?\s*exhibits?\b", page_text):
        score += 2
    if "exhibit index" in text:
        score += 2
    if "the exhibits listed below are filed as part of" in text:
        score += 3
    if "exhibit number" in text and "exhibit title" in text:
        score += 1

    # Many filings include a compact exhibits list without the long "exhibits listed below" caption.
    # Heuristic: detect multiple exhibit numbers like 31.1 / 32.1 / 101.INS / 104.
    exhibit_id_hits = re.findall(r"(?im)^\s*\*?\s*(\d{1,3}\.\d{1,2}|\d{1,3}\.[A-Z]{2,4}|\d{1,3})\b", page_text)
    if len(exhibit_id_hits) >= 8:
        score += 3
    elif len(exhibit_id_hits) >= 4:
        score += 2
    elif len(exhibit_id_hits) >= 2:
        score += 1
    return score


def infer_sec_page_window(
    pdf_path: Path,
    *,
    max_front_scan_pages: int = 15,
    max_back_scan_pages: int = 25,
) -> tuple[int | None, int | None, dict[str, Any]]:
    """
    Infer a [start, end) window of pages to keep for SEC 10-K/10-Q style PDFs.

    - start: first "real" Table of Contents listing page (drops cover/boilerplate before it)
    - end: first Exhibits/Signatures page near the end (drops exhibits/signatures and anything after)
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(doc)
        front_scan = min(max_front_scan_pages, page_count)
        back_scan = min(max_back_scan_pages, page_count)

        toc_page: int | None = None
        toc_score: int = 0
        for i in range(front_scan):
            text = _pdfium_extract_page_text(doc, i)
            score = _sec_toc_listing_score(text)
            if score > toc_score:
                toc_score = score
                toc_page = i
            if score >= 5:
                toc_page = i
                break

        # Only trust the ToC heuristic when it clears a minimum score; otherwise treat as unknown.
        if toc_page is not None and toc_score < 4:
            toc_page = None

        exhibits_page: int | None = None
        exhibits_score: int = 0
        signatures_page: int | None = None
        signatures_score: int = 0
        start_back = max(0, page_count - back_scan)
        for i in range(start_back, page_count):
            text = _pdfium_extract_page_text(doc, i)

            e_score = _sec_exhibits_score(text)
            if e_score > exhibits_score:
                exhibits_score = e_score
                exhibits_page = i

            s_score = _sec_signatures_score(text)
            if s_score > signatures_score:
                signatures_score = s_score
                signatures_page = i

        if exhibits_page is not None and exhibits_score < 2:
            exhibits_page = None
        if signatures_page is not None and signatures_score < 3:
            signatures_page = None

        cutoff_page: int | None = None
        if exhibits_page is not None and signatures_page is not None:
            cutoff_page = min(exhibits_page, signatures_page)
        else:
            cutoff_page = exhibits_page if exhibits_page is not None else signatures_page

        info: dict[str, Any] = {
            "page_count": page_count,
            "toc_page": toc_page,
            "toc_score": toc_score,
            "exhibits_page": exhibits_page,
            "exhibits_score": exhibits_score,
            "signatures_page": signatures_page,
            "signatures_score": signatures_score,
            "cutoff_page": cutoff_page,
            "max_front_scan_pages": front_scan,
            "max_back_scan_pages": back_scan,
        }
        return toc_page, cutoff_page, info
    finally:
        doc.close()


def _first_nonempty_line(text: str) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


_TOC_LINK_RE = re.compile(r"(?is)<a\b[^>]*>\s*table\s+of\s+contents\s*</a>")


def strip_table_of_contents_links_from_html(html: str) -> tuple[str, int]:
    """
    SEC filings often include repeated 'Table of Contents' backlink anchors throughout the HTML.
    When paginated to PDF, these links frequently end up at the top of pages and contaminate OCR.
    """
    cleaned, count = _TOC_LINK_RE.subn("", html)
    return cleaned, count


def detect_repeated_toc_header_artifact(
    pdf_path: Path,
    *,
    max_scan_pages: int = 60,
    min_pages: int = 5,
    min_ratio: float = 0.35,
) -> tuple[bool, dict[str, Any]]:
    """
    Detect the common SEC artifact where 'Table of Contents' appears as the top line on many pages.

    We exclude ToC *listing* pages (high ToC listing score) because they legitimately contain that heading.
    """
    import pypdfium2 as pdfium

    doc = pdfium.PdfDocument(str(pdf_path))
    try:
        page_count = len(doc)
        scan_pages = min(max_scan_pages, page_count)

        toc_topline_pages: list[int] = []
        excluded_listing_pages: list[int] = []
        considered_pages: list[int] = []
        for i in range(scan_pages):
            text = _pdfium_extract_page_text(doc, i)
            if _sec_toc_listing_score(text) >= 6:
                excluded_listing_pages.append(i)
                continue
            considered_pages.append(i)
            if _first_nonempty_line(text).lower() == "table of contents":
                toc_topline_pages.append(i)

        considered_count = len(considered_pages)
        toc_count = len(toc_topline_pages)
        ratio = toc_count / considered_count if considered_count else 0.0
        is_artifact = (toc_count >= min_pages) and (ratio >= min_ratio)
        info = {
            "page_count": page_count,
            "scan_pages": scan_pages,
            "considered_pages": considered_pages,
            "excluded_listing_pages": excluded_listing_pages,
            "toc_topline_pages": toc_topline_pages,
            "toc_topline_count": toc_count,
            "toc_topline_ratio": ratio,
            "min_pages": min_pages,
            "min_ratio": min_ratio,
        }
        return is_artifact, info
    finally:
        doc.close()


def process_one(
    html_file_path: Path, html_dir: Path, pdf_root: Path, md_root: Path, debug_root: Path, args_dict: dict
) -> None:
    worker_converter, worker_markdown_renderer, renderer_config = get_worker_pipeline()
    converter_config = worker_converter.config
    if converter_config is None or not isinstance(converter_config, dict):
        raise RuntimeError("Expected PdfConverter.config to be a dict in this script")
    converter_config = cast(dict[str, Any], converter_config)

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

    t_start = time.perf_counter()

    # 1. convert HTML to PDF
    source_url = f"file://{html_file_path.absolute()}"
    weasyprint.HTML(source_url).write_pdf(str(pdf_path), stylesheets=[CSS_STYLESHEET])

    toc_artifact_info: dict[str, Any] | None = None
    if bool(args_dict.get("strip_repeated_toc", True)):
        has_artifact, artifact_info = detect_repeated_toc_header_artifact(pdf_path)
        toc_artifact_info = {"detected": has_artifact, **artifact_info}
        if has_artifact:
            html_raw = html_file_path.read_text(errors="ignore")
            cleaned_html, removed = strip_table_of_contents_links_from_html(html_raw)
            toc_artifact_info["removed_toc_link_anchors"] = removed
            if bool(args_dict.get("save_cleaned_html", False)):
                cleaned_html_path = debug_dir / "cleaned_for_pdf.html"
                with open(cleaned_html_path, "w") as f:
                    f.write(cleaned_html)

            weasyprint.HTML(string=cleaned_html, base_url=str(html_file_path.parent)).write_pdf(
                str(pdf_path), stylesheets=[CSS_STYLESHEET]
            )
            has_artifact_after, artifact_info_after = detect_repeated_toc_header_artifact(pdf_path)
            toc_artifact_info["after"] = {"detected": has_artifact_after, **artifact_info_after}
            logger.info(
                f"{rel_path}: removed repeated ToC links (removed={removed}) detected_before={has_artifact} detected_after={has_artifact_after}"
            )
    logger.success(f"Wrote intermediate PDF to {pdf_path}, time taken: {time.perf_counter() - t_start:.2f}s")

    # 2. convert PDF to document (so we can save multiple renderings/artifacts)
    # each 10Q usually requires ~40 LLM calls for LLMTableProcessor
    drop_front_pages = int(args_dict.get("drop_front_pages") or 0)
    drop_back_pages = int(args_dict.get("drop_back_pages") or 0)
    if drop_front_pages < -1 or drop_back_pages < -1:
        raise ValueError("--drop-front-pages and --drop-back-pages must be >= 0 (or -1 for SEC auto-detect)")

    sec_auto_drop_info: dict[str, Any] | None = None
    if drop_front_pages == -1 or drop_back_pages == -1:
        auto_start, auto_end, sec_auto_drop_info = infer_sec_page_window(pdf_path)
        logger.info(
            f"{rel_path}: SEC auto-drop inferred start={auto_start} end={auto_end} info={sec_auto_drop_info}"
        )
        if drop_front_pages == -1:
            drop_front_pages = int(auto_start or 0)
        if drop_back_pages == -1:
            page_count = int(sec_auto_drop_info.get("page_count") or get_pdf_page_count(pdf_path))
            drop_back_pages = int(page_count - auto_end) if auto_end is not None else 0

    if drop_front_pages or drop_back_pages:
        page_count = get_pdf_page_count(pdf_path)
        start = min(max(drop_front_pages, 0), page_count)
        end = max(start, page_count - max(drop_back_pages, 0))
        page_range = list(range(start, end))
        if not page_range:
            logger.warning(
                f"Skipping {rel_path}: after dropping front/back pages ({drop_front_pages}/{drop_back_pages}), no pages remain (page_count={page_count})"
            )
            return
        converter_config["page_range"] = page_range
    else:
        converter_config.pop("page_range", None)

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

    with open(debug_dir / "run_info.json", "w") as f:
        json.dump(
            {
                "input_html": str(html_file_path),
                "output_pdf": str(pdf_path),
                "output_markdown": str(md_path),
                "args": args_dict,
                "config": renderer_config,
                "page_range": converter_config.get("page_range"),
                "sec_auto_drop": sec_auto_drop_info,
                "toc_artifact": toc_artifact_info,
                "llm_error_count": llm_err_total,
                "page_count": page_cnt,
                "llm_error_ratio": llm_err_total / page_cnt if page_cnt > 0 else None,
            },
            f,
            indent=2,
        )

    logger.success(f"Finished processing {rel_path}")
    t_end = time.perf_counter()
    elapsed_s = t_end - t_start
    logger.info(f"Elapsed time for {rel_path}: {elapsed_s:.2f}s")
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

    html_paths = list(html_dir.rglob("*.html")) + list(html_dir.rglob("*.htm"))
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

    # Sort by year (desc) based on the expected YYYY-MM-DD filename suffix; files without a date suffix go last.
    def _html_sort_key(p: Path) -> tuple[int, int, str]:
        year = extract_year_from_filename(p)
        if year is None:
            return (1, 0, p.as_posix())
        return (0, -year, p.as_posix())

    html_paths.sort(key=_html_sort_key)

    if args.workers < 1:
        raise RuntimeError("--workers must be >= 1")

    schema_routes: dict[str, dict] = {}
    sectionheader_schema_key = "marker.processors.llm.llm_sectionheader.SectionHeaderSchema"
    if args.sectionheader_openai_base_url:
        if args.sectionheader_openai_model is None or args.sectionheader_openai_model.strip() == "":
            raise RuntimeError("--sectionheader-openai-model must be set when --sectionheader-openai-base-url is set")
        schema_routes[sectionheader_schema_key] = {}
        schema_routes[sectionheader_schema_key]["openai_base_url"] = args.sectionheader_openai_base_url
        schema_routes[sectionheader_schema_key]["openai_model"] = args.sectionheader_openai_model
        if args.sectionheader_timeout and args.sectionheader_timeout > 0:
            schema_routes[sectionheader_schema_key]["openai_timeout"] = args.sectionheader_timeout
    if args.sectionheader_max_prompt_tokens and args.sectionheader_max_prompt_tokens > 0:
        schema_routes.setdefault(sectionheader_schema_key, {})
        schema_routes[sectionheader_schema_key]["max_prompt_tokens"] = args.sectionheader_max_prompt_tokens
    if (args.sectionheader_hf_model_id or "").strip():
        schema_routes.setdefault(sectionheader_schema_key, {})
        schema_routes[sectionheader_schema_key]["hf_model_id"] = args.sectionheader_hf_model_id

    processors_override: str | None = None
    if args.disable_forms:
        from marker.converters.pdf import PdfConverter as _PdfConverter
        from marker.processors import BaseProcessor
        from marker.util import classes_to_strings

        disabled = {"marker.processors.llm.llm_form.LLMFormProcessor"}
        default_processors = cast(list[type[BaseProcessor]], list(_PdfConverter.default_processors))
        filtered_processors = [p for p in default_processors if f"{p.__module__}.{p.__name__}" not in disabled]
        processors_override = ",".join(classes_to_strings(filtered_processors))

    config = {
        "output_format": "markdown",
        "use_llm": True,
        "llm_service": "scripts.process_html_to_markdown.CustomOpenAIService",
        "openai_api_key": openai_api_key,
        "openai_base_url": args.openai_base_url,
        "openai_model": args.openai_model,
        "CustomOpenAIService_openai_temperature": args.openai_temperature,
        "CustomOpenAIService_openai_system_prompt": args.openai_system_prompt,
        "CustomOpenAIService_openai_timeout": args.timeout,
        "CustomOpenAIService_openai_max_retries": args.max_retries,
        "CustomOpenAIService_max_image_long_side": args.max_image_long_side,
        "CustomOpenAIService_schema_routes": schema_routes,
        "CustomOpenAIService_hf_model_id": args.hf_model_id,
        "CustomOpenAIService_hf_trust_remote_code": args.hf_trust_remote_code,
        "CustomOpenAIService_hf_count_image_tokens": args.hf_count_image_tokens,
        "CustomOpenAIService_log_prompt_token_count": args.log_prompt_token_count,
        "CustomOpenAIService_max_prompt_tokens": args.max_prompt_tokens,
        # Marker calls this "max_concurrency" (default 3). This controls parallel in-flight
        # LLM requests (not "batching" into a single request).
        "max_concurrency": args.max_concurrency,
        "LLMTableProcessor_max_concurrency": args.max_concurrency,
        "disable_image_extraction": True,
        "force_ocr": False,
    }
    if processors_override:
        config["processors"] = processors_override

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

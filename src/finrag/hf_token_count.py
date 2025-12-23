import base64
from dataclasses import dataclass, field
from functools import lru_cache
from io import BytesIO
from typing import Any


@dataclass(frozen=True)
class TokenCountResult:
    total_tokens: int
    text_tokens: int
    image_tokens: int | None
    model_max_length: int | None
    method: str
    warnings: list[str] = field(default_factory=list)


def count_tokens_openai_messages(
    messages: list[dict[str, Any]],
    model_id: str,
    *,
    revision: str | None = None,
    trust_remote_code: bool = False,
    count_images: bool = True,
    add_generation_prompt: bool = True,
) -> TokenCountResult:
    """
    Best-effort token counting for OpenAI-style chat `messages` using HuggingFace models.

    Supports:
    - Text-only chat models via `AutoTokenizer`
    - Some multimodal models via `AutoProcessor` when messages include `image_url` data URLs

    Returns a TokenCountResult with `image_tokens=None` when image tokenization isn't supported.
    """
    warnings: list[str] = []

    images = _extract_pil_images_from_openai_messages(messages, warnings) if count_images else []
    has_images = bool(images)

    if has_images:
        try:
            processor = _load_processor(model_id, revision=revision, trust_remote_code=trust_remote_code)
            total_tokens = _count_with_processor(
                processor,
                messages,
                images,
                add_generation_prompt=add_generation_prompt,
            )
            tokenizer = getattr(processor, "tokenizer", None) or _load_tokenizer(
                model_id, revision=revision, trust_remote_code=trust_remote_code
            )
            text_only_messages = _strip_images_from_messages(messages)
            text_tokens = _count_with_tokenizer(
                tokenizer,
                text_only_messages,
                add_generation_prompt=add_generation_prompt,
            )
            image_tokens = max(0, total_tokens - text_tokens)
            model_max_length = _get_model_max_length(tokenizer)
            return TokenCountResult(
                total_tokens=total_tokens,
                text_tokens=text_tokens,
                image_tokens=image_tokens,
                model_max_length=model_max_length,
                method="processor",
                warnings=warnings,
            )
        except Exception as exc:  # noqa: BLE001 - best effort fallback
            warnings.append(f"processor_count_failed: {exc}")

    tokenizer = _load_tokenizer(model_id, revision=revision, trust_remote_code=trust_remote_code)
    text_tokens = _count_with_tokenizer(
        tokenizer,
        messages if not has_images else _strip_images_from_messages(messages),
        add_generation_prompt=add_generation_prompt,
    )
    model_max_length = _get_model_max_length(tokenizer)
    method = "tokenizer_text_only" if has_images else "tokenizer"
    return TokenCountResult(
        total_tokens=text_tokens,
        text_tokens=text_tokens,
        image_tokens=(0 if has_images else None),
        model_max_length=model_max_length,
        method=method,
        warnings=warnings,
    )


def _get_model_max_length(tokenizer: Any) -> int | None:
    try:
        val = int(getattr(tokenizer, "model_max_length", 0) or 0)
    except Exception:  # noqa: BLE001
        return None
    # HF uses very large sentinels for "unknown"
    if val <= 0 or val >= 10**12:
        return None
    return val


@lru_cache(maxsize=8)
def _load_tokenizer(model_id: str, *, revision: str | None, trust_remote_code: bool) -> Any:
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)


@lru_cache(maxsize=4)
def _load_processor(model_id: str, *, revision: str | None, trust_remote_code: bool) -> Any:
    from transformers import AutoProcessor

    return AutoProcessor.from_pretrained(model_id, revision=revision, trust_remote_code=trust_remote_code)


def _count_with_tokenizer(tokenizer: Any, messages: list[dict[str, Any]], *, add_generation_prompt: bool) -> int:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=add_generation_prompt)
            if isinstance(ids, list) and ids and isinstance(ids[0], int):
                return len(ids)
            if isinstance(ids, list) and ids and isinstance(ids[0], list):
                return len(ids[0])
            return len(ids)  # best effort
        except Exception:  # noqa: BLE001 - fall back when chat templates aren't configured
            pass

    text = _fallback_messages_to_text(messages)
    enc = tokenizer(text, add_special_tokens=True, return_attention_mask=False, return_tensors=None)
    return len(enc["input_ids"])


def _count_with_processor(
    processor: Any,
    messages: list[dict[str, Any]],
    images: list[Any],
    *,
    add_generation_prompt: bool,
) -> int:
    if hasattr(processor, "apply_chat_template"):
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
    else:
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)
        else:
            text = _fallback_messages_to_text(messages)

    inputs = processor(text=[text], images=images, padding=False, return_tensors="pt")
    if "attention_mask" in inputs:
        return int(inputs["attention_mask"][0].sum().item())
    return int(inputs["input_ids"].shape[-1])


def _strip_images_from_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    stripped: list[dict[str, Any]] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            content = [c for c in content if not (isinstance(c, dict) and c.get("type") == "image_url")]
        stripped.append({**msg, "content": content})
    return stripped


def _extract_pil_images_from_openai_messages(messages: list[dict[str, Any]], warnings: list[str]) -> list[Any]:
    from PIL import Image

    images: list[Any] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not isinstance(part, dict) or part.get("type") != "image_url":
                continue
            image_url = part.get("image_url") or {}
            url = image_url.get("url")
            if not isinstance(url, str):
                continue
            if not url.startswith("data:image/") or ";base64," not in url:
                warnings.append("unsupported_image_url")
                continue
            try:
                b64 = url.split(";base64,", 1)[1]
                data = base64.b64decode(b64)
                img = Image.open(BytesIO(data))
                images.append(img)
            except Exception as exc:  # noqa: BLE001 - best effort
                warnings.append(f"image_decode_failed: {exc}")
    return images


def _fallback_messages_to_text(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for msg in messages:
        role = str(msg.get("role", "user"))
        content = msg.get("content", "")
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text = "\n".join(
                str(c.get("text", "")) for c in content if isinstance(c, dict) and c.get("type") == "text"
            )
        else:
            text = str(content)
        parts.append(f"{role}: {text}".strip())
    return "\n".join(parts).strip()

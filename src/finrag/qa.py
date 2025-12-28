import os
from typing import Sequence

from finrag.dataclasses import ScoredChunk
from finrag.generation_controls import AnswerStyle
from finrag.llm_clients import ChatMessage, LLMClient

_DRAFT_SYSTEM_PROMPT = (
    "You are a senior investment banking analyst. "
    "You are tasked with answering questions over SEC financial filings of publicly-traded companies. "
    "Write detailed and accurate analyses that cite the provided context. "
    "Use only the provided context to answer the question. If the context does not contain sufficient information, "
    "state that you cannot answer the question based on the provided context."
)

_REFINE_SYSTEM_PROMPT = (
    "You are a principal investment banking analyst leading a top-tier hedge fund. "
    "Your subordinate has written up a draft report for your review. "
    "After your review, you will finalize the report for submission to the investment board, "
    "where millions of dollars will be invested. "
    "You must:\n"
    "1) check the draft answer against the context;\n"
    "2) fix hallucinations;\n"
    "3) clearly state if context is insufficient."
)

_STYLE_GUIDANCE: dict[AnswerStyle, str] = {
    "concise": (
        "Write a concise answer. Prefer a short paragraph + bullets. "
        "Avoid long preambles. Keep it as short as possible while still accurate."
    ),
    "normal": (
        "Write a clear, structured analysis. Include key numbers and key takeaways. "
        "Keep it reasonably detailed but not overly long."
    ),
    "detailed": (
        "Write a detailed report with clear section headers (e.g., Executive summary, Key points, Risks, Data points). "
        "Be comprehensive and specific."
    ),
}

_CITATION_GUIDANCE = (
    "Cite sources (chunks) inline by quoting the doc_id of the chunk as [doc=...] "
    "for the relevant claim(s). Use only the provided context."
)


def _system_prompt(base: str, *, answer_style: AnswerStyle, extra: str | None) -> str:
    parts = [base.strip(), _STYLE_GUIDANCE[answer_style].strip(), _CITATION_GUIDANCE.strip()]
    if extra and extra.strip():
        parts.append(extra.strip())
    return "\n\n".join(parts)


def build_context(chunks: Sequence[ScoredChunk], max_tokens: int) -> str:
    budget_chars = max_tokens * 4
    parts: list[str] = []
    used = 0
    context_key = os.getenv("CONTEXT_METADATA_KEY", "context").strip() or "context"
    for sc in chunks:
        # headings_s = "; ".join(sc.chunk.headings)
        meta_bits = [f"doc={sc.chunk.doc_id}"]
        # NOTE: page_no seems to be None for all our chunks.
        # TODO: see if we can get `marker` to populate it (during `process_html_to_markdown.py`)
        # if sc.chunk.page_no not in (None, ""):
        #     meta_bits.append(f"page={sc.chunk.page_no}")
        # Don't include headings for now, as "index_text" already has similar "section path" info
        # meta_bits.append(f"headings={headings_s}")
        meta = "[" + " ".join(meta_bits) + "]"

        meta_dict = sc.chunk.metadata or {}
        text = (meta_dict.get("index_text") or sc.chunk.text or "").strip()
        context = str(meta_dict.get(context_key) or "").strip()
        if context:
            block = f"{meta}\n{text}\n\nContext:\n{context}\n"
        else:
            block = f"{meta}\n{text}\n"
        if used + len(block) > budget_chars:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)


def answer_question_two_stage(
    llm: LLMClient,
    question: str,
    reranked: Sequence[ScoredChunk],
    *,
    draft_max_tokens: int = 65_536,
    final_max_tokens: int = 32_768,
    temperature_draft: float = 0.1,
) -> tuple[str, str]:
    draft_prompt = build_draft_prompt(question, reranked, draft_max_tokens=draft_max_tokens)
    draft = llm.chat(draft_prompt, temperature=temperature_draft)

    refine_prompt = build_refine_prompt(question, draft, reranked, final_max_tokens=final_max_tokens)
    final = llm.chat(refine_prompt, temperature=0.0)
    return draft, final


def build_draft_prompt(
    question: str,
    reranked: Sequence[ScoredChunk],
    *,
    draft_max_tokens: int = 65_536,
    answer_style: AnswerStyle = "normal",
    system_extra: str | None = None,
) -> list[ChatMessage]:
    ctx1 = build_context(reranked, max_tokens=draft_max_tokens)
    return [
        {
            "role": "system",
            "content": _system_prompt(_DRAFT_SYSTEM_PROMPT, answer_style=answer_style, extra=system_extra),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Context:\n{ctx1}\n\n"
                "Write your analysis to address the question based on the provided context. "
                # "At the end, list which [doc=..., page=... (if page is present)] segments you used."
            ),
        },
    ]


def build_refine_prompt(
    question: str,
    draft: str,
    reranked: Sequence[ScoredChunk],
    *,
    final_max_tokens: int = 32_768,
    answer_style: AnswerStyle = "normal",
    system_extra: str | None = None,
) -> list[ChatMessage]:
    ctx2 = build_context(reranked, max_tokens=final_max_tokens)
    return [
        {
            "role": "system",
            "content": _system_prompt(_REFINE_SYSTEM_PROMPT, answer_style=answer_style, extra=system_extra),
        },
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Draft answer:\n{draft}\n\n"
                f"Context:\n{ctx2}\n\n"
                "Now write a refined answer. "
                # "At the end, add a 'Sources' section referencing [doc=..., page=... (if page present)]."
            ),
        },
    ]

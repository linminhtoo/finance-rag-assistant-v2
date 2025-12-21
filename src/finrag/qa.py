from typing import Sequence

from finrag.dataclasses import ScoredChunk
from finrag.llm_clients import ChatMessage, LLMClient

_DRAFT_SYSTEM_PROMPT = (
    "You are a careful assistant answering questions over SEC filings of publicly-traded companies. "
    "Always stay grounded in the provided context."
)

_REFINE_SYSTEM_PROMPT = (
    "You are a senior investment banking analyst. You must:\n"
    "1) check the draft answer against the context;\n"
    "2) fix hallucinations;\n"
    "3) clearly state if context is insufficient."
)


def build_context(chunks: Sequence[ScoredChunk], max_tokens: int) -> str:
    budget_chars = max_tokens * 4
    parts: list[str] = []
    used = 0
    for sc in chunks:
        meta = f"[doc={sc.chunk.doc_id} page={sc.chunk.page_no} headings={'; '.join(sc.chunk.headings)}]"
        text = sc.chunk.text.strip()
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
    draft_max_tokens: int = 900,
    final_max_tokens: int = 1500,
    temperature_draft: float = 0.1,
) -> tuple[str, str]:
    ctx1 = build_context(reranked, max_tokens=draft_max_tokens)
    draft_prompt: list[ChatMessage] = [
        {"role": "system", "content": _DRAFT_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Question:\n{question}\n\n"
                f"Context:\n{ctx1}\n\n"
                "Answer concisely and list which [doc=..., page=...] segments you used."
            ),
        },
    ]
    draft = llm.chat(draft_prompt, temperature=temperature_draft)

    ctx2 = build_context(reranked, max_tokens=final_max_tokens)
    refine_prompt: list[ChatMessage] = [
        {"role": "system", "content": _REFINE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"User question:\n{question}\n\n"
                f"Draft answer:\n{draft}\n\n"
                f"Context:\n{ctx2}\n\n"
                "Now write a refined answer. Start with a short paragraph, "
                "then add a 'Sources' section referencing [doc=..., page=...]."
            ),
        },
    ]
    final = llm.chat(refine_prompt, temperature=0.0)
    return draft, final

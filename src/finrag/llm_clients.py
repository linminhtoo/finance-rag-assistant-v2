import os
from typing import Literal, Protocol, TypedDict, cast, runtime_checkable

import numpy as np
from langsmith.wrappers import wrap_openai
from mistralai import Mistral
from mistralai.models.chatcompletionrequest import MessagesTypedDict
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@runtime_checkable
class LLMClient(Protocol):
    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> str: ...


class MistralClientWrapper:
    def __init__(
        self,
        api_key_env: str = "MISTRAL_API_KEY",
        chat_model: str = "mistral-small-latest",
        embed_model: str = "mistral-embed",
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")
        self.client = Mistral(api_key=api_key)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embed_model, inputs=texts)
        vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        return np.vstack(vectors)

    def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> str:
        mistral_messages = [cast(MessagesTypedDict, msg) for msg in messages]
        res = self.client.chat.complete(
            model=self.chat_model, messages=mistral_messages, temperature=temperature, stream=False
        )
        try:
            return res.choices[0].message.content  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to get chat response: {e}") from e


class OpenAIClientWrapper:
    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        chat_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-large",
        base_url_env: str = "OPENAI_BASE_URL",
        organization_env: str = "OPENAI_ORGANIZATION",
        project_env: str = "OPENAI_PROJECT",
        langsmith_trace: bool = False,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")

        base_url = os.getenv(base_url_env) or None
        organization = os.getenv(organization_env) or None
        project = os.getenv(project_env) or None

        self.client = OpenAI(api_key=api_key, base_url=base_url, organization=organization, project=project)
        if langsmith_trace:
            self.client = wrap_openai(self.client)
        self.chat_model = chat_model
        self.embed_model = embed_model

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        resp = self.client.embeddings.create(model=self.embed_model, input=texts)
        vectors = [np.array(d.embedding, dtype=np.float32) for d in resp.data]
        return np.vstack(vectors)

    def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> str:
        oa_messages: list[ChatCompletionMessageParam] = [cast(ChatCompletionMessageParam, msg) for msg in messages]
        res = self.client.chat.completions.create(model=self.chat_model, messages=oa_messages, temperature=temperature)
        try:
            content = res.choices[0].message.content
            if content is None:
                raise RuntimeError("OpenAI returned empty message content")
            return content
        except Exception as e:
            raise RuntimeError(f"Failed to get OpenAI chat response: {e}") from e


class FastEmbedClientWrapper:
    """
    Local embedding-only client using `fastembed`.

    This is useful for running retrieval-only evals without networked embedding APIs.
    """

    def __init__(self, embed_model: str = "BAAI/bge-m3"):
        try:
            from fastembed import TextEmbedding  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError("fastembed is not available; install qdrant-client[fastembed-gpu]") from e
        self.embed_model = embed_model
        self._model = TextEmbedding(model_name=embed_model)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = [np.array(v, dtype=np.float32) for v in self._model.embed(texts)]
        return np.vstack(vectors)

    def chat(self, messages: list[ChatMessage], temperature: float = 0.1) -> str:  # pragma: no cover
        raise RuntimeError("FastEmbedClientWrapper does not support chat()")


def get_llm_client(provider: str | None = None, langsmith_trace: bool = False, **kwargs) -> LLMClient:
    name = (provider or os.getenv("LLM_PROVIDER") or "mistral").strip().lower()
    if langsmith_trace and name != "openai":
        raise ValueError("Langsmith tracing is only supported for OpenAI provider")

    if name == "mistral":
        return MistralClientWrapper(**kwargs)
    if name == "openai":
        return OpenAIClientWrapper(langsmith_trace=langsmith_trace, **kwargs)
    if name == "fastembed":
        return FastEmbedClientWrapper(**kwargs)
    raise ValueError(f"Unsupported LLM provider: {provider}")

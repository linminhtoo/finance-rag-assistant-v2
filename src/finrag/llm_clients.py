import os
from collections.abc import Iterator
from typing import Any, Literal, Protocol, TypedDict, cast, overload, runtime_checkable

import numpy as np
from langsmith.wrappers import wrap_openai
from mistralai import Mistral
from mistralai.models.chatcompletionrequest import MessagesTypedDict
from mistralai.models.responseformat import ResponseFormatTypedDict as MistralResponseFormatTypedDict
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from openai.types.chat.completion_create_params import ResponseFormat as OpenAIResponseFormat
from pydantic import BaseModel


class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant", "tool"]
    content: str


@runtime_checkable
class LLMClient(Protocol):
    chat_model: str
    embed_model: str

    def embed_texts(self, texts: list[str]) -> np.ndarray: ...

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> str: ...

    def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> Iterator[str]: ...


@overload
def _build_response_format(
    *, provider: Literal["mistral"], response_model: type[BaseModel]
) -> MistralResponseFormatTypedDict: ...


@overload
def _build_response_format(*, provider: Literal["openai"], response_model: type[BaseModel]) -> OpenAIResponseFormat: ...


def _build_response_format(*, provider: Literal["mistral", "openai"], response_model: type[BaseModel]) -> Any:
    schema = response_model.model_json_schema()
    schema_name = response_model.__name__
    if provider == "mistral":
        return cast(
            MistralResponseFormatTypedDict,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema_definition": schema,
                },
            },
        )
    if provider == "openai":
        return cast(
            OpenAIResponseFormat,
            {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "schema": schema,
                },
            },
        )
    raise ValueError(f"Unsupported provider for response_format: {provider}")


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

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        mistral_messages = [cast(MessagesTypedDict, msg) for msg in messages]
        response_format: MistralResponseFormatTypedDict | None = (
            _build_response_format(provider="mistral", response_model=response_model) if response_model else None
        )
        res = self.client.chat.complete(
            model=self.chat_model,
            messages=mistral_messages,
            temperature=temperature,
            stream=False,
            response_format=response_format,
        )
        try:
            return res.choices[0].message.content  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Failed to get chat response: {e}") from e

    def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> Iterator[str]:
        mistral_messages = [cast(MessagesTypedDict, msg) for msg in messages]
        response_format: MistralResponseFormatTypedDict | None = (
            _build_response_format(provider="mistral", response_model=response_model) if response_model else None
        )

        stream = self.client.chat.stream(  # type: ignore[attr-defined]
            model=self.chat_model, messages=mistral_messages, temperature=temperature, response_format=response_format
        )
        for evt in stream:
            content = None
            try:
                content = evt.data.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                content = None
            if content:
                yield str(content)


class OpenAIClientWrapper:
    def __init__(
        self,
        api_key_env: str = "OPENAI_API_KEY",
        chat_model: str = "gpt-4o-mini",
        embed_model: str = "text-embedding-3-large",
        base_url: str | None = None,
        base_url_env: str = "OPENAI_BASE_URL",
        organization_env: str = "OPENAI_ORGANIZATION",
        project_env: str = "OPENAI_PROJECT",
        langsmith_trace: bool = False,
    ):
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise RuntimeError(f"{api_key_env} is not set")

        base_url = base_url or (os.getenv(base_url_env) or None)
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

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> str:
        oa_messages: list[ChatCompletionMessageParam] = [cast(ChatCompletionMessageParam, msg) for msg in messages]
        if response_model is None:
            res = self.client.chat.completions.create(model=self.chat_model, messages=oa_messages, temperature=temperature)
        else:
            response_format = _build_response_format(provider="openai", response_model=response_model)
            res = self.client.chat.completions.create(
                model=self.chat_model, messages=oa_messages, temperature=temperature, response_format=response_format
            )
        try:
            content = res.choices[0].message.content
            if content is None:
                raise RuntimeError("OpenAI returned empty message content")
            return content
        except Exception as e:
            raise RuntimeError(f"Failed to get OpenAI chat response: {e}") from e

    def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> Iterator[str]:
        oa_messages: list[ChatCompletionMessageParam] = [cast(ChatCompletionMessageParam, msg) for msg in messages]
        if response_model is None:
            stream = self.client.chat.completions.create(
                model=self.chat_model, messages=oa_messages, temperature=temperature, stream=True
            )
        else:
            response_format = _build_response_format(provider="openai", response_model=response_model)
            stream = self.client.chat.completions.create(
                model=self.chat_model,
                messages=oa_messages,
                temperature=temperature,
                stream=True,
                response_format=response_format,
            )
        for evt in stream:
            try:
                delta = evt.choices[0].delta.content  # type: ignore[attr-defined]
            except Exception:
                delta = None
            if delta:
                yield str(delta)


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
        self.chat_model = ""
        self.embed_model = embed_model
        self._model = TextEmbedding(model_name=embed_model)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        vectors = [np.array(v, dtype=np.float32) for v in self._model.embed(texts)]
        return np.vstack(vectors)

    def chat(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> str:  # pragma: no cover
        raise RuntimeError("FastEmbedClientWrapper does not support chat()")

    def chat_stream(
        self,
        messages: list[ChatMessage],
        temperature: float = 0.1,
        response_model: type[BaseModel] | None = None,
    ) -> Iterator[str]:  # pragma: no cover
        raise RuntimeError("FastEmbedClientWrapper does not support chat_stream()")


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

"""Any server that speaks the OpenAI Chat Completions API (OpenAI, vLLM, Ollama, ...)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from evalkit.errors import ProviderError
from evalkit.models import Message
from evalkit.providers._http import HTTPProvider, HTTPProviderConfig
from evalkit.providers.base import Completion, register_provider
from evalkit.results import Usage


class OpenAICompatibleConfig(HTTPProviderConfig):
    model: str | None = "gpt-4o-mini"
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    embedding_model: str = "text-embedding-3-small"


@register_provider("openai", "openai-compatible")
class OpenAICompatibleProvider(HTTPProvider):
    name = "openai"
    Config = OpenAICompatibleConfig
    config: OpenAICompatibleConfig

    def auth_headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}"}

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.model_dump() for m in messages],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        body = self.post_json("/chat/completions", payload)
        try:
            text = body["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"openai: unexpected response shape: {body!r:.500}") from exc
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            model=body.get("model", self.model),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens"),
                output_tokens=usage.get("completion_tokens"),
            ),
        )

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        body = self.post_json(
            "/embeddings", {"model": self.config.embedding_model, "input": list(texts)}
        )
        try:
            rows = sorted(body["data"], key=lambda row: row["index"])
            return [list(map(float, row["embedding"])) for row in rows]
        except (KeyError, TypeError) as exc:
            raise ProviderError(f"openai: unexpected embeddings response: {body!r:.500}") from exc

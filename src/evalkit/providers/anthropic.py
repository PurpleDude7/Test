"""Anthropic's Messages API."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import Field

from evalkit.errors import ProviderError
from evalkit.models import Message
from evalkit.providers._http import HTTPProvider, HTTPProviderConfig
from evalkit.providers.base import Completion, register_provider
from evalkit.results import Usage


class AnthropicConfig(HTTPProviderConfig):
    model: str | None = "claude-opus-5-5"
    base_url: str = "https://api.anthropic.com/v1"
    api_key_env: str = "ANTHROPIC_API_KEY"
    api_version: str = "2023-06-01"
    default_max_tokens: int = Field(default=1024, gt=0)
    """The Messages API requires ``max_tokens``; used when the suite does not set one."""


@register_provider("anthropic")
class AnthropicProvider(HTTPProvider):
    name = "anthropic"
    Config = AnthropicConfig
    config: AnthropicConfig

    def auth_headers(self) -> dict[str, str]:
        return {"x-api-key": self.api_key, "anthropic-version": self.config.api_version}

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        # The API takes the system prompt as a top-level field, not as a message.
        system = "\n\n".join(m.content for m in messages if m.role == "system")
        payload: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens or self.config.default_max_tokens,
            "messages": [m.model_dump() for m in messages if m.role != "system"],
        }
        if system:
            payload["system"] = system
        if temperature is not None:
            payload["temperature"] = temperature
        body = self.post_json("/messages", payload)
        blocks = body.get("content")
        if not isinstance(blocks, list):
            raise ProviderError(f"anthropic: unexpected response shape: {body!r:.500}")
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        usage = body.get("usage") or {}
        return Completion(
            text=text,
            model=body.get("model", self.model),
            usage=Usage(
                input_tokens=usage.get("input_tokens"),
                output_tokens=usage.get("output_tokens"),
            ),
        )

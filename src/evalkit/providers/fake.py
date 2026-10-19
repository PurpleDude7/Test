"""A deterministic, offline provider for tests, demos, and dry runs."""

from __future__ import annotations

import hashlib
import math
import re
import threading
from collections.abc import Sequence

from pydantic import Field

from evalkit.errors import ProviderError
from evalkit.models import Message
from evalkit.providers.base import Completion, Provider, ProviderConfig, register_provider
from evalkit.results import Usage

_TOKEN = re.compile(r"\w+")


class FakeConfig(ProviderConfig):
    model: str | None = "fake-1"
    responses: dict[str, str] = Field(default_factory=dict)
    """Pattern -> reply. Checked in order against the last user message; first match wins."""
    regex: bool = False
    """Treat ``responses`` keys as regular expressions instead of substrings."""
    default: str | None = None
    """Reply when nothing matches. ``None`` echoes the last user message back."""
    fail_on: list[str] = Field(default_factory=list)
    """Substrings that make the provider raise ProviderError, to exercise error paths."""
    embedding_dim: int = Field(default=256, gt=0)


@register_provider("fake")
class FakeProvider(Provider):
    """Replies from a lookup table and records every call.

    Embeddings are hashed bags of words, so texts sharing words get similar vectors,
    which is enough to exercise ``embedding_similarity`` without a real model.
    """

    name = "fake"
    Config = FakeConfig
    config: FakeConfig

    def __init__(self, config: FakeConfig) -> None:
        super().__init__(config)
        self.calls: list[list[Message]] = []
        self._lock = threading.Lock()

    @classmethod
    def with_responses(cls, responses: dict[str, str], **kwargs: object) -> FakeProvider:
        """Convenience constructor for tests."""
        return cls.from_config({"responses": responses, **kwargs})

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        with self._lock:
            self.calls.append(list(messages))
        prompt = next((m.content for m in reversed(messages) if m.role == "user"), "")
        for trigger in self.config.fail_on:
            if trigger in prompt:
                raise ProviderError(f"fake failure triggered by {trigger!r}")
        text = self._lookup(prompt)
        usage = Usage(input_tokens=len(prompt.split()), output_tokens=len(text.split()))
        return Completion(text=text, model=self.model, usage=usage)

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _lookup(self, prompt: str) -> str:
        for pattern, reply in self.config.responses.items():
            hit = re.search(pattern, prompt) if self.config.regex else pattern in prompt
            if hit:
                return reply
        return self.config.default if self.config.default is not None else prompt

    def _embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self.config.embedding_dim
        for token in _TOKEN.findall(text.lower()):
            digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
            vec[int.from_bytes(digest, "big") % len(vec)] += 1.0
        norm = math.sqrt(sum(v * v for v in vec))
        return [v / norm for v in vec] if norm else vec

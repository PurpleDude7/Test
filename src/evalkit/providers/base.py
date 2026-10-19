"""The Provider interface every model backend implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Self

from pydantic import BaseModel, ConfigDict, ValidationError

from evalkit.errors import NotSupportedError, ProviderError
from evalkit.models import Message
from evalkit.registry import Registry
from evalkit.results import Usage


class Completion(BaseModel):
    """What a provider returns for one chat request."""

    text: str
    model: str | None = None
    usage: Usage | None = None
    cached: bool = False
    """True when the reply came from a response cache instead of the provider."""


class ProviderConfig(BaseModel):
    """Base for provider settings. Unknown keys are rejected so typos surface early."""

    model_config = ConfigDict(extra="forbid")

    model: str | None = None


class Provider(ABC):
    """A chat model backend.

    Subclasses declare a pydantic ``Config`` model and implement :meth:`complete`.
    Implementations must be safe to call from several threads at once, because the
    runner fans cases out over a thread pool.
    """

    name: str
    """Registry name. A class attribute on real providers; wrappers copy it per instance."""
    Config: ClassVar[type[ProviderConfig]] = ProviderConfig

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @classmethod
    def from_config(cls, config: Mapping[str, Any] | None = None) -> Self:
        try:
            parsed = cls.Config.model_validate(dict(config or {}))
        except ValidationError as exc:
            raise ProviderError(f"invalid config for provider {cls.name!r}: {exc}") from exc
        return cls(parsed)

    @property
    def model(self) -> str | None:
        return self.config.model

    @abstractmethod
    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        """Return the assistant's reply to ``messages``."""

    def cache_identity(self) -> dict[str, Any]:
        """What, besides the request, determines a reply; part of the response-cache key.

        Never include secrets: the key is hashed, but keep it safe to log anyway.
        """
        return {"provider": self.name, "model": self.model}

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per text. Optional capability."""
        raise NotSupportedError(f"provider {self.name!r} does not support embeddings")

    def close(self) -> None:  # noqa: B027 - optional hook, a no-op by default
        """Release network resources. Safe to call more than once."""


provider_registry: Registry[Provider] = Registry("provider", "evalkit.providers")


def register_provider(*names: str) -> Any:
    """Class decorator: make a Provider subclass available under ``names``."""
    return provider_registry.register(*names)


def create_provider(name: str, config: Mapping[str, Any] | None = None) -> Provider:
    """Instantiate a registered provider by name."""
    return provider_registry.get(name).from_config(config)

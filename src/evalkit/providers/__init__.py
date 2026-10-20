"""Model providers. Importing this package registers the built-in ones."""

from evalkit.providers.anthropic import AnthropicProvider
from evalkit.providers.base import (
    Completion,
    Provider,
    ProviderConfig,
    create_provider,
    provider_registry,
    register_provider,
)
from evalkit.providers.fake import FakeProvider
from evalkit.providers.openai_compat import OpenAICompatibleProvider

__all__ = [
    "AnthropicProvider",
    "Completion",
    "FakeProvider",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderConfig",
    "create_provider",
    "provider_registry",
    "register_provider",
]

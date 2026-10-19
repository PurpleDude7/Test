"""Shared plumbing for providers that talk JSON over HTTPS."""

from __future__ import annotations

import os
import threading
import time
from typing import Any

import httpx
from pydantic import Field

from evalkit.errors import ProviderError
from evalkit.providers.base import Provider, ProviderConfig

_RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})


class HTTPProviderConfig(ProviderConfig):
    base_url: str
    api_key: str | None = None
    """Literal key. Prefer ``api_key_env`` so secrets stay out of suite files."""
    api_key_env: str
    timeout_s: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)
    headers: dict[str, str] = Field(default_factory=dict)


class HTTPProvider(Provider):
    """Owns an ``httpx.Client`` and retries transient failures with backoff."""

    config: HTTPProviderConfig

    def __init__(
        self, config: HTTPProviderConfig, *, transport: httpx.BaseTransport | None = None
    ) -> None:
        super().__init__(config)
        self._transport = transport
        self._client: httpx.Client | None = None
        self._client_lock = threading.Lock()

    @property
    def api_key(self) -> str:
        key = self.config.api_key or os.environ.get(self.config.api_key_env)
        if not key:
            raise ProviderError(
                f"provider {self.name!r} needs an API key: set ${self.config.api_key_env}"
            )
        return key

    def auth_headers(self) -> dict[str, str]:
        raise NotImplementedError

    @property
    def client(self) -> httpx.Client:
        with self._client_lock:
            if self._client is None:
                self._client = httpx.Client(
                    base_url=self.config.base_url.rstrip("/"),
                    timeout=self.config.timeout_s,
                    transport=self._transport,
                )
            return self._client

    def post_json(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        headers = {**self.auth_headers(), **self.config.headers}
        attempt = 0
        while True:
            try:
                response = self.client.post(path, json=payload, headers=headers)
            except httpx.TransportError as exc:
                if attempt >= self.config.max_retries:
                    raise ProviderError(f"{self.name}: request failed: {exc}") from exc
            else:
                if response.status_code < 400:
                    try:
                        body: dict[str, Any] = response.json()
                    except ValueError as exc:
                        raise ProviderError(f"{self.name}: response was not JSON") from exc
                    return body
                if (
                    response.status_code not in _RETRY_STATUSES
                    or attempt >= self.config.max_retries
                ):
                    raise ProviderError(
                        f"{self.name}: HTTP {response.status_code}: {response.text[:500]}"
                    )
            attempt += 1
            time.sleep(min(2.0**attempt * 0.5, 8.0))

    def close(self) -> None:
        with self._client_lock:
            if self._client is not None:
                self._client.close()
                self._client = None

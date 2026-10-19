"""On-disk response cache, so re-running a suite doesn't pay for identical requests twice."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from evalkit.models import Message
from evalkit.providers.base import Completion, Provider


class ResponseCache:
    """Completions stored as one JSON file per request under ``directory``.

    Files are written atomically (temp file + rename), so concurrent runs and threads can
    share a directory. Unreadable entries count as misses and are overwritten.
    """

    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    @staticmethod
    def key(
        identity: dict[str, Any],
        messages: Sequence[Message],
        temperature: float | None,
        max_tokens: int | None,
    ) -> str:
        payload = {
            "v": 1,
            "identity": identity,
            "messages": [m.model_dump() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def get(self, key: str) -> Completion | None:
        try:
            return Completion.model_validate_json(self._path(key).read_bytes())
        except (OSError, ValidationError):
            return None

    def put(self, key: str, completion: Completion) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = completion.model_copy(update={"cached": False}).model_dump_json()
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".json")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(data)
            Path(tmp).replace(path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise


class CachingProvider(Provider):
    """Wraps a provider and serves repeated ``complete()`` requests from a ResponseCache.

    Only successful completions are stored; errors always reach the real provider again.
    Embeddings pass straight through.
    """

    def __init__(self, inner: Provider, cache: ResponseCache) -> None:
        super().__init__(inner.config)
        self.name = inner.name
        self.inner = inner
        self.cache = cache

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        key = self.cache.key(self.inner.cache_identity(), messages, temperature, max_tokens)
        hit = self.cache.get(key)
        if hit is not None:
            return hit.model_copy(update={"cached": True})
        completion = self.inner.complete(messages, temperature=temperature, max_tokens=max_tokens)
        self.cache.put(key, completion)
        return completion

    def cache_identity(self) -> dict[str, Any]:
        return self.inner.cache_identity()

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return self.inner.embed(texts)

    def close(self) -> None:
        self.inner.close()

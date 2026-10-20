"""Semantic similarity between the output and a reference text."""

from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

from evalkit.errors import SuiteError
from evalkit.results import ScoreResult
from evalkit.scorers.base import ScoreContext, Scorer, register_scorer

_local_models: dict[str, Any] = {}
_local_lock = threading.Lock()


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"vector size mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


def _local_embed(model_name: str, texts: list[str]) -> list[list[float]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise SuiteError(
            "embedding_similarity with backend 'sentence-transformers' needs the optional "
            "extra: pip install 'evalkit[embeddings]'"
        ) from exc
    with _local_lock:
        model = _local_models.get(model_name)
        if model is None:
            model = _local_models[model_name] = SentenceTransformer(model_name)
    vectors = model.encode(texts)
    return [list(map(float, v)) for v in vectors]


@register_scorer("embedding_similarity")
class EmbeddingSimilarity(Scorer):
    """Cosine similarity of embeddings, passing at or above ``threshold``.

    ``backend: provider`` asks a provider's ``embed`` method (the one under test unless
    ``provider`` is set). ``backend: sentence-transformers`` runs a local model and needs
    ``pip install 'evalkit[embeddings]'``.
    """

    name = "embedding_similarity"
    reference_text: str | None = Field(default=None, alias="reference")
    threshold: float = Field(default=0.8, ge=-1, le=1)
    backend: Literal["provider", "sentence-transformers"] = "provider"
    provider: str | None = None
    model: str | None = None
    """Embedding model: a provider option for ``provider``, a model id for local runs."""

    def score(self, ctx: ScoreContext) -> ScoreResult:
        reference = self.resolve_reference(self.reference_text, ctx, "reference")
        texts = [ctx.output, reference]
        if self.backend == "sentence-transformers":
            vectors = _local_embed(self.model or "all-MiniLM-L6-v2", texts)
        else:
            overrides = {"embedding_model": self.model} if self.model else {}
            vectors = ctx.get_provider(self.provider, overrides).embed(texts)
        sim = cosine(vectors[0], vectors[1])
        ok = sim >= self.threshold
        # Cosine lives in [-1, 1]; clamp to [0, 1] for the normalised score.
        return self.result(sim, ok, f"cosine={sim:.3f} (threshold {self.threshold})")

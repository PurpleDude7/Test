"""High-level entry points: analyze a lecture, ask it a question, build an engine."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from evalkit.cache import CachingProvider, ResponseCache
from evalkit.providers import create_provider
from lectureiq.engines import Engine, PackOptions
from lectureiq.llm import LLMEngine
from lectureiq.models import Answer, StudyPack
from lectureiq.offline import OfflineEngine
from lectureiq.retrieval import BM25Index
from lectureiq.transcript import Lecture, chunk_lecture

DEFAULT_CHUNK_WORDS = 300


def make_engine(
    provider: str = "offline",
    config: Mapping[str, Any] | None = None,
    *,
    cache_dir: str | Path | None = None,
    concurrency: int = 4,
) -> Engine:
    """``"offline"`` for the extractive engine, else any evalkit provider name."""
    if provider == "offline":
        return OfflineEngine()
    backend = create_provider(provider, config)
    if cache_dir is not None:
        backend = CachingProvider(backend, ResponseCache(cache_dir))
    return LLMEngine(backend, concurrency=concurrency)


def _ensure_chunks(lecture: Lecture, chunk_words: int) -> None:
    if not lecture.chunks:
        chunk_lecture(lecture, chunk_words)


def analyze(
    lecture: Lecture,
    engine: Engine,
    *,
    options: PackOptions | None = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> StudyPack:
    """Chunk the lecture (once) and generate its study pack."""
    _ensure_chunks(lecture, chunk_words)
    return engine.study_pack(lecture, lecture.chunks, options or PackOptions())


def ask(
    lecture: Lecture,
    question: str,
    engine: Engine,
    *,
    k: int = 4,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
) -> Answer:
    """Retrieve the ``k`` most relevant chunks with BM25 and let the engine answer."""
    question = question.strip()
    if not question:
        raise ValueError("question is empty")
    _ensure_chunks(lecture, chunk_words)
    passages = [c for c, _ in BM25Index(lecture.chunks).search(question, k)]
    if not passages:
        return Answer(
            question=question,
            answer="The lecture doesn't seem to cover that.",
            grounded=False,
        )
    return engine.answer(question, passages)

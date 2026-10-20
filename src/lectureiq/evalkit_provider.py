"""LectureIQ as an evalkit provider, so evalkit suites can grade its answers.

Each case's last user message is treated as a question about one lecture; the reply is
LectureIQ's answer, followed by a "Sources:" line of timestamps when it cites any. Point
any engine at it (offline, or an LLM through the usual evalkit providers):

    providers:
      lectureiq: {transcript: "@sample", engine: offline}
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from evalkit.errors import ProviderError
from evalkit.models import Message
from evalkit.providers.base import Completion, Provider, ProviderConfig, register_provider
from lectureiq.engines import Engine
from lectureiq.pipeline import DEFAULT_CHUNK_WORDS, ask, make_engine
from lectureiq.sample import SAMPLE_TITLE, sample_transcript
from lectureiq.transcript import Lecture, chunk_lecture, format_ts, parse_transcript

SAMPLE = "@sample"
_SECRET_HINTS = ("key", "token", "secret", "password", "authorization")


class LectureIQConfig(ProviderConfig):
    transcript: str = SAMPLE
    """Transcript path (relative to the working directory), or "@sample"."""
    engine: str = "offline"
    """"offline", or an evalkit provider name for the LLM engine (openai, anthropic, ...)."""
    engine_options: dict[str, Any] = Field(default_factory=dict)
    chunk_words: int = Field(default=DEFAULT_CHUNK_WORDS, ge=20)
    k: int = Field(default=4, ge=1, description="chunks retrieved per question")
    cite: bool = True
    """Append a "Sources: 7:12, 8:38" line so suites can check that answers are grounded."""


@register_provider("lectureiq")
class LectureIQProvider(Provider):
    name = "lectureiq"
    Config = LectureIQConfig
    config: LectureIQConfig

    def __init__(self, config: LectureIQConfig) -> None:
        super().__init__(config)
        self._lecture: Lecture | None = None
        self._engine: Engine | None = None
        self._lock = threading.Lock()

    def _setup(self) -> tuple[Lecture, Engine]:
        """Load and chunk the lecture once, on first use (the runner calls from threads)."""
        with self._lock:
            if self._lecture is None or self._engine is None:
                cfg = self.config
                if cfg.transcript == SAMPLE:
                    lecture = parse_transcript(sample_transcript(), title=SAMPLE_TITLE)
                else:
                    path = Path(cfg.transcript)
                    try:
                        text = path.read_text(encoding="utf-8")
                    except OSError as exc:
                        raise ProviderError(f"lectureiq: cannot read {path}: {exc}") from exc
                    lecture = parse_transcript(text, title=path.stem)
                chunk_lecture(lecture, cfg.chunk_words)
                options = dict(cfg.engine_options)
                if cfg.model and cfg.engine != "offline":
                    options["model"] = cfg.model
                self._engine = make_engine(cfg.engine, options)
                self._lecture = lecture
            return self._lecture, self._engine

    def complete(
        self,
        messages: Sequence[Message],
        *,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> Completion:
        question = next((m.content for m in reversed(messages) if m.role == "user"), "").strip()
        if not question:
            raise ProviderError("lectureiq: the case has no question")
        lecture, engine = self._setup()
        answer = ask(lecture, question, engine, k=self.config.k)
        text = answer.answer
        stamps = [format_ts(c.start) for c in answer.citations if c.start is not None]
        if self.config.cite and stamps:
            text += "\n\nSources: " + ", ".join(dict.fromkeys(stamps))
        return Completion(text=text, model=self.model or engine.name)

    def cache_identity(self) -> dict[str, Any]:
        cfg = self.config
        options = {
            k: v
            for k, v in cfg.engine_options.items()
            if not any(hint in k.lower() for hint in _SECRET_HINTS)
        }
        return {
            "provider": self.name,
            "model": cfg.model,
            "transcript": cfg.transcript,
            "engine": cfg.engine,
            "engine_options": options,
            "chunk_words": cfg.chunk_words,
            "k": cfg.k,
            "cite": cfg.cite,
        }

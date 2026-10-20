"""The Engine interface: what turns chunks into a StudyPack and passages into an Answer."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from lectureiq.models import Answer, StudyPack
from lectureiq.transcript import Chunk, Lecture


@dataclass(frozen=True)
class PackOptions:
    """Caps on how much material to produce."""

    max_key_points: int = 10
    max_terms: int = 25
    max_flashcards: int = 20
    max_quiz: int = 10


class Engine(Protocol):
    name: str

    def study_pack(
        self, lecture: Lecture, chunks: Sequence[Chunk], options: PackOptions
    ) -> StudyPack:
        """Build study material for the whole lecture."""
        ...

    def answer(self, question: str, passages: Sequence[Chunk]) -> Answer:
        """Answer ``question`` from ``passages`` (already retrieved, best first)."""
        ...

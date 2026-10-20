"""The study material LectureIQ produces. Every item points back to its source chunk."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator


class Section(BaseModel):
    heading: str
    summary: str
    chunk_ids: list[int] = Field(default_factory=list)
    start: float | None = None


class KeyPoint(BaseModel):
    text: str
    chunk_id: int | None = None
    start: float | None = None
    """When it was said, if known more precisely than the chunk's start."""


class Term(BaseModel):
    term: str
    definition: str
    chunk_id: int | None = None
    start: float | None = None
    """When it was said, if known more precisely than the chunk's start."""


class Flashcard(BaseModel):
    front: str
    back: str
    chunk_id: int | None = None
    start: float | None = None
    """When it was said, if known more precisely than the chunk's start."""


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(min_length=2, max_length=6)
    answer: int = Field(ge=0, description="index of the correct option")
    explanation: str = ""
    chunk_id: int | None = None
    start: float | None = None
    """When it was said, if known more precisely than the chunk's start."""

    @model_validator(mode="after")
    def _answer_in_range(self) -> QuizQuestion:
        if self.answer >= len(self.options):
            raise ValueError(f"answer index {self.answer} out of range")
        return self


class StudyPack(BaseModel):
    """Everything generated for one lecture."""

    title: str
    tldr: str
    sections: list[Section] = Field(default_factory=list)
    key_points: list[KeyPoint] = Field(default_factory=list)
    terms: list[Term] = Field(default_factory=list)
    flashcards: list[Flashcard] = Field(default_factory=list)
    quiz: list[QuizQuestion] = Field(default_factory=list)
    engine: str = Field(description="what generated the pack, e.g. 'offline' or 'anthropic'")
    warnings: list[str] = Field(default_factory=list)


class Citation(BaseModel):
    chunk_id: int
    start: float | None = None
    quote: str = ""


class Answer(BaseModel):
    question: str
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    grounded: bool = Field(
        default=True, description="False when the lecture doesn't seem to cover the question"
    )

    @field_validator("answer")
    @classmethod
    def _strip(cls, value: str) -> str:
        return value.strip()

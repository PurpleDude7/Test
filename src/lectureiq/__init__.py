"""LectureIQ: turn lecture transcripts into study material, with or without an LLM."""

from lectureiq.models import Answer, Citation, Flashcard, QuizQuestion, StudyPack, Term
from lectureiq.pipeline import analyze, ask
from lectureiq.transcript import Chunk, Lecture, Segment, chunk_lecture, parse_transcript

__version__ = "0.1.0"

__all__ = [
    "Answer",
    "Chunk",
    "Citation",
    "Flashcard",
    "Lecture",
    "QuizQuestion",
    "Segment",
    "StudyPack",
    "Term",
    "__version__",
    "analyze",
    "ask",
    "chunk_lecture",
    "parse_transcript",
]

"""LLM engine: map each chunk to notes, then reduce the summaries into an outline.

Talks to any evalkit Provider (OpenAI-compatible, Anthropic, fake, or your own). Model
replies are JSON validated with pydantic; an invalid reply gets one repair attempt, a
chunk that still fails is skipped with a warning, and bad individual items are dropped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from pydantic import BaseModel, Field, ValidationError

from evalkit.errors import EvalkitError
from evalkit.models import Message
from evalkit.providers.base import Provider
from lectureiq.engines import PackOptions
from lectureiq.errors import GenerationError
from lectureiq.models import (
    Answer,
    Citation,
    Flashcard,
    KeyPoint,
    QuizQuestion,
    Section,
    StudyPack,
    Term,
)
from lectureiq.transcript import Chunk, Lecture

M = TypeVar("M", bound=BaseModel)
T = TypeVar("T")

SYSTEM = (
    "You are LectureIQ, a precise teaching assistant that turns lecture transcripts into "
    "study material. Use only the lecture text you are given; never add outside facts. "
    "Reply with a single JSON object and nothing else."
)

CHUNK_PROMPT = """\
TASK: chunk_notes
Extract study material from lecture excerpt {chunk_id}.

<excerpt id="{chunk_id}">
{text}
</excerpt>

Return JSON with exactly these keys:
{{
  "summary": "2-3 sentence summary of the excerpt",
  "key_points": ["up to 4 important takeaways, one sentence each"],
  "terms": [{{"term": "...", "definition": "one-sentence definition as used in the lecture"}}],
  "flashcards": [{{"front": "question or prompt", "back": "concise answer"}}],
  "quiz": [{{"question": "...", "options": ["A", "B", "C", "D"], "answer": 0,
            "explanation": "why the correct option is right"}}]
}}
Use at most {per_chunk} flashcards and {per_chunk} quiz questions. "answer" is the 0-based
index of the correct option. Use empty lists when the excerpt has nothing suitable.
"""

SYNTHESIS_PROMPT = """\
TASK: synthesis
Below are summaries of consecutive excerpts of one lecture, each tagged with its id.

{summaries}

Return JSON with exactly these keys:
{{
  "title": "a short descriptive lecture title",
  "tldr": "3-4 sentence overview of the whole lecture",
  "sections": [{{"heading": "...", "summary": "1-2 sentences", "chunk_ids": [1, 2]}}]
}}
Sections follow the lecture's order, cover every excerpt, and group related excerpts.
"""

ANSWER_PROMPT = """\
TASK: answer
Answer the student's question using only these lecture excerpts.

{excerpts}

Question: {question}

Return JSON: {{"answer": "...", "citations": [excerpt ids you used], "grounded": true}}.
If the excerpts don't answer the question, set "grounded" to false and say so briefly.
"""


class _ChunkNotes(BaseModel):
    summary: str
    key_points: list[str] = Field(default_factory=list)
    terms: list[dict[str, Any]] = Field(default_factory=list)
    flashcards: list[dict[str, Any]] = Field(default_factory=list)
    quiz: list[dict[str, Any]] = Field(default_factory=list)


class _SectionDraft(BaseModel):
    heading: str
    summary: str
    chunk_ids: list[int] = Field(default_factory=list)


class _Synthesis(BaseModel):
    title: str
    tldr: str
    sections: list[_SectionDraft] = Field(default_factory=list)


class _AnswerDraft(BaseModel):
    answer: str
    citations: list[int] = Field(default_factory=list)
    grounded: bool = True


_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_CITATION = re.compile(r"\[(\d+)\]")


def parse_json(text: str, model: type[M]) -> M:
    """Validate the first JSON object in ``text`` (models like to wrap it in prose/fences)."""
    match = _JSON_OBJECT.search(text)
    if not match:
        raise ValueError("no JSON object in reply")
    return model.model_validate(json.loads(match.group(0)))


def _round_robin(groups: Iterable[list[T]]) -> list[T]:
    """Interleave per-chunk lists so capped output still covers the whole lecture."""
    pending = [list(g) for g in groups]
    out: list[T] = []
    while any(pending):
        for group in pending:
            if group:
                out.append(group.pop(0))
    return out


def _items(raw: list[dict[str, Any]], model: type[M], chunk_id: int, dropped: list[str]) -> list[M]:
    items = []
    for entry in raw:
        try:
            items.append(model.model_validate({**entry, "chunk_id": chunk_id}))
        except ValidationError:
            dropped.append(model.__name__)
    return items


class LLMEngine:
    """Study material from a chat model. ``concurrency`` chunks are processed at once."""

    def __init__(self, provider: Provider, *, concurrency: int = 4) -> None:
        self.provider = provider
        self.name = provider.name
        self.concurrency = max(1, concurrency)

    def _call(self, prompt: str, model: type[M], max_tokens: int) -> M:
        messages = [Message(role="system", content=SYSTEM), Message(role="user", content=prompt)]
        reply = self.provider.complete(messages, max_tokens=max_tokens).text
        try:
            return parse_json(reply, model)
        except (ValueError, ValidationError) as first_error:
            messages += [
                Message(role="assistant", content=reply),
                Message(
                    role="user",
                    content=f"That reply was not valid ({first_error!s:.300}). "
                    "Reply again with only the corrected JSON object.",
                ),
            ]
            retry = self.provider.complete(messages, max_tokens=max_tokens).text
            try:
                return parse_json(retry, model)
            except (ValueError, ValidationError) as exc:
                raise GenerationError(f"model returned invalid JSON twice: {exc!s:.300}") from exc

    def _map(self, fn: Callable[[Chunk], T], chunks: Sequence[Chunk]) -> list[T]:
        with ThreadPoolExecutor(max_workers=self.concurrency) as pool:
            return list(pool.map(fn, chunks))

    def study_pack(
        self, lecture: Lecture, chunks: Sequence[Chunk], options: PackOptions
    ) -> StudyPack:
        per_chunk = max(2, -(-options.max_quiz // max(1, len(chunks))) + 1)

        def notes_for(chunk: Chunk) -> _ChunkNotes | str:
            prompt = CHUNK_PROMPT.format(chunk_id=chunk.id, text=chunk.text, per_chunk=per_chunk)
            try:
                return self._call(prompt, _ChunkNotes, max_tokens=1500)
            except EvalkitError as exc:  # provider failures and GenerationError alike
                return f"excerpt {chunk.id} skipped: {exc}"

        results = self._map(notes_for, chunks)
        warnings = [r for r in results if isinstance(r, str)]
        notes = [(c, r) for c, r in zip(chunks, results, strict=True) if not isinstance(r, str)]
        if not notes:
            raise GenerationError("no excerpt could be processed: " + "; ".join(warnings[:3]))

        dropped: list[str] = []
        terms: dict[str, Term] = {}
        for chunk, n in notes:
            for term in _items(n.terms, Term, chunk.id, dropped):
                terms.setdefault(term.term.strip().lower(), term)
        key_points = _round_robin(
            [KeyPoint(text=p, chunk_id=c.id) for p in n.key_points if p.strip()] for c, n in notes
        )
        flashcards = _round_robin(_items(n.flashcards, Flashcard, c.id, dropped) for c, n in notes)
        quiz = _round_robin(_items(n.quiz, QuizQuestion, c.id, dropped) for c, n in notes)
        if dropped:
            warnings.append(f"dropped {len(dropped)} malformed item(s) from the model's output")

        title, tldr, sections = self._synthesize(lecture, chunks, notes, warnings)
        return StudyPack(
            title=title,
            tldr=tldr,
            sections=sections,
            key_points=key_points[: options.max_key_points],
            terms=list(terms.values())[: options.max_terms],
            flashcards=flashcards[: options.max_flashcards],
            quiz=quiz[: options.max_quiz],
            engine=self.name,
            warnings=warnings,
        )

    def _synthesize(
        self,
        lecture: Lecture,
        chunks: Sequence[Chunk],
        notes: list[tuple[Chunk, _ChunkNotes]],
        warnings: list[str],
    ) -> tuple[str, str, list[Section]]:
        starts = {c.id: c.start for c in chunks}
        summaries = "\n".join(f"[{c.id}] {n.summary}" for c, n in notes)
        try:
            synthesis = self._call(
                SYNTHESIS_PROMPT.format(summaries=summaries), _Synthesis, max_tokens=1200
            )
        except EvalkitError as exc:
            warnings.append(f"outline fell back to per-excerpt summaries: {exc}")
            sections = [
                Section(heading=f"Part {c.id}", summary=n.summary, chunk_ids=[c.id], start=c.start)
                for c, n in notes
            ]
            return lecture.title, " ".join(n.summary for _, n in notes[:3]), sections
        sections = []
        for draft in synthesis.sections:
            ids = [i for i in draft.chunk_ids if i in starts]
            known = [s for i in ids if (s := starts[i]) is not None]
            sections.append(
                Section(
                    heading=draft.heading,
                    summary=draft.summary,
                    chunk_ids=ids,
                    start=min(known) if known else None,
                )
            )
        # Keep the user's title if they gave one; otherwise take the model's.
        title = lecture.title if lecture.title != "Untitled lecture" else synthesis.title
        return title, synthesis.tldr, sections

    def answer(self, question: str, passages: Sequence[Chunk]) -> Answer:
        by_id = {c.id: c for c in passages}
        excerpts = "\n\n".join(f'<excerpt id="{c.id}">\n{c.text}\n</excerpt>' for c in passages)
        prompt = ANSWER_PROMPT.format(excerpts=excerpts, question=question)
        messages = [Message(role="system", content=SYSTEM), Message(role="user", content=prompt)]
        reply = self.provider.complete(messages, max_tokens=600).text
        try:
            draft = parse_json(reply, _AnswerDraft)
        except (ValueError, ValidationError):
            # Fall back to the raw text and any [n] markers in it.
            cited = [int(n) for n in _CITATION.findall(reply)]
            draft = _AnswerDraft(answer=reply, citations=cited)
        ids = list(dict.fromkeys(i for i in draft.citations if i in by_id))
        return Answer(
            question=question,
            answer=draft.answer,
            citations=[Citation(chunk_id=i, start=by_id[i].start) for i in ids],
            grounded=draft.grounded,
        )

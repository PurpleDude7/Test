"""Extractive engine: study material straight from the transcript, no model or API key.

It is a heuristic baseline, not a substitute for an LLM: summaries are the most
representative sentences, glossary entries come from definitional sentences ("X is ..."),
and quiz distractors are other definitions from the same lecture.
"""

from __future__ import annotations

import math
import random
import re
import zlib
from collections import Counter
from collections.abc import Sequence

from lectureiq.engines import PackOptions
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
from lectureiq.text import STOPWORDS, content_words, keywords, sentences
from lectureiq.transcript import Chunk, Lecture

_DEFINITION = re.compile(
    r"^(?:(?:so|now|and|well|okay|remember)[, ]+)*(?:an?\s+|the\s+)?"
    r"(?P<term>[A-Za-z][\w\-]*(?:\s+[\w\-]+){0,3}?)\s*,?\s+"
    r"(?:is|are|refers to|is defined as|means|describes)\s+"
    r"(?P<definition>.{12,})$",
    re.IGNORECASE,
)
_NOT_TERMS = STOPWORDS | {"everything", "something", "nothing", "anything", "answer", "idea"}
_MAX_POINT_WORDS = 45
_MIN_DEFINITION_WORDS = 4
_DEFINITION_BONUS = 1.3  # definitional sentences make better summaries and key points
_TLDR_SENTENCES = 4
_DEFINES_TERM_BONUS = 2.0


def _rng(seed: str) -> random.Random:
    """Deterministic per-item randomness, so the same lecture always yields the same quiz."""
    return random.Random(zlib.crc32(seed.encode()))


class OfflineEngine:
    name = "offline"

    def study_pack(
        self, lecture: Lecture, chunks: Sequence[Chunk], options: PackOptions
    ) -> StudyPack:
        freq = Counter(content_words(lecture.text))
        by_id = {c.id: c for c in chunks}
        scored = [
            (self._score(s, freq), chunk.id, s)
            for chunk in chunks
            for s in sentences(chunk.text)
            if len(s.split()) <= _MAX_POINT_WORDS
        ]
        best = sorted(scored, key=lambda t: t[0], reverse=True)
        order = {id(t): i for i, t in enumerate(scored)}

        def in_order(items: list[tuple[float, int, str]]) -> list[tuple[float, int, str]]:
            return sorted(items, key=lambda t: order[id(t)])

        key_points = [
            KeyPoint(text=s, chunk_id=cid, start=by_id[cid].time_of(s))
            for _, cid, s in in_order(best[: options.max_key_points])
        ]
        # One leading sentence per part of the lecture reads better than the global top 3.
        per_chunk: dict[int, tuple[float, int, str]] = {}
        for item in best:
            per_chunk.setdefault(item[1], item)
        leads = sorted(per_chunk.values(), key=lambda t: t[0], reverse=True)[:_TLDR_SENTENCES]
        tldr = " ".join(s for _, _, s in in_order(leads)) or lecture.text[:400]
        terms = self._terms(chunks)[: options.max_terms]
        return StudyPack(
            title=lecture.title,
            tldr=tldr,
            sections=[self._section(c, freq) for c in chunks],
            key_points=key_points,
            terms=terms,
            flashcards=self._flashcards(terms, key_points, freq)[: options.max_flashcards],
            quiz=self._quiz(terms, key_points, freq)[: options.max_quiz],
            engine=self.name,
            warnings=[] if terms else ["No definitional sentences found; glossary is empty."],
        )

    def answer(self, question: str, passages: Sequence[Chunk]) -> Answer:
        wanted = set(content_words(question))
        ranked = sorted(
            (
                (self._relevance(s, wanted), -rank, chunk, s)
                for rank, chunk in enumerate(passages)
                for s in sentences(chunk.text) or [chunk.text]
            ),
            key=lambda t: (t[0], t[1]),
            reverse=True,
        )
        hits = [(chunk, s) for relevance, _, chunk, s in ranked[:2] if relevance]
        if not hits:
            return Answer(
                question=question,
                answer="The lecture doesn't seem to cover that.",
                grounded=False,
            )
        citations: dict[int, Citation] = {}
        for chunk, s in hits:
            citations.setdefault(
                chunk.id, Citation(chunk_id=chunk.id, start=chunk.time_of(s), quote=s)
            )
        return Answer(
            question=question,
            answer=" ".join(s for _, s in hits),
            citations=list(citations.values()),
        )

    @staticmethod
    def _relevance(sentence: str, wanted: set[str]) -> float:
        """Word overlap with the question, plus a bonus when the sentence defines the very
        term asked about ("What is gradient descent?" -> "Gradient descent is ...")."""
        overlap = len(wanted & set(content_words(sentence)))
        match = _DEFINITION.match(sentence.strip())
        if overlap and match:
            term = set(content_words(match["term"]))
            if term and term <= wanted:
                return overlap + _DEFINES_TERM_BONUS
        return float(overlap)

    @staticmethod
    def _score(sentence: str, freq: Counter[str]) -> float:
        tokens = content_words(sentence)
        if not tokens:
            return 0.0
        score = sum(freq[t] for t in set(tokens)) / math.pow(len(tokens), 0.6)
        return score * _DEFINITION_BONUS if _DEFINITION.match(sentence) else score

    @staticmethod
    def _section(chunk: Chunk, freq: Counter[str]) -> Section:
        top = keywords(chunk.text, 3)
        ranked = sorted(sentences(chunk.text), key=lambda s: OfflineEngine._score(s, freq))
        summary = " ".join(ranked[-2:]) if ranked else chunk.text[:300]
        return Section(
            heading=" · ".join(w.capitalize() for w in top) or f"Part {chunk.id}",
            summary=summary,
            chunk_ids=[chunk.id],
            start=chunk.start,
        )

    @staticmethod
    def _terms(chunks: Sequence[Chunk]) -> list[Term]:
        found: dict[str, Term] = {}
        for chunk in chunks:
            for sentence in sentences(chunk.text):
                match = _DEFINITION.match(sentence.strip())
                if not match:
                    continue
                term = match["term"].strip()
                first = term.split()[0].lower()
                if first in _NOT_TERMS or all(w.lower() in _NOT_TERMS for w in term.split()):
                    continue
                definition = match["definition"].strip().rstrip(".")
                if len(definition.split()) < _MIN_DEFINITION_WORDS:
                    continue
                key = term.lower()
                if key not in found:
                    found[key] = Term(
                        term=term[0].upper() + term[1:],
                        definition=definition[0].upper() + definition[1:],
                        chunk_id=chunk.id,
                        start=chunk.time_of(sentence),
                    )
        return list(found.values())

    @staticmethod
    def _blank(sentence: str, freq: Counter[str]) -> tuple[str, str] | None:
        """Replace the sentence's most lecture-relevant word with a blank."""
        candidates = sorted(set(content_words(sentence)), key=lambda w: freq[w], reverse=True)
        for word in candidates:
            pattern = re.compile(rf"\b{re.escape(word)}\b", re.IGNORECASE)
            match = pattern.search(sentence)
            if match:
                return pattern.sub("_____", sentence, count=1), match.group(0)
        return None

    def _flashcards(
        self, terms: list[Term], points: list[KeyPoint], freq: Counter[str]
    ) -> list[Flashcard]:
        cards = [
            Flashcard(
                front=f"What is meant by “{t.term}”?",
                back=t.definition,
                chunk_id=t.chunk_id,
                start=t.start,
            )
            for t in terms
        ]
        for point in points:
            blanked = self._blank(point.text, freq)
            if blanked:
                cards.append(
                    Flashcard(
                        front=f"Fill in the blank: {blanked[0]}",
                        back=blanked[1],
                        chunk_id=point.chunk_id,
                        start=point.start,
                    )
                )
        return cards

    def _quiz(
        self, terms: list[Term], points: list[KeyPoint], freq: Counter[str]
    ) -> list[QuizQuestion]:
        quiz: list[QuizQuestion] = []
        for term in terms if len(terms) >= 2 else []:
            rng = _rng(term.term)
            others = [t.definition for t in terms if t is not term]
            options = [term.definition, *rng.sample(others, min(3, len(others)))]
            rng.shuffle(options)
            quiz.append(
                QuizQuestion(
                    question=f"Which statement best describes “{term.term}”?",
                    options=options,
                    answer=options.index(term.definition),
                    explanation=f"{term.term}: {term.definition}.",
                    chunk_id=term.chunk_id,
                    start=term.start,
                )
            )
        vocabulary = [w for w, _ in freq.most_common(40)]
        for point in points:
            blanked = self._blank(point.text, freq)
            if blanked is None:
                continue
            sentence, word = blanked
            rng = _rng(sentence)
            pool = [w for w in vocabulary if w != word.lower()]
            options = [word, *rng.sample(pool, min(3, len(pool)))]
            if len(options) < 2:
                continue
            rng.shuffle(options)
            quiz.append(
                QuizQuestion(
                    question=f"Complete the statement: {sentence}",
                    options=options,
                    answer=options.index(word),
                    explanation=point.text,
                    chunk_id=point.chunk_id,
                    start=point.start,
                )
            )
        return quiz

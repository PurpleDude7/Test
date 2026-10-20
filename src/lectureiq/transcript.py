"""Parse transcripts (plain text, SRT, WebVTT) and split them into timestamped chunks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from lectureiq.errors import TranscriptError

MAX_CHARS = 2_000_000

_TIMESTAMP = r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})"
_CUE_TIMING = re.compile(rf"^\s*{_TIMESTAMP}\s*-->\s*{_TIMESTAMP}")
_TAG = re.compile(r"<[^>]+>")
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Segment:
    """A run of transcript text, with its time span when the source had one."""

    text: str
    start: float | None = None
    end: float | None = None


@dataclass(frozen=True)
class Chunk:
    """A retrieval/generation unit: a few hundred words of consecutive segments."""

    id: int
    text: str
    start: float | None = None
    end: float | None = None
    segments: tuple[Segment, ...] = ()

    def time_of(self, quote: str) -> float | None:
        """Start time of the segment a quote comes from, falling back to the chunk start."""
        probe = " ".join(quote.split())[:60]
        for segment in self.segments:
            if segment.start is not None and (probe in segment.text or segment.text in quote):
                return segment.start
        return self.start


@dataclass
class Lecture:
    title: str
    segments: list[Segment]
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments)

    @property
    def word_count(self) -> int:
        return len(self.text.split())

    @property
    def has_timestamps(self) -> bool:
        return any(s.start is not None for s in self.segments)


def format_ts(seconds: float) -> str:
    """``75.0 -> "1:15"``, ``3725.0 -> "1:02:05"``."""
    total = int(seconds)
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"


def _seconds(groups: tuple[str | None, ...]) -> float:
    hours, minutes, secs, millis = groups
    return (
        int(hours or 0) * 3600 + int(minutes or 0) * 60 + int(secs or 0) + int(millis or 0) / 1000
    )


def _parse_cues(text: str) -> list[Segment]:
    """SRT and WebVTT share the shape that matters: a timing line followed by text lines."""
    segments: list[Segment] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        match = _CUE_TIMING.match(lines[i])
        if not match:
            i += 1
            continue
        start, end = _seconds(match.groups()[:4]), _seconds(match.groups()[4:])
        i += 1
        body: list[str] = []
        while i < len(lines) and lines[i].strip():
            body.append(_TAG.sub("", lines[i]).strip())
            i += 1
        cue = " ".join(b for b in body if b)
        if cue:
            segments.append(Segment(text=cue, start=start, end=end))
    return segments


def _parse_plain(text: str) -> list[Segment]:
    paragraphs = [" ".join(p.split()) for p in re.split(r"\n\s*\n", text)]
    return [Segment(text=p) for p in paragraphs if p]


def parse_transcript(text: str, *, title: str = "Untitled lecture") -> Lecture:
    """Detect the format from the content, not the file name, and parse it."""
    if len(text) > MAX_CHARS:
        raise TranscriptError(f"transcript is too large ({len(text):,} chars; max {MAX_CHARS:,})")
    text = text.lstrip("﻿")
    is_cued = any(_CUE_TIMING.match(line) for line in text.splitlines())
    segments = _parse_cues(text) if is_cued else _parse_plain(text)
    if not segments:
        raise TranscriptError("transcript has no text")
    return Lecture(title=title.strip() or "Untitled lecture", segments=segments)


def _split_long(segment: Segment, max_words: int) -> list[Segment]:
    """Break an oversized segment (e.g. one huge paragraph) at sentence boundaries."""
    if len(segment.text.split()) <= max_words:
        return [segment]
    pieces: list[Segment] = []
    current: list[str] = []
    for sentence in _SENTENCE_END.split(segment.text):
        if current and len(" ".join([*current, sentence]).split()) > max_words:
            pieces.append(Segment(" ".join(current), segment.start, segment.end))
            current = []
        current.append(sentence)
    if current:
        pieces.append(Segment(" ".join(current), segment.start, segment.end))
    return pieces


def chunk_lecture(lecture: Lecture, max_words: int = 300) -> list[Chunk]:
    """Group consecutive segments into chunks of at most ``max_words`` words (ids from 1)."""
    if max_words < 20:
        raise ValueError("max_words must be at least 20")
    chunks: list[Chunk] = []
    group: list[Segment] = []
    words = 0

    def flush() -> None:
        nonlocal group, words
        if group:
            starts = [s.start for s in group if s.start is not None]
            ends = [s.end for s in group if s.end is not None]
            chunks.append(
                Chunk(
                    id=len(chunks) + 1,
                    text=" ".join(s.text for s in group),
                    start=min(starts) if starts else None,
                    end=max(ends) if ends else None,
                    segments=tuple(group),
                )
            )
        group, words = [], 0

    for original in lecture.segments:
        for segment in _split_long(original, max_words):
            n = len(segment.text.split())
            if group and words + n > max_words:
                flush()
            group.append(segment)
            words += n
    flush()
    lecture.chunks = chunks
    return chunks


def _segment_dict(segment: Segment) -> dict[str, Any]:
    return {"text": segment.text, "start": segment.start, "end": segment.end}


def _segment(data: dict[str, Any]) -> Segment:
    return Segment(text=str(data["text"]), start=data.get("start"), end=data.get("end"))


def lecture_to_dict(lecture: Lecture) -> dict[str, Any]:
    """JSON-safe form of a lecture, chunks included, for persistence."""
    return {
        "title": lecture.title,
        "segments": [_segment_dict(s) for s in lecture.segments],
        "chunks": [
            {
                "id": c.id,
                "text": c.text,
                "start": c.start,
                "end": c.end,
                "segments": [_segment_dict(s) for s in c.segments],
            }
            for c in lecture.chunks
        ],
    }


def lecture_from_dict(data: dict[str, Any]) -> Lecture:
    chunks = [
        Chunk(
            id=int(c["id"]),
            text=str(c["text"]),
            start=c.get("start"),
            end=c.get("end"),
            segments=tuple(_segment(s) for s in c.get("segments", [])),
        )
        for c in data.get("chunks", [])
    ]
    return Lecture(
        title=str(data["title"]),
        segments=[_segment(s) for s in data["segments"]],
        chunks=chunks,
    )

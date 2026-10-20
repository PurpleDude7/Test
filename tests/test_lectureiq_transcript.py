from __future__ import annotations

import pytest

from lectureiq import chunk_lecture, parse_transcript
from lectureiq.errors import TranscriptError
from lectureiq.retrieval import BM25Index
from lectureiq.transcript import MAX_CHARS, Chunk, Segment, format_ts

SRT = """\
1
00:00:01,000 --> 00:00:04,500
Hello <i>world</i>.

2
01:02:05,250 --> 01:02:09,000
Second cue
over two lines.
"""

VTT = """﻿WEBVTT
Kind: captions

NOTE a comment block

intro
00:00.000 --> 00:02.500 align:start position:10%
<v Prof>Welcome back.</v>

00:03.000 --> 00:05.000

01:00.000 --> 01:04.000
Today: entropy.
"""


def test_parses_srt_with_tags_and_multiline_cues() -> None:
    lecture = parse_transcript(SRT, title="  Week 1 ")
    assert lecture.title == "Week 1"
    assert lecture.segments == [
        Segment("Hello world.", 1.0, 4.5),
        Segment("Second cue over two lines.", 3725.25, 3729.0),
    ]
    assert lecture.has_timestamps


def test_parses_vtt_with_bom_settings_voice_tags_and_empty_cues() -> None:
    lecture = parse_transcript(VTT)
    assert [(s.text, s.start) for s in lecture.segments] == [
        ("Welcome back.", 0.0),
        ("Today: entropy.", 60.0),
    ]
    assert lecture.title == "Untitled lecture"


def test_plain_text_paragraphs() -> None:
    lecture = parse_transcript("First   paragraph\nwraps here.\n\n\nSecond one.\n")
    assert [s.text for s in lecture.segments] == ["First paragraph wraps here.", "Second one."]
    assert not lecture.has_timestamps
    assert lecture.word_count == 6


@pytest.mark.parametrize("text", ["", "   \n\n ", "WEBVTT\n\n00:00.000 --> 00:01.000\n\n"])
def test_empty_transcripts_are_rejected(text: str) -> None:
    with pytest.raises(TranscriptError, match="no text"):
        parse_transcript(text)


def test_oversized_transcripts_are_rejected() -> None:
    with pytest.raises(TranscriptError, match="too large"):
        parse_transcript("x" * (MAX_CHARS + 1))


def test_format_ts() -> None:
    assert format_ts(0) == "0:00"
    assert format_ts(75.9) == "1:15"
    assert format_ts(3725) == "1:02:05"


def test_chunking_respects_word_budget_and_time_spans(sample_vtt: str) -> None:
    lecture = parse_transcript(sample_vtt)
    chunks = chunk_lecture(lecture, max_words=120)
    assert lecture.chunks == chunks
    assert [c.id for c in chunks] == list(range(1, len(chunks) + 1))
    assert len(chunks) >= 5
    assert all(len(c.text.split()) <= 120 for c in chunks)
    assert " ".join(c.text for c in chunks) == lecture.text
    starts = [c.start for c in chunks if c.start is not None]
    assert len(starts) == len(chunks) and starts == sorted(starts) and starts[0] == 2.0
    assert all(c.end is not None and c.start is not None and c.end > c.start for c in chunks)


def test_long_paragraphs_split_at_sentences() -> None:
    sentence = "This sentence has exactly eight words in it. "
    lecture = parse_transcript(sentence * 30)
    chunks = chunk_lecture(lecture, max_words=40)
    assert all(len(c.text.split()) <= 40 for c in chunks)
    assert all(c.text.endswith(".") for c in chunks)
    with pytest.raises(ValueError, match="at least 20"):
        chunk_lecture(lecture, max_words=5)


def test_time_of_finds_the_quoted_segment() -> None:
    a, b = Segment("Alpha beta gamma.", 10.0, 12.0), Segment("Delta epsilon zeta.", 20.0, 22.0)
    chunk = Chunk(id=1, text="Alpha beta gamma. Delta epsilon zeta.", start=10.0, segments=(a, b))
    assert chunk.time_of("Delta epsilon zeta.") == 20.0
    assert chunk.time_of("not here") == 10.0


def test_bm25_ranks_relevant_chunks_first(sample_vtt: str) -> None:
    lecture = parse_transcript(sample_vtt)
    chunks = chunk_lecture(lecture, max_words=120)
    index = BM25Index(chunks)
    best, score = index.search("what does the learning rate control in gradient descent", k=2)[0]
    assert "learning rate" in best.text and score > 0
    assert index.search("photosynthesis chlorophyll") == []
    assert BM25Index([]).search("anything") == []

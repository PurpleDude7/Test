from __future__ import annotations

import csv
import io
import json
from typing import Any

import pytest

from evalkit import FakeProvider
from lectureiq import analyze, ask, parse_transcript
from lectureiq.engines import PackOptions
from lectureiq.errors import GenerationError
from lectureiq.export import flashcards_csv, to_markdown
from lectureiq.llm import LLMEngine, parse_json
from lectureiq.models import QuizQuestion, StudyPack
from lectureiq.offline import OfflineEngine
from lectureiq.pipeline import make_engine


def notes(n: int, **extra: Any) -> str:
    data: dict[str, Any] = {
        "summary": f"Summary of part {n}.",
        "key_points": [f"Point {n}a", f"Point {n}b"],
        "terms": [
            {"term": f"Term {n}", "definition": f"Meaning {n}"},
            {"term": "Shared", "definition": f"from {n}"},
        ],
        "flashcards": [{"front": f"Q{n}", "back": f"A{n}"}],
        "quiz": [
            {
                "question": f"Quiz {n}?",
                "options": ["w", "x", "y", "z"],
                "answer": n % 4,
                "explanation": "because",
            }
        ],
    }
    data.update(extra)
    return json.dumps(data)


SYNTHESIS = json.dumps(
    {
        "title": "Model Title",
        "tldr": "Overall summary.",
        "sections": [
            {"heading": "Start", "summary": "s1", "chunk_ids": [1, 2]},
            {"heading": "Rest", "summary": "s2", "chunk_ids": [3, 99]},
        ],
    }
)


def lecture3() -> Any:
    vtt = "WEBVTT\n\n" + "\n".join(
        f"00:0{i}:00.000 --> 00:0{i}:30.000\n"
        f"Part {i} talks about topic {i} in depth, with examples, details and a recap at the end.\n"
        for i in range(1, 4)
    )
    return parse_transcript(vtt)


def engine(responses: dict[str, str], **config: Any) -> tuple[FakeProvider, LLMEngine]:
    fake = FakeProvider.from_config({"responses": responses, "regex": True, **config})
    return fake, LLMEngine(fake, concurrency=1)


def run(eng: LLMEngine, **opts: Any) -> StudyPack:
    return analyze(lecture3(), eng, chunk_words=20, options=PackOptions(**opts))


BASE = {
    "TASK: synthesis": SYNTHESIS,
    'excerpt id="1"': notes(1),
    'excerpt id="2"': notes(2),
    'excerpt id="3"': notes(3),
}


class TestLLMEngine:
    def test_builds_pack_from_chunk_notes_and_synthesis(self) -> None:
        fake, eng = engine(BASE)
        pack = run(eng)
        assert pack.engine == "fake"
        assert (pack.title, pack.tldr) == ("Model Title", "Overall summary.")
        assert [(s.heading, s.chunk_ids, s.start) for s in pack.sections] == [
            ("Start", [1, 2], 60.0),
            ("Rest", [3], 180.0),  # unknown chunk id 99 dropped
        ]
        # round-robin across chunks so a cap still covers the whole lecture
        assert [p.text for p in pack.key_points[:3]] == ["Point 1a", "Point 2a", "Point 3a"]
        assert [p.chunk_id for p in pack.key_points[:3]] == [1, 2, 3]
        assert [t.term for t in pack.terms] == ["Term 1", "Shared", "Term 2", "Term 3"]
        assert pack.terms[1].definition == "from 1"  # first definition wins
        assert [c.chunk_id for c in pack.flashcards] == [1, 2, 3]
        assert [q.answer for q in pack.quiz] == [1, 2, 3]
        assert pack.warnings == []
        assert len(fake.calls) == 4  # 3 chunks + synthesis
        system, user = fake.calls[0]
        assert system.role == "system" and "LectureIQ" in system.content
        assert "Part 1 talks about topic 1" in user.content

    def test_caps_are_applied(self) -> None:
        _, eng = engine(BASE)
        pack = run(eng, max_key_points=2, max_terms=1, max_flashcards=1, max_quiz=2)
        assert (len(pack.key_points), len(pack.terms), len(pack.flashcards), len(pack.quiz)) == (
            2,
            1,
            1,
            2,
        )

    def test_user_title_wins_over_model_title(self) -> None:
        _, eng = engine(BASE)
        lecture = lecture3()
        lecture.title = "My Lecture"
        assert analyze(lecture, eng, chunk_words=20).title == "My Lecture"

    def test_malformed_items_are_dropped_with_a_warning(self) -> None:
        bad_quiz = [
            {"question": "?", "options": ["a", "b"], "answer": 5},
            {"question": "no options"},
        ]
        _, eng = engine(
            {**BASE, 'excerpt id="2"': notes(2, quiz=bad_quiz, flashcards=[{"front": "x"}])}
        )
        pack = run(eng)
        assert [q.chunk_id for q in pack.quiz] == [1, 3]
        assert any("dropped 3 malformed" in w for w in pack.warnings)

    def test_invalid_json_gets_one_repair_attempt(self) -> None:
        # Chunk 2 first answers with prose; the repair turn matches the default reply.
        fake, eng = engine({**BASE, 'excerpt id="2"': "Sure! Here are notes."}, default=notes(2))
        pack = run(eng)
        assert pack.warnings == []
        assert any(c.chunk_id == 2 for c in pack.flashcards)
        repair = [
            call for call in fake.calls if call[-1].content.startswith("That reply was not valid")
        ]
        assert len(repair) == 1 and repair[0][-2].role == "assistant"

    def test_chunk_failing_twice_is_skipped(self) -> None:
        _, eng = engine({**BASE, 'excerpt id="2"': "not json"}, default="still not json")
        pack = run(eng)
        assert [c.chunk_id for c in pack.flashcards] == [1, 3]
        assert any("excerpt 2 skipped" in w and "invalid JSON twice" in w for w in pack.warnings)

    def test_provider_errors_skip_the_chunk(self) -> None:
        _, eng = engine(BASE, fail_on=["Part 3 talks"])
        pack = run(eng)
        assert any("excerpt 3 skipped" in w for w in pack.warnings)

    def test_all_chunks_failing_raises(self) -> None:
        _, eng = engine({}, default="nope")
        with pytest.raises(GenerationError, match="no excerpt could be processed"):
            run(eng)

    def test_synthesis_failure_falls_back_to_chunk_summaries(self) -> None:
        _, eng = engine({**BASE, "TASK: synthesis": "garbage"}, default="garbage")
        pack = run(eng)
        assert [s.heading for s in pack.sections] == ["Part 1", "Part 2", "Part 3"]
        assert pack.tldr.startswith("Summary of part 1.")
        assert any("outline fell back" in w for w in pack.warnings)

    def test_answer_filters_citations_to_retrieved_passages(self) -> None:
        reply = json.dumps(
            {"answer": " Topic 2 is covered. ", "citations": [2, 2, 42], "grounded": True}
        )
        _, eng = engine({"TASK: answer": reply})
        answer = ask(lecture3(), "What about topic 2?", eng, chunk_words=20)
        assert answer.answer == "Topic 2 is covered."
        assert [(c.chunk_id, c.start) for c in answer.citations] == [(2, 120.0)]

    def test_answer_falls_back_to_raw_text_and_bracket_citations(self) -> None:
        _, eng = engine({"TASK: answer": "Topic 2 is discussed [2] and again [9]."})
        answer = ask(lecture3(), "topic 2", eng, chunk_words=20)
        assert answer.answer.startswith("Topic 2 is discussed")
        assert [c.chunk_id for c in answer.citations] == [2]

    def test_parse_json_accepts_fenced_output(self) -> None:
        parsed = parse_json(
            'Here:\n```json\n{"question": "q", "options": ["a", "b"], "answer": 1}\n```',
            QuizQuestion,
        )
        assert parsed.answer == 1
        with pytest.raises(ValueError, match="no JSON"):
            parse_json("nothing", QuizQuestion)


class TestOfflineEngine:
    def test_sample_lecture_yields_a_full_pack(self, sample_vtt: str) -> None:
        lecture = parse_transcript(sample_vtt, title="ML")
        pack = analyze(lecture, OfflineEngine())
        assert pack.engine == "offline" and pack.title == "ML"
        terms = {t.term for t in pack.terms}
        assert {"Overfitting", "Gradient descent", "Regularization", "Hyperparameters"} <= terms
        assert not any("workhorse" in t.lower() for t in terms)  # short "definitions" rejected
        assert len(pack.key_points) == 10
        assert len(pack.sections) == len(lecture.chunks)
        assert all(s.start is not None for s in pack.sections)
        assert len(pack.flashcards) == 20 and len(pack.quiz) == 10
        for q in pack.quiz:
            assert len(set(q.options)) == len(q.options) >= 2
        overfit = next(t for t in pack.terms if t.term == "Overfitting")
        q = next(q for q in pack.quiz if "Overfitting" in q.question)
        assert q.options[q.answer] == overfit.definition

    def test_output_is_deterministic(self, sample_vtt: str) -> None:
        first = analyze(parse_transcript(sample_vtt), OfflineEngine())
        second = analyze(parse_transcript(sample_vtt), OfflineEngine())
        assert first == second

    def test_lecture_without_definitions_warns(self) -> None:
        text = "We walked to the river early. The water looked calm and cold today. " * 5
        pack = analyze(parse_transcript(text), OfflineEngine())
        assert pack.terms == []
        assert pack.warnings and "glossary is empty" in pack.warnings[0]
        assert all(q.question.startswith("Complete the statement") for q in pack.quiz)

    def test_answers_cite_the_sentence_time(self, sample_vtt: str) -> None:
        lecture = parse_transcript(sample_vtt)
        answer = ask(lecture, "What is overfitting?", OfflineEngine())
        assert answer.grounded and answer.answer.startswith("Overfitting is when")
        (citation,) = answer.citations
        assert citation.start == 432.0  # the cue's own start, not the chunk's
        missing = ask(lecture, "Who painted the Mona Lisa?", OfflineEngine())
        assert not missing.grounded and missing.citations == []
        with pytest.raises(ValueError, match="empty"):
            ask(lecture, "   ", OfflineEngine())


def test_make_engine(tmp_path: Any) -> None:
    assert isinstance(make_engine("offline"), OfflineEngine)
    eng = make_engine("fake", {"default": "x"}, cache_dir=tmp_path)
    assert isinstance(eng, LLMEngine) and eng.name == "fake"


def test_exports(sample_vtt: str) -> None:
    pack = analyze(parse_transcript(sample_vtt, title="ML"), OfflineEngine())
    md = to_markdown(pack)
    assert md.startswith("# ML\n")
    for heading in (
        "## Outline",
        "## Key points",
        "## Glossary",
        "## Flashcards",
        "## Quiz",
        "### Answers",
    ):
        assert heading in md
    assert "(0:02)" in md
    first_answer = chr(65 + pack.quiz[0].answer)
    assert f"1. **{first_answer}**" in md
    rows = list(csv.reader(io.StringIO(flashcards_csv(pack))))
    assert rows == [[c.front, c.back] for c in pack.flashcards]
    empty = to_markdown(StudyPack(title="T", tldr="x", engine="offline"))
    assert "## Quiz" not in empty and empty.endswith("_Generated by LectureIQ (offline engine)._\n")

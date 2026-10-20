"""Persistence (SQLite), audio transcription, and LectureIQ as an evalkit provider."""

from __future__ import annotations

import io
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import pytest
from starlette.testclient import TestClient

from evalkit import Message, ProviderError, create_provider, load_suite, run_suite
from lectureiq import analyze, offline, parse_transcript
from lectureiq import cli as lectureiq_cli
from lectureiq.audio import WhisperTranscriber, is_audio, transcribe_lecture
from lectureiq.errors import TranscriptionError
from lectureiq.evalkit_provider import LectureIQProvider
from lectureiq.export import to_vtt
from lectureiq.offline import OfflineEngine
from lectureiq.store import MemoryStore, SQLiteStore
from lectureiq.transcript import Segment, lecture_from_dict, lecture_to_dict
from lectureiq.web.app import create_app

QA_SUITE = Path(__file__).resolve().parent.parent / "examples" / "lectureiq_qa.yaml"
SEGMENTS = [
    Segment("Entropy is a measure of uncertainty in a random variable.", 0.0, 4.0),
    Segment("Cross entropy compares two probability distributions over the same events.", 4.0, 9.5),
]


class StubTranscriber:
    name = "stub"

    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    def transcribe(self, audio: bytes, filename: str) -> list[Segment]:
        self.calls.append((len(audio), filename))
        return list(SEGMENTS)


# ---------- persistence ----------


def test_lecture_dict_round_trip(sample_vtt: str) -> None:
    lecture = parse_transcript(sample_vtt, title="ML")
    analyze(lecture, OfflineEngine())
    restored = lecture_from_dict(json.loads(json.dumps(lecture_to_dict(lecture))))
    assert restored.segments == lecture.segments and restored.chunks == lecture.chunks
    assert restored.title == "ML"


@pytest.mark.parametrize("make_store", [lambda p: MemoryStore(), lambda p: SQLiteStore(p / "l.db")])
def test_stores_add_get_recent_delete(
    make_store: Callable[[Path], Any], tmp_path: Path, sample_vtt: str
) -> None:
    store = make_store(tmp_path)
    lectures = [parse_transcript(sample_vtt, title=t) for t in ("First", "Second")]
    entries = [store.add(lec, analyze(lec, OfflineEngine())) for lec in lectures]
    got = store.get(entries[0].id)
    assert got is not None and got.pack == entries[0].pack
    assert got.lecture.chunks == entries[0].lecture.chunks
    assert [s.title for s in store.recent()] == ["Second", "First"]
    assert store.recent(limit=1)[0].word_count == 648
    assert store.delete(entries[0].id) and not store.delete(entries[0].id)
    assert store.get(entries[0].id) is None and store.get("missing") is None


def test_sqlite_survives_restart(tmp_path: Path, sample_vtt: str) -> None:
    db = tmp_path / "nested" / "lectures.db"
    with TestClient(create_app(OfflineEngine(), store=SQLiteStore(db))) as first:
        assert first.get("/api/health").json()["persistent"] is True
        lecture_id = first.post("/api/lectures/sample").json()["id"]
    with TestClient(create_app(OfflineEngine(), store=SQLiteStore(db))) as second:
        assert [item["id"] for item in second.get("/api/lectures").json()] == [lecture_id]
        body = second.get(f"/api/lectures/{lecture_id}").json()
        assert body["title"] == "Introduction to Machine Learning"
        answer = second.post(
            f"/api/lectures/{lecture_id}/ask", json={"question": "What is overfitting?"}
        )
        assert answer.json()["answer"].startswith("Overfitting is")


def test_list_and_delete_endpoints() -> None:
    with TestClient(create_app(OfflineEngine())) as client:
        assert client.get("/api/lectures").json() == []
        lecture_id = client.post("/api/lectures/sample").json()["id"]
        (item,) = client.get("/api/lectures").json()
        assert (
            item["id"] == lecture_id and item["engine"] == "offline" and item["word_count"] == 648
        )
        assert client.delete(f"/api/lectures/{lecture_id}").status_code == 204
        assert client.delete(f"/api/lectures/{lecture_id}").status_code == 404
        missing = client.get(f"/api/lectures/{lecture_id}").json()["error"]
        assert "--db" in missing


# ---------- audio ----------


def test_is_audio() -> None:
    assert is_audio("Lecture 3.MP3") and is_audio("talk.webm")
    assert not is_audio("notes.vtt") and not is_audio("noext")


def whisper(handler: Callable[[httpx.Request], httpx.Response], **kw: Any) -> WhisperTranscriber:
    return WhisperTranscriber(api_key="sk-test", transport=httpx.MockTransport(handler), **kw)


def test_whisper_request_and_segments() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(
            200,
            json={
                "text": "ignored when segments exist",
                "segments": [
                    {"start": 0.0, "end": 2.5, "text": "  Hello   class. "},
                    {"start": 2.5, "end": 4.0, "text": "   "},
                    {"start": 4.0, "end": 7.0, "text": "Today: entropy."},
                ],
            },
        )

    t = whisper(handler, base_url="http://stt.local/v1/", model="large-v3", language="en")
    lecture = transcribe_lecture(t, b"RIFF....", "/tmp/week_3_entropy.wav")
    assert lecture.title == "week 3 entropy"
    assert lecture.segments == [
        Segment("Hello class.", 0.0, 2.5),
        Segment("Today: entropy.", 4.0, 7.0),
    ]
    (request,) = seen
    assert str(request.url) == "http://stt.local/v1/audio/transcriptions"
    assert request.headers["authorization"] == "Bearer sk-test"
    body = request.content.decode("latin-1")
    for field in (
        'name="model"',
        "large-v3",
        "verbose_json",
        'name="language"',
        'filename="week_3_entropy.wav"',
    ):
        assert field in body


def test_whisper_text_only_response_has_no_timestamps() -> None:
    t = whisper(lambda r: httpx.Response(200, json={"text": "Just the text."}))
    assert t.transcribe(b"x", "a.mp3") == [Segment("Just the text.")]


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(401, text="bad key"), "HTTP 401: bad key"),
        (httpx.Response(200, text="not json"), "not JSON"),
        (httpx.Response(200, json={"text": "  "}), "no text"),
        (httpx.Response(200, json=["list"]), "unexpected"),
    ],
)
def test_whisper_errors(response: httpx.Response, message: str) -> None:
    with pytest.raises(TranscriptionError, match=message):
        whisper(lambda r: response).transcribe(b"x", "a.mp3")


def test_whisper_needs_audio_and_key(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(TranscriptionError, match="empty"):
        whisper(lambda r: httpx.Response(200)).transcribe(b"", "a.mp3")
    monkeypatch.delenv("STT_KEY", raising=False)
    with pytest.raises(TranscriptionError, match=r"\$STT_KEY"):
        WhisperTranscriber(api_key_env="STT_KEY").transcribe(b"x", "a.mp3")

    def boom(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused", request=request)

    with pytest.raises(TranscriptionError, match="request failed"):
        whisper(boom).transcribe(b"x", "a.mp3")


def test_audio_upload_through_the_api() -> None:
    stub = StubTranscriber()
    with TestClient(create_app(OfflineEngine(), transcriber=stub)) as client:
        assert client.get("/api/health").json()["audio"] is True
        created = client.post(
            "/api/lectures", files={"file": ("info_theory.mp3", b"\x00" * 1000, "audio/mpeg")}
        )
        assert created.status_code == 201, created.text
        data = created.json()
        assert stub.calls == [(1000, "info_theory.mp3")]
        assert data["title"] == "info theory" and data["has_timestamps"]
        vtt = client.get(f"/api/lectures/{data['id']}/transcript.vtt")
        assert vtt.status_code == 200 and vtt.headers["content-type"].startswith("text/vtt")
        assert parse_transcript(vtt.text).segments == SEGMENTS


def test_audio_rejected_without_transcriber_and_text_has_no_vtt() -> None:
    with TestClient(create_app(OfflineEngine())) as client:
        response = client.post("/api/lectures", files={"file": ("talk.m4a", b"\x00", "audio/mp4")})
        assert response.status_code == 400 and "--transcribe" in response.json()["error"]
        plain = client.post(
            "/api/lectures", json={"text": "Entropy is a measure of uncertainty here."}
        )
        conflict = client.get(f"/api/lectures/{plain.json()['id']}/transcript.vtt")
        assert conflict.status_code == 409


def test_transcription_failure_is_502() -> None:
    class Failing(StubTranscriber):
        def transcribe(self, audio: bytes, filename: str) -> list[Segment]:
            raise TranscriptionError("HTTP 500: upstream down")

    with TestClient(create_app(OfflineEngine(), transcriber=Failing())) as client:
        response = client.post("/api/lectures", files={"file": ("a.wav", b"x", "audio/wav")})
        assert response.status_code == 502 and "upstream down" in response.json()["error"]


def test_to_vtt_round_trips_and_needs_timestamps(sample_vtt: str) -> None:
    lecture = parse_transcript(sample_vtt)
    assert parse_transcript(to_vtt(lecture)).segments == lecture.segments
    with pytest.raises(ValueError, match="no timestamps"):
        to_vtt(parse_transcript("plain text only here"))


# ---------- CLI with audio ----------


def cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return lectureiq_cli.main(list(argv), out=out), out.getvalue()


def test_cli_transcribe_and_analyze_audio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    audio = tmp_path / "week_3.mp3"
    audio.write_bytes(b"\x00" * 10)
    code, _ = cli("analyze", str(audio))
    assert code == 2 and "add --transcribe" in capsys.readouterr().err

    stub = StubTranscriber()
    monkeypatch.setattr(lectureiq_cli, "make_transcriber", lambda args: stub)
    code, out = cli("transcribe", str(audio))
    assert code == 0 and f"-> {tmp_path / 'week_3.vtt'}" in out
    assert parse_transcript((tmp_path / "week_3.vtt").read_text()).segments == SEGMENTS

    code, out = cli("analyze", str(audio), "--transcribe")
    assert code == 0 and "week 3" in out
    code, out = cli("ask", str(audio), "What is entropy?", "--transcribe")
    assert code == 0 and out.startswith("Entropy is a measure of uncertainty")


def test_cli_transcribe_text_only_writes_txt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class TextOnly(StubTranscriber):
        def transcribe(self, audio: bytes, filename: str) -> list[Segment]:
            return [Segment("No timestamps here.")]

    monkeypatch.setattr(lectureiq_cli, "make_transcriber", lambda args: TextOnly())
    audio = tmp_path / "talk.wav"
    audio.write_bytes(b"x")
    code, _ = cli("transcribe", str(audio))
    assert code == 0 and (tmp_path / "talk.txt").read_text() == "No timestamps here.\n"


def test_make_transcriber_reads_flags() -> None:
    args = lectureiq_cli.build_parser().parse_args(
        ["serve", "--transcribe", "--transcribe-model", "m", "--transcribe-url", "http://x/v1"]
    )
    t = lectureiq_cli.make_transcriber(args)
    assert isinstance(t, WhisperTranscriber) and (t.model, t.base_url) == ("m", "http://x/v1")
    assert (
        lectureiq_cli.make_transcriber(lectureiq_cli.build_parser().parse_args(["serve"])) is None
    )


# ---------- LectureIQ as an evalkit provider ----------


def test_qa_suite_passes_with_the_offline_engine() -> None:
    suite = load_suite(QA_SUITE)
    _, config = suite.resolve_provider("lectureiq")
    result = run_suite(suite, create_provider("lectureiq", config))
    failures = [
        (c.case_id, c.output, [s.reason for s in c.scores if not s.passed])
        for c in result.cases
        if not c.passed
    ]
    assert failures == []
    assert result.total == 10 and result.passed


def test_qa_suite_catches_the_definition_ranking_regression(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Without the bonus for sentences that define the asked-about term, the offline engine
    # answers "What are hyperparameters?" with a recap sentence that merely mentions them.
    monkeypatch.setattr(offline, "_DEFINES_TERM_BONUS", 0.0)
    suite = load_suite(QA_SUITE).filter_tags(["definition"])
    result = run_suite(suite, LectureIQProvider.from_config({}))
    failed = {c.case_id for c in result.cases if not c.passed}
    assert {"define-hyperparameters", "define-gradient-descent"} <= failed


def test_provider_answers_with_sources_and_declines() -> None:
    provider = LectureIQProvider.from_config({"cite": True})
    reply = provider.complete(
        [Message(role="user", content="What does the learning rate control?")]
    )
    assert reply.text.endswith("Sources: 6:12") and reply.model == "offline"
    declined = provider.complete([Message(role="user", content="Who painted the Mona Lisa?")])
    assert declined.text == "The lecture doesn't seem to cover that."
    quiet = LectureIQProvider.from_config({"cite": False})
    assert (
        "Sources" not in quiet.complete([Message(role="user", content="What is overfitting?")]).text
    )
    with pytest.raises(ProviderError, match="no question"):
        provider.complete([Message(role="system", content="only a system prompt")])


def test_provider_reads_transcript_files(tmp_path: Path) -> None:
    path = tmp_path / "entropy.txt"
    path.write_text("Entropy is a measure of uncertainty in a random variable.")
    provider = LectureIQProvider.from_config({"transcript": str(path)})
    assert provider.complete([Message(role="user", content="What is entropy?")]).text.startswith(
        "Entropy is"
    )
    missing = LectureIQProvider.from_config({"transcript": str(tmp_path / "nope.txt")})
    with pytest.raises(ProviderError, match="cannot read"):
        missing.complete([Message(role="user", content="q")])


def test_provider_cache_identity_has_no_secrets() -> None:
    provider = LectureIQProvider.from_config(
        {
            "engine": "openai",
            "model": "m",
            "engine_options": {"api_key": "sk-secret", "base_url": "http://x"},
        }
    )
    identity = json.dumps(provider.cache_identity())
    assert "sk-secret" not in identity and "http://x" in identity and '"model": "m"' in identity

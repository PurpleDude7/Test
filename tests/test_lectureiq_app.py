from __future__ import annotations

from collections.abc import Iterator

import pytest
from starlette.testclient import TestClient

from evalkit import FakeProvider
from lectureiq.llm import LLMEngine
from lectureiq.models import StudyPack
from lectureiq.offline import OfflineEngine
from lectureiq.sample import SAMPLE_TITLE
from lectureiq.store import MemoryStore
from lectureiq.transcript import Lecture, Segment
from lectureiq.web import app as web

SRT = (
    "1\n00:00:01,000 --> 00:00:05,000\nEntropy is a measure of uncertainty in a random variable.\n"
)


@pytest.fixture
def client() -> Iterator[TestClient]:
    with TestClient(web.create_app(OfflineEngine())) as c:
        yield c


def test_index_and_static_assets(client: TestClient) -> None:
    page = client.get("/")
    assert page.status_code == 200 and "<title>LectureIQ</title>" in page.text
    assert client.get("/static/app.js").status_code == 200
    assert client.get("/static/style.css").status_code == 200
    assert client.get("/static/../app.py").status_code == 404


def test_health(client: TestClient) -> None:
    assert client.get("/api/health").json() == {
        "status": "ok",
        "engine": "offline",
        "model": None,
        "audio": False,
        "persistent": False,
    }


def test_sample_round_trip(client: TestClient) -> None:
    created = client.post("/api/lectures/sample")
    assert created.status_code == 201
    data = created.json()
    assert data["title"] == SAMPLE_TITLE and data["has_timestamps"]
    assert data["chunks"][0]["start"] == 2.0
    assert len(data["pack"]["quiz"]) == 10

    lecture_id = data["id"]
    assert client.get(f"/api/lectures/{lecture_id}").json()["pack"] == data["pack"]

    answer = client.post(
        f"/api/lectures/{lecture_id}/ask", json={"question": "What is regularization?"}
    )
    assert answer.status_code == 200
    body = answer.json()
    assert body["grounded"] and body["answer"].startswith("Regularization is")
    assert body["citations"][0]["start"] is not None

    md = client.get(f"/api/lectures/{lecture_id}/export.md")
    assert (
        md.headers["content-disposition"]
        == 'attachment; filename="introduction-to-machine-learning.md"'
    )
    assert md.text.startswith("# Introduction to Machine Learning")
    cards = client.get(f"/api/lectures/{lecture_id}/flashcards.csv")
    assert cards.headers["content-type"].startswith("text/csv")
    assert cards.text.count("\n") == len(data["pack"]["flashcards"])


def test_paste_json_and_upload_file(client: TestClient) -> None:
    pasted = client.post(
        "/api/lectures",
        json={"text": "Entropy is a measure of uncertainty in a random variable.", "title": "Info"},
    )
    assert pasted.status_code == 201 and pasted.json()["title"] == "Info"
    assert not pasted.json()["has_timestamps"]

    uploaded = client.post(
        "/api/lectures", files={"file": ("info_theory_1.srt", SRT, "text/plain")}
    )
    assert uploaded.status_code == 201
    data = uploaded.json()
    assert data["title"] == "info theory 1" and data["has_timestamps"]

    form_text = client.post(
        "/api/lectures", data={"text": "Entropy is a measure of uncertainty here.", "title": "Form"}
    )
    assert form_text.status_code == 201 and form_text.json()["title"] == "Form"


def test_bad_submissions(client: TestClient) -> None:
    responses = {
        "no text": client.post("/api/lectures", json={"text": "   "}),
        "JSON": client.post(
            "/api/lectures", content=b"not json", headers={"content-type": "application/json"}
        ),
        "object": client.post("/api/lectures", json=["a", "list"]),
    }
    for message, response in responses.items():
        assert response.status_code == 400
        assert message in response.json()["error"]


def test_upload_size_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web, "MAX_UPLOAD_BYTES", 100)
    big = client.post("/api/lectures", json={"text": "word " * 100})
    assert big.status_code == 413 and "too large" in big.json()["error"]


def test_unknown_lecture_and_bad_questions(client: TestClient) -> None:
    for path in (
        "/api/lectures/nope",
        "/api/lectures/nope/export.md",
        "/api/lectures/nope/flashcards.csv",
    ):
        response = client.get(path)
        assert response.status_code == 404 and "not found" in response.json()["error"]
    assert client.post("/api/lectures/nope/ask", json={"question": "x"}).status_code == 404

    lecture_id = client.post("/api/lectures/sample").json()["id"]
    ask_url = f"/api/lectures/{lecture_id}/ask"
    assert client.post(ask_url, json={"question": "  "}).json()["error"] == "question is empty"
    assert "too long" in client.post(ask_url, json={"question": "x" * 1001}).json()["error"]
    assert client.post(ask_url, content=b"[]").status_code == 400


def test_generation_failures_return_502() -> None:
    engine = LLMEngine(FakeProvider.from_config({"default": "not json"}))
    with TestClient(web.create_app(engine)) as c:
        response = c.post("/api/lectures/sample")
        assert response.status_code == 502
        assert "generation failed" in response.json()["error"]
        assert c.get("/api/health").json()["model"] == "fake-1"


def test_store_evicts_least_recently_used() -> None:
    store = MemoryStore(capacity=2)
    pack = StudyPack(title="t", tldr="x", engine="offline")
    lecture = Lecture(title="t", segments=[Segment("x")])
    a, b = store.add(lecture, pack), store.add(lecture, pack)
    assert store.get(a.id) is a  # touch a, so b is now the oldest
    c = store.add(lecture, pack)
    assert store.get(b.id) is None
    assert store.get(a.id) is a and store.get(c.id) is c and len(store) == 2

"""Starlette app: a JSON API plus a single-page frontend (vanilla JS, no build step)."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse, PlainTextResponse, Response
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from evalkit.errors import EvalkitError
from lectureiq.engines import Engine, PackOptions
from lectureiq.errors import TranscriptError
from lectureiq.export import flashcards_csv, to_markdown
from lectureiq.pipeline import DEFAULT_CHUNK_WORDS, analyze, ask
from lectureiq.sample import SAMPLE_TITLE, sample_transcript
from lectureiq.store import Entry, LectureStore
from lectureiq.transcript import parse_transcript

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2_000_000
MAX_QUESTION_CHARS = 1000


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "lecture"


def _lecture_json(entry: Entry) -> dict[str, Any]:
    lecture = entry.lecture
    return {
        "id": entry.id,
        "title": entry.pack.title,
        "word_count": lecture.word_count,
        "has_timestamps": lecture.has_timestamps,
        "chunks": [
            {"id": c.id, "start": c.start, "end": c.end, "text": c.text} for c in lecture.chunks
        ],
        "pack": entry.pack.model_dump(mode="json"),
    }


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


async def _read_submission(request: Request) -> tuple[str, str]:
    """(text, title) from form data (a file or a text field) or a JSON body."""
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"upload too large (max {MAX_UPLOAD_BYTES // 1_000_000} MB)")
    content_type = request.headers.get("content-type", "")
    if content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        async with request.form(max_files=1, max_fields=10) as form:
            title = str(form.get("title") or "")
            upload = form.get("file")
            if isinstance(upload, UploadFile) and upload.filename:
                raw = await upload.read(MAX_UPLOAD_BYTES + 1)
                if len(raw) > MAX_UPLOAD_BYTES:
                    raise HTTPException(413, "upload too large")
                title = title or Path(upload.filename).stem.replace("_", " ")
                return raw.decode("utf-8", errors="replace"), title
            return str(form.get("text") or ""), title
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        raise HTTPException(400, "body must be JSON or multipart form data") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body must be an object")
    return str(body.get("text") or ""), str(body.get("title") or "")


def create_app(
    engine: Engine,
    *,
    options: PackOptions | None = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    store: LectureStore | None = None,
) -> Starlette:
    """Build the web app around an engine (offline or LLM-backed)."""
    lectures = store or LectureStore()
    pack_options = options or PackOptions()

    async def _analyze(text: str, title: str) -> Response:
        try:
            lecture = parse_transcript(text, title=title or "Untitled lecture")
            pack = await run_in_threadpool(
                analyze, lecture, engine, options=pack_options, chunk_words=chunk_words
            )
        except TranscriptError as exc:
            return _error(400, str(exc))
        except EvalkitError as exc:
            return _error(502, f"generation failed: {exc}")
        return JSONResponse(_lecture_json(lectures.add(lecture, pack)), status_code=201)

    def _entry(request: Request) -> Entry:
        entry = lectures.get(request.path_params["lecture_id"])
        if entry is None:
            raise HTTPException(404, "lecture not found (the server keeps recent lectures only)")
        return entry

    async def index(request: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html")

    async def health(request: Request) -> Response:
        model = getattr(getattr(engine, "provider", None), "model", None)
        return JSONResponse({"status": "ok", "engine": engine.name, "model": model})

    async def create(request: Request) -> Response:
        return await _analyze(*await _read_submission(request))

    async def create_sample(request: Request) -> Response:
        return await _analyze(sample_transcript(), SAMPLE_TITLE)

    async def get_lecture(request: Request) -> Response:
        return JSONResponse(_lecture_json(_entry(request)))

    async def ask_lecture(request: Request) -> Response:
        entry = _entry(request)
        try:
            body = json.loads(await request.body())
            question = str(body.get("question", "")).strip()
        except (json.JSONDecodeError, AttributeError):
            return _error(400, 'body must be JSON like {"question": "..."}')
        if not question:
            return _error(400, "question is empty")
        if len(question) > MAX_QUESTION_CHARS:
            return _error(400, f"question is too long (max {MAX_QUESTION_CHARS} chars)")
        try:
            answer = await run_in_threadpool(ask, entry.lecture, question, engine)
        except EvalkitError as exc:
            return _error(502, f"answering failed: {exc}")
        return JSONResponse(answer.model_dump(mode="json"))

    def _download(entry: Entry, body: str, ext: str, media_type: str) -> Response:
        name = f"{_slug(entry.pack.title)}.{ext}"
        return PlainTextResponse(
            body,
            media_type=media_type,
            headers={"Content-Disposition": f'attachment; filename="{name}"'},
        )

    async def export_markdown(request: Request) -> Response:
        entry = _entry(request)
        return _download(entry, to_markdown(entry.pack), "md", "text/markdown")

    async def export_flashcards(request: Request) -> Response:
        entry = _entry(request)
        return _download(entry, flashcards_csv(entry.pack), "csv", "text/csv")

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        if request.url.path.startswith("/api/"):
            return _error(exc.status_code, str(exc.detail))
        return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

    routes = [
        Route("/", index),
        Route("/api/health", health),
        Route("/api/lectures", create, methods=["POST"]),
        Route("/api/lectures/sample", create_sample, methods=["POST"]),
        Route("/api/lectures/{lecture_id}", get_lecture),
        Route("/api/lectures/{lecture_id}/ask", ask_lecture, methods=["POST"]),
        Route("/api/lectures/{lecture_id}/export.md", export_markdown),
        Route("/api/lectures/{lecture_id}/flashcards.csv", export_flashcards),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    return Starlette(routes=routes, exception_handlers={HTTPException: http_error})

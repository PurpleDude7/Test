"""Starlette app: a JSON API plus a single-page frontend (vanilla JS, no build step)."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
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
from lectureiq.audio import MAX_AUDIO_BYTES, Transcriber, is_audio, transcribe_lecture
from lectureiq.engines import Engine, PackOptions
from lectureiq.errors import TranscriptError, TranscriptionError
from lectureiq.export import flashcards_csv, to_markdown, to_vtt
from lectureiq.pipeline import DEFAULT_CHUNK_WORDS, analyze, ask
from lectureiq.sample import SAMPLE_TITLE, sample_transcript
from lectureiq.store import Entry, MemoryStore, Store
from lectureiq.transcript import Lecture, parse_transcript

STATIC_DIR = Path(__file__).parent / "static"
MAX_UPLOAD_BYTES = 2_000_000
MAX_QUESTION_CHARS = 1000
RECENT_LIMIT = 20


@dataclass(frozen=True)
class Submission:
    title: str
    text: str = ""
    audio: bytes | None = None
    filename: str = ""


def _slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")[:60] or "lecture"


def _lecture_json(entry: Entry) -> dict[str, Any]:
    lecture = entry.lecture
    return {
        "id": entry.id,
        "title": entry.pack.title,
        "created_at": entry.created_at.isoformat(),
        "word_count": lecture.word_count,
        "has_timestamps": lecture.has_timestamps,
        "chunks": [
            {"id": c.id, "start": c.start, "end": c.end, "text": c.text} for c in lecture.chunks
        ],
        "pack": entry.pack.model_dump(mode="json"),
    }


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status)


def _mb(n: int) -> str:
    return f"{n // 1_000_000} MB"


async def _read_submission(request: Request, *, accept_audio: bool) -> Submission:
    """A transcript (form file, form text, or JSON body) or, if enabled, an audio file."""
    limit = MAX_AUDIO_BYTES if accept_audio else MAX_UPLOAD_BYTES
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        raise HTTPException(413, f"upload too large (max {_mb(limit)})")
    content_type = request.headers.get("content-type", "")
    if content_type.startswith(("multipart/form-data", "application/x-www-form-urlencoded")):
        async with request.form(max_files=1, max_fields=10, max_part_size=limit) as form:
            title = str(form.get("title") or "")
            upload = form.get("file")
            if not (isinstance(upload, UploadFile) and upload.filename):
                return Submission(title=title, text=str(form.get("text") or ""))
            filename = upload.filename
            audio = is_audio(filename)
            if audio and not accept_audio:
                raise HTTPException(
                    400, "audio uploads are off; start the server with --transcribe"
                )
            cap = MAX_AUDIO_BYTES if audio else MAX_UPLOAD_BYTES
            raw = await upload.read(cap + 1)
            if len(raw) > cap:
                raise HTTPException(413, f"upload too large (max {_mb(cap)})")
            title = title or Path(filename).stem.replace("_", " ")
            if audio:
                return Submission(title=title, audio=raw, filename=filename)
            return Submission(title=title, text=raw.decode("utf-8", errors="replace"))
    try:
        body = json.loads(await request.body())
    except json.JSONDecodeError:
        raise HTTPException(400, "body must be JSON or multipart form data") from None
    if not isinstance(body, dict):
        raise HTTPException(400, "JSON body must be an object")
    return Submission(title=str(body.get("title") or ""), text=str(body.get("text") or ""))


def create_app(
    engine: Engine,
    *,
    options: PackOptions | None = None,
    chunk_words: int = DEFAULT_CHUNK_WORDS,
    store: Store | None = None,
    transcriber: Transcriber | None = None,
) -> Starlette:
    """Build the web app around an engine (offline or LLM-backed).

    ``store`` defaults to an in-memory store; pass a ``SQLiteStore`` to keep lectures across
    restarts. Audio uploads are accepted only when a ``transcriber`` is given.
    """
    lectures: Store = store if store is not None else MemoryStore()
    pack_options = options or PackOptions()
    persistent = not isinstance(lectures, MemoryStore)

    async def _analyze(submission: Submission) -> Response:
        title = submission.title or "Untitled lecture"
        try:
            lecture: Lecture
            if submission.audio is not None:
                assert transcriber is not None  # guaranteed by _read_submission
                lecture = await run_in_threadpool(
                    transcribe_lecture,
                    transcriber,
                    submission.audio,
                    submission.filename,
                    title=submission.title or None,
                )
            else:
                lecture = parse_transcript(submission.text, title=title)
            pack = await run_in_threadpool(
                analyze, lecture, engine, options=pack_options, chunk_words=chunk_words
            )
        except TranscriptError as exc:
            return _error(400, str(exc))
        except TranscriptionError as exc:
            return _error(502, str(exc))
        except EvalkitError as exc:
            return _error(502, f"generation failed: {exc}")
        entry = await run_in_threadpool(lectures.add, lecture, pack)
        return JSONResponse(_lecture_json(entry), status_code=201)

    async def _entry(request: Request) -> Entry:
        entry = await run_in_threadpool(lectures.get, request.path_params["lecture_id"])
        if entry is None:
            hint = "" if persistent else " (without --db the server keeps recent lectures only)"
            raise HTTPException(404, f"lecture not found{hint}")
        return entry

    async def index(request: Request) -> Response:
        return FileResponse(STATIC_DIR / "index.html")

    async def health(request: Request) -> Response:
        model = getattr(getattr(engine, "provider", None), "model", None)
        return JSONResponse(
            {
                "status": "ok",
                "engine": engine.name,
                "model": model,
                "audio": transcriber is not None,
                "persistent": persistent,
            }
        )

    async def create(request: Request) -> Response:
        return await _analyze(await _read_submission(request, accept_audio=transcriber is not None))

    async def create_sample(request: Request) -> Response:
        return await _analyze(Submission(title=SAMPLE_TITLE, text=sample_transcript()))

    async def recent(request: Request) -> Response:
        items = await run_in_threadpool(lectures.recent, RECENT_LIMIT)
        return JSONResponse(
            [
                {
                    "id": s.id,
                    "title": s.title,
                    "created_at": s.created_at.isoformat(),
                    "word_count": s.word_count,
                    "engine": s.engine,
                }
                for s in items
            ]
        )

    async def lecture(request: Request) -> Response:
        if request.method == "DELETE":
            deleted = await run_in_threadpool(lectures.delete, request.path_params["lecture_id"])
            return Response(status_code=204) if deleted else _error(404, "lecture not found")
        return JSONResponse(_lecture_json(await _entry(request)))

    async def ask_lecture(request: Request) -> Response:
        entry = await _entry(request)
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
        entry = await _entry(request)
        return _download(entry, to_markdown(entry.pack), "md", "text/markdown")

    async def export_flashcards(request: Request) -> Response:
        entry = await _entry(request)
        return _download(entry, flashcards_csv(entry.pack), "csv", "text/csv")

    async def export_vtt(request: Request) -> Response:
        entry = await _entry(request)
        if not entry.lecture.has_timestamps:
            return _error(409, "this lecture has no timestamps to export")
        return _download(entry, to_vtt(entry.lecture), "vtt", "text/vtt")

    async def http_error(request: Request, exc: Exception) -> Response:
        assert isinstance(exc, HTTPException)
        if request.url.path.startswith("/api/"):
            return _error(exc.status_code, str(exc.detail))
        return PlainTextResponse(str(exc.detail), status_code=exc.status_code)

    routes = [
        Route("/", index),
        Route("/api/health", health),
        Route("/api/lectures", recent, methods=["GET"]),
        Route("/api/lectures", create, methods=["POST"]),
        Route("/api/lectures/sample", create_sample, methods=["POST"]),
        Route("/api/lectures/{lecture_id}", lecture, methods=["GET", "DELETE"]),
        Route("/api/lectures/{lecture_id}/ask", ask_lecture, methods=["POST"]),
        Route("/api/lectures/{lecture_id}/export.md", export_markdown),
        Route("/api/lectures/{lecture_id}/flashcards.csv", export_flashcards),
        Route("/api/lectures/{lecture_id}/transcript.vtt", export_vtt),
        Mount("/static", StaticFiles(directory=STATIC_DIR), name="static"),
    ]
    return Starlette(routes=routes, exception_handlers={HTTPException: http_error})

"""Speech-to-text for recorded lectures, via any OpenAI-compatible transcription endpoint.

That covers OpenAI's Whisper API and self-hosted servers that mimic it (for example
faster-whisper or whisper.cpp servers). We ask for ``verbose_json`` so each segment keeps
its timestamps; servers that only return plain text still work, just without timestamps.
"""

from __future__ import annotations

import os
from pathlib import PurePath
from typing import Any, Protocol

import httpx

from lectureiq.errors import TranscriptionError
from lectureiq.transcript import Lecture, Segment

AUDIO_EXTENSIONS = frozenset(
    {".flac", ".m4a", ".mp3", ".mp4", ".mpeg", ".mpga", ".oga", ".ogg", ".wav", ".webm"}
)
MAX_AUDIO_BYTES = 25_000_000  # OpenAI's limit; self-hosted servers may allow more


def is_audio(filename: str) -> bool:
    return PurePath(filename).suffix.lower() in AUDIO_EXTENSIONS


class Transcriber(Protocol):
    name: str

    def transcribe(self, audio: bytes, filename: str) -> list[Segment]:
        """Return timestamped segments (``start``/``end`` may be None if unsupported)."""
        ...


class WhisperTranscriber:
    """Calls ``POST {base_url}/audio/transcriptions`` with a multipart upload."""

    name = "whisper"

    def __init__(
        self,
        *,
        model: str = "whisper-1",
        base_url: str = "https://api.openai.com/v1",
        api_key: str | None = None,
        api_key_env: str = "OPENAI_API_KEY",
        language: str | None = None,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._api_key = api_key
        self.api_key_env = api_key_env
        self.language = language
        self.timeout_s = timeout_s
        self._transport = transport

    def _key(self) -> str:
        key = self._api_key or os.environ.get(self.api_key_env)
        if not key:
            raise TranscriptionError(f"transcription needs an API key: set ${self.api_key_env}")
        return key

    def transcribe(self, audio: bytes, filename: str) -> list[Segment]:
        if not audio:
            raise TranscriptionError("audio file is empty")
        data = {"model": self.model, "response_format": "verbose_json"}
        if self.language:
            data["language"] = self.language
        try:
            with httpx.Client(timeout=self.timeout_s, transport=self._transport) as client:
                response = client.post(
                    f"{self.base_url}/audio/transcriptions",
                    headers={"Authorization": f"Bearer {self._key()}"},
                    data=data,
                    files={"file": (PurePath(filename).name, audio)},
                )
        except httpx.HTTPError as exc:
            raise TranscriptionError(f"transcription request failed: {exc}") from exc
        if response.status_code >= 400:
            raise TranscriptionError(
                f"transcription failed: HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            body = response.json()
        except ValueError as exc:
            raise TranscriptionError("transcription response was not JSON") from exc
        return _segments(body)


def _segments(body: Any) -> list[Segment]:
    if not isinstance(body, dict):
        raise TranscriptionError("unexpected transcription response")
    segments = [
        Segment(
            text=" ".join(str(s.get("text", "")).split()), start=s.get("start"), end=s.get("end")
        )
        for s in body.get("segments") or []
        if isinstance(s, dict) and str(s.get("text", "")).strip()
    ]
    if segments:
        return segments
    text = " ".join(str(body.get("text", "")).split())
    if not text:
        raise TranscriptionError("transcription returned no text")
    return [Segment(text=text)]


def transcribe_lecture(
    transcriber: Transcriber, audio: bytes, filename: str, *, title: str | None = None
) -> Lecture:
    segments = transcriber.transcribe(audio, filename)
    return Lecture(title=title or PurePath(filename).stem.replace("_", " "), segments=segments)

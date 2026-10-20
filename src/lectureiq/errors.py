"""LectureIQ errors. Everything raised on purpose derives from LectureIQError."""

from evalkit.errors import EvalkitError


class LectureIQError(EvalkitError):
    """Base class; subclassing EvalkitError lets callers catch provider and app errors alike."""


class TranscriptError(LectureIQError):
    """The transcript is empty, too large, or cannot be parsed."""


class GenerationError(LectureIQError):
    """The model's output could not be turned into study material."""


class TranscriptionError(LectureIQError):
    """Speech-to-text failed (no API key, HTTP error, unexpected response)."""

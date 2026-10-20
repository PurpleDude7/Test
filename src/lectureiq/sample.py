"""The bundled demo lecture (an original intro-to-ML transcript in WebVTT)."""

from __future__ import annotations

from importlib.resources import files

SAMPLE_TITLE = "Introduction to Machine Learning"


def sample_transcript() -> str:
    return files("lectureiq").joinpath("samples/intro_to_ml.vtt").read_text(encoding="utf-8")

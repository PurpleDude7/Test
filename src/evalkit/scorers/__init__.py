"""Scorers. Importing this package registers the built-in ones."""

from evalkit.scorers.base import (
    ScoreContext,
    Scorer,
    build_scorer,
    register_scorer,
    scorer_registry,
)
from evalkit.scorers.basic import Contains, ExactMatch, Regex
from evalkit.scorers.embedding import EmbeddingSimilarity
from evalkit.scorers.judge import LLMJudge

__all__ = [
    "Contains",
    "EmbeddingSimilarity",
    "ExactMatch",
    "LLMJudge",
    "Regex",
    "ScoreContext",
    "Scorer",
    "build_scorer",
    "register_scorer",
    "scorer_registry",
]

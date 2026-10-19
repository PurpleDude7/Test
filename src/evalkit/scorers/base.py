"""The Scorer interface and registry."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, ValidationError

from evalkit.errors import SuiteError
from evalkit.models import ScorerSpec, TestCase
from evalkit.providers.base import Provider
from evalkit.registry import Registry
from evalkit.results import ScoreResult

ProviderLookup = Callable[[str | None, Mapping[str, Any]], Provider]
"""``(name, overrides) -> Provider``. ``name=None`` means the provider under test."""


@dataclass(frozen=True)
class ScoreContext:
    """Everything a scorer may look at when judging one output."""

    case: TestCase
    output: str
    get_provider: ProviderLookup

    @property
    def provider(self) -> Provider:
        """The provider that produced ``output``."""
        return self.get_provider(None, {})


class Scorer(BaseModel):
    """Base class for scorers.

    A scorer's fields are its YAML parameters, so pydantic validates them when the suite
    loads. Subclasses set ``name`` and implement :meth:`score`.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: ClassVar[str]

    @abstractmethod
    def score(self, ctx: ScoreContext) -> ScoreResult:
        """Judge ``ctx.output``."""

    def result(self, score: float, passed: bool, reason: str = "") -> ScoreResult:
        return ScoreResult(
            scorer=self.name, score=min(max(score, 0.0), 1.0), passed=passed, reason=reason
        )

    @staticmethod
    def resolve_reference(explicit: str | None, ctx: ScoreContext, field: str) -> str:
        """``explicit`` if set, else the case's ``expected``; raise if neither exists."""
        if explicit is not None:
            return explicit
        if ctx.case.expected is not None:
            return ctx.case.expected
        raise SuiteError(f"case {ctx.case.id!r}: scorer needs `{field}` or case `expected`")


scorer_registry: Registry[Scorer] = Registry("scorer", "evalkit.scorers")


def register_scorer(*names: str) -> Any:
    """Class decorator: make a Scorer subclass available under ``names``."""
    return scorer_registry.register(*names)


def build_scorer(spec: ScorerSpec | Mapping[str, Any]) -> Scorer:
    """Instantiate a scorer from its YAML spec, raising SuiteError on bad params."""
    if not isinstance(spec, ScorerSpec):
        spec = ScorerSpec.model_validate(spec)
    cls = scorer_registry.get(spec.type)
    try:
        return cls.model_validate(spec.params)
    except ValidationError as exc:
        raise SuiteError(f"invalid params for scorer {spec.type!r}: {exc}") from exc

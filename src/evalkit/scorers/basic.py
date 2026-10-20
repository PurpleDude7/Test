"""String-matching scorers: exact_match, contains, regex."""

from __future__ import annotations

import re
from functools import cached_property
from typing import Literal

from pydantic import Field, field_validator

from evalkit.results import ScoreResult
from evalkit.scorers.base import ScoreContext, Scorer, register_scorer


def _normalise(text: str, *, case_sensitive: bool, strip: bool) -> str:
    if strip:
        text = " ".join(text.split())
    return text if case_sensitive else text.casefold()


@register_scorer("exact_match")
class ExactMatch(Scorer):
    """Output equals ``value`` (default: the case's ``expected``)."""

    name = "exact_match"
    value: str | None = None
    case_sensitive: bool = True
    strip: bool = True
    """Trim ends and collapse internal whitespace before comparing."""

    def score(self, ctx: ScoreContext) -> ScoreResult:
        want = self.resolve_reference(self.value, ctx, "value")
        norm = {"case_sensitive": self.case_sensitive, "strip": self.strip}
        ok = _normalise(ctx.output, **norm) == _normalise(want, **norm)
        return self.result(1.0 if ok else 0.0, ok, "" if ok else f"expected {want!r}")


@register_scorer("contains")
class Contains(Scorer):
    """Output contains ``value`` (a string or list of strings)."""

    name = "contains"
    value: str | list[str] | None = None
    mode: Literal["any", "all"] = "all"
    case_sensitive: bool = False
    negate: bool = False
    """Pass when the output does *not* contain the value(s)."""

    def score(self, ctx: ScoreContext) -> ScoreResult:
        raw = self.value if self.value is not None else self.resolve_reference(None, ctx, "value")
        needles = [raw] if isinstance(raw, str) else raw
        hay = ctx.output if self.case_sensitive else ctx.output.casefold()
        found = [n for n in needles if (n if self.case_sensitive else n.casefold()) in hay]
        missing = [n for n in needles if n not in found]
        hit = bool(found) if self.mode == "any" else not missing
        ok = hit != self.negate
        if self.negate:
            reason = "" if ok else f"output contains forbidden {found!r}"
        else:
            reason = "" if ok else f"missing {missing!r}"
        frac = len(found) / len(needles) if needles else 1.0
        return self.result(1.0 - frac if self.negate else frac, ok, reason)


@register_scorer("regex")
class Regex(Scorer):
    """Output matches ``pattern`` (``re.search`` by default, ``fullmatch`` optional)."""

    name = "regex"
    pattern: str = Field(min_length=1)
    ignore_case: bool = False
    multiline: bool = False
    fullmatch: bool = False
    negate: bool = False

    @field_validator("pattern")
    @classmethod
    def _compiles(cls, value: str) -> str:
        try:
            re.compile(value)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return value

    @cached_property
    def _compiled(self) -> re.Pattern[str]:
        flags = (re.IGNORECASE if self.ignore_case else 0) | (re.MULTILINE if self.multiline else 0)
        return re.compile(self.pattern, flags)

    def score(self, ctx: ScoreContext) -> ScoreResult:
        text = ctx.output.strip() if self.fullmatch else ctx.output
        matched = (self._compiled.fullmatch if self.fullmatch else self._compiled.search)(text)
        ok = (matched is not None) != self.negate
        reason = "" if ok else f"{'matched' if self.negate else 'no match for'} /{self.pattern}/"
        return self.result(1.0 if ok else 0.0, ok, reason)

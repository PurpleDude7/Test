"""Compare two runs case by case to catch regressions a pass-rate gate would miss."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ValidationError, computed_field

from evalkit.errors import EvalkitError
from evalkit.results import CaseResult, RunResult


class Change(StrEnum):
    REGRESSED = "regressed"
    FIXED = "fixed"
    NEW = "new"
    REMOVED = "removed"
    STILL_PASSING = "still_passing"
    STILL_FAILING = "still_failing"


def mean_score(case: CaseResult) -> float:
    """Average scorer score; a case that errored or has no scores counts as 0."""
    if case.error is not None or not case.scores:
        return 0.0
    return sum(s.score for s in case.scores) / len(case.scores)


class CaseChange(BaseModel):
    case_id: str
    change: Change
    baseline_score: float | None = None
    current_score: float | None = None
    detail: str = ""
    """Why the current run failed, for regressions and still-failing cases."""


class Comparison(BaseModel):
    baseline_pass_rate: float
    current_pass_rate: float
    changes: list[CaseChange]

    def by_change(self, change: Change) -> list[CaseChange]:
        return [c for c in self.changes if c.change is change]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def regressions(self) -> int:
        return len(self.by_change(Change.REGRESSED))

    @computed_field  # type: ignore[prop-decorator]
    @property
    def fixes(self) -> int:
        return len(self.by_change(Change.FIXED))

    @property
    def pass_rate_delta(self) -> float:
        return self.current_pass_rate - self.baseline_pass_rate


def _failure_detail(case: CaseResult) -> str:
    if case.error:
        return case.error
    return "; ".join(f"{s.scorer}: {s.reason}" for s in case.scores if not s.passed)


def compare_runs(baseline: RunResult, current: RunResult) -> Comparison:
    """Classify every case id seen in either run.

    Order follows the current run, with cases that only exist in the baseline appended.
    """
    before = {c.case_id: c for c in baseline.cases}
    after = {c.case_id: c for c in current.cases}
    changes: list[CaseChange] = []
    for case in current.cases:
        old = before.get(case.case_id)
        if old is None:
            change = Change.NEW
        elif old.passed and not case.passed:
            change = Change.REGRESSED
        elif not old.passed and case.passed:
            change = Change.FIXED
        else:
            change = Change.STILL_PASSING if case.passed else Change.STILL_FAILING
        changes.append(
            CaseChange(
                case_id=case.case_id,
                change=change,
                baseline_score=None if old is None else mean_score(old),
                current_score=mean_score(case),
                detail="" if case.passed else _failure_detail(case),
            )
        )
    changes.extend(
        CaseChange(case_id=old.case_id, change=Change.REMOVED, baseline_score=mean_score(old))
        for old in baseline.cases
        if old.case_id not in after
    )
    return Comparison(
        baseline_pass_rate=baseline.pass_rate,
        current_pass_rate=current.pass_rate,
        changes=changes,
    )


def load_run(path: str | Path) -> RunResult:
    """Read a run back from a JSON report written by ``evalkit run --report x.json``."""
    path = Path(path)
    try:
        return RunResult.model_validate_json(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise EvalkitError(f"cannot read run {path}: {exc.strerror or exc}") from exc
    except ValidationError as exc:
        raise EvalkitError(f"{path} is not an evalkit JSON report: {exc}") from exc

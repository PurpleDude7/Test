"""Result models produced by the runner and consumed by reporters."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field, computed_field


class ScoreResult(BaseModel):
    """The verdict of one scorer on one output. ``score`` is normalised to [0, 1]."""

    scorer: str
    score: float = Field(ge=0, le=1)
    passed: bool
    reason: str = ""


class Usage(BaseModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class CaseResult(BaseModel):
    case_id: str
    prompt: str
    expected: str | None
    output: str | None
    tags: list[str] = Field(default_factory=list)
    scores: list[ScoreResult] = Field(default_factory=list)
    error: str | None = None
    latency_s: float = 0.0
    usage: Usage | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        return self.error is None and all(s.passed for s in self.scores)


class ScorerSummary(BaseModel):
    scorer: str
    count: int
    passed: int
    mean_score: float


class RunResult(BaseModel):
    suite: str
    provider: str
    model: str | None
    threshold: float
    started_at: datetime
    duration_s: float
    cases: list[CaseResult]

    @computed_field  # type: ignore[prop-decorator]
    @property
    def total(self) -> int:
        return len(self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed_count(self) -> int:
        return sum(c.passed for c in self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def error_count(self) -> int:
        return sum(c.error is not None for c in self.cases)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def pass_rate(self) -> float:
        return self.passed_count / self.total if self.cases else 0.0

    @computed_field  # type: ignore[prop-decorator]
    @property
    def passed(self) -> bool:
        """Whether the run clears the CI gate."""
        return self.pass_rate >= self.threshold

    def scorer_summaries(self) -> list[ScorerSummary]:
        buckets: dict[str, list[ScoreResult]] = {}
        for case in self.cases:
            for s in case.scores:
                buckets.setdefault(s.scorer, []).append(s)
        return [
            ScorerSummary(
                scorer=name,
                count=len(items),
                passed=sum(i.passed for i in items),
                mean_score=sum(i.score for i in items) / len(items),
            )
            for name, items in buckets.items()
        ]

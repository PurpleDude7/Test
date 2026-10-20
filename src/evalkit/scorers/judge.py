"""Ask a model to grade the output against a rubric."""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import Field, model_validator

from evalkit.models import Message
from evalkit.results import ScoreResult
from evalkit.scorers.base import ScoreContext, Scorer, register_scorer

DEFAULT_TEMPLATE = """\
You are an impartial evaluator. Grade the RESPONSE to the INPUT against the CRITERIA.

<input>
{input}
</input>
{expected_block}
<response>
{output}
</response>

<criteria>
{criteria}
</criteria>

Reply with only a JSON object, no other text:
{{"score": <integer from {min_score} to {max_score}>, "reasoning": "<one sentence>"}}
"""

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)
_SCORE_FALLBACK = re.compile(r"score\W{0,3}(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def parse_verdict(text: str) -> tuple[float, str] | None:
    """Pull ``(score, reasoning)`` out of a judge reply, tolerating chatty models."""
    match = _JSON_OBJECT.search(text)
    if match:
        try:
            data: Any = json.loads(match.group(0))
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict) and isinstance(data.get("score"), int | float):
            return float(data["score"]), str(data.get("reasoning", ""))
    fallback = _SCORE_FALLBACK.search(text)
    if fallback:
        return float(fallback.group(1)), text.strip()
    return None


@register_scorer("llm_judge")
class LLMJudge(Scorer):
    """Grades on an integer scale; passes when the normalised score >= ``threshold``.

    By default the provider under test also acts as judge. Set ``provider`` (and
    optionally ``model``/``provider_options``) to use a different one.
    """

    name = "llm_judge"
    criteria: str = Field(min_length=1)
    min_score: int = 1
    max_score: int = 5
    threshold: float = Field(default=0.75, ge=0, le=1)
    provider: str | None = None
    model: str | None = None
    provider_options: dict[str, Any] = Field(default_factory=dict)
    prompt_template: str = DEFAULT_TEMPLATE
    """``str.format`` template with input, output, expected_block, criteria, min/max_score."""
    temperature: float | None = 0.0

    @model_validator(mode="after")
    def _check_scale(self) -> LLMJudge:
        if self.max_score <= self.min_score:
            raise ValueError("max_score must be greater than min_score")
        return self

    def score(self, ctx: ScoreContext) -> ScoreResult:
        expected_block = (
            f"\n<expected>\n{ctx.case.expected}\n</expected>\n" if ctx.case.expected else ""
        )
        prompt = self.prompt_template.format(
            input=ctx.case.prompt_text(),
            output=ctx.output,
            expected_block=expected_block,
            criteria=self.criteria,
            min_score=self.min_score,
            max_score=self.max_score,
        )
        overrides = dict(self.provider_options)
        if self.model:
            overrides["model"] = self.model
        judge = ctx.get_provider(self.provider, overrides)
        reply = judge.complete(
            [Message(role="user", content=prompt)], temperature=self.temperature
        ).text
        verdict = parse_verdict(reply)
        if verdict is None:
            return self.result(0.0, False, f"could not parse judge reply: {reply[:200]!r}")
        raw, reasoning = verdict
        raw = min(max(raw, self.min_score), self.max_score)
        normalised = (raw - self.min_score) / (self.max_score - self.min_score)
        ok = normalised >= self.threshold
        return self.result(normalised, ok, f"{raw:g}/{self.max_score}: {reasoning}".strip())

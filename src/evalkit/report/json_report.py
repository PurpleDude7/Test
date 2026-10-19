"""Machine-readable report: the RunResult plus per-scorer summaries."""

from __future__ import annotations

import json

from evalkit.compare import Comparison
from evalkit.results import RunResult


def render_json(result: RunResult, comparison: Comparison | None = None) -> str:
    payload = result.model_dump(mode="json")
    payload["scorers"] = [s.model_dump(mode="json") for s in result.scorer_summaries()]
    if comparison is not None:
        payload["comparison"] = comparison.model_dump(mode="json")
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

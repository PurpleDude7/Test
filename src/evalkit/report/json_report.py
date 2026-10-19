"""Machine-readable report: the RunResult plus per-scorer summaries."""

from __future__ import annotations

import json

from evalkit.results import RunResult


def render_json(result: RunResult) -> str:
    payload = result.model_dump(mode="json")
    payload["scorers"] = [s.model_dump(mode="json") for s in result.scorer_summaries()]
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"

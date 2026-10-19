"""Report writers."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from evalkit.compare import Comparison
from evalkit.report.html_report import render_html
from evalkit.report.json_report import render_json
from evalkit.results import RunResult

__all__ = ["render_html", "render_json", "write_report"]

_RENDERERS: dict[str, Callable[[RunResult, Comparison | None], str]] = {
    ".json": render_json,
    ".html": render_html,
    ".htm": render_html,
}


def write_report(result: RunResult, path: str | Path, comparison: Comparison | None = None) -> Path:
    """Write ``result`` to ``path``; the format follows the file extension.

    ``comparison`` (from :func:`evalkit.compare.compare_runs`) adds a baseline section.
    """
    path = Path(path)
    renderer = _RENDERERS.get(path.suffix.lower())
    if renderer is None:
        supported = ", ".join(sorted(_RENDERERS))
        raise ValueError(f"unsupported report extension {path.suffix!r} (use {supported})")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(renderer(result, comparison), encoding="utf-8")
    return path

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalkit.report import render_html, render_json, write_report
from evalkit.results import CaseResult, RunResult, ScoreResult


@pytest.fixture
def result() -> RunResult:
    return RunResult(
        suite="demo <suite>",
        provider="fake",
        model="fake-1",
        threshold=0.5,
        started_at=datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC),
        duration_s=1.25,
        cases=[
            CaseResult(
                case_id="good",
                prompt="2+2?",
                expected="4",
                output="4",
                tags=["math"],
                scores=[ScoreResult(scorer="exact_match", score=1.0, passed=True)],
            ),
            CaseResult(
                case_id="xss",
                prompt="<script>alert(1)</script>",
                expected=None,
                output="<b>bold</b>",
                scores=[
                    ScoreResult(scorer="exact_match", score=0.0, passed=False, reason="nope & no"),
                    ScoreResult(scorer="contains", score=0.5, passed=False),
                ],
            ),
            CaseResult(
                case_id="err", prompt="p", expected="x", output=None, error="ProviderError: boom"
            ),
        ],
    )


def test_summary_properties(result: RunResult) -> None:
    assert result.total == 3
    assert result.passed_count == 1
    assert result.error_count == 1
    assert result.pass_rate == pytest.approx(1 / 3)
    assert not result.passed
    summaries = {s.scorer: s for s in result.scorer_summaries()}
    assert summaries["exact_match"].count == 2 and summaries["exact_match"].passed == 1
    assert summaries["exact_match"].mean_score == 0.5


def test_json_report_round_trips(result: RunResult) -> None:
    data = json.loads(render_json(result))
    assert data["suite"] == "demo <suite>"
    assert data["pass_rate"] == pytest.approx(1 / 3)
    assert data["passed"] is False
    assert [c["passed"] for c in data["cases"]] == [True, False, False]
    assert {s["scorer"] for s in data["scorers"]} == {"exact_match", "contains"}
    assert RunResult.model_validate(data).cases[1].scores[0].reason == "nope & no"


def test_html_report_is_escaped_and_self_contained(result: RunResult) -> None:
    html = render_html(result)
    assert html.startswith("<!doctype html>")
    assert "<script>" not in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "&lt;b&gt;bold&lt;/b&gt;" in html
    assert "demo &lt;suite&gt;" in html
    assert "nope &amp; no" in html
    assert "GATE FAILED" in html
    assert "ProviderError: boom" in html
    assert "http://" not in html and "https://" not in html


def test_write_report_picks_format_from_extension(result: RunResult, tmp_path: Path) -> None:
    html = write_report(result, tmp_path / "nested" / "out.html")
    js = write_report(result, tmp_path / "out.json")
    assert html.read_text().startswith("<!doctype html>")
    assert json.loads(js.read_text())["total"] == 3
    with pytest.raises(ValueError, match="unsupported"):
        write_report(result, tmp_path / "out.txt")

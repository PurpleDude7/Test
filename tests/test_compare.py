from __future__ import annotations

import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from evalkit.cli import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, main
from evalkit.compare import Change, compare_runs, load_run, mean_score
from evalkit.errors import EvalkitError
from evalkit.report import render_html, render_json, write_report
from evalkit.results import CaseResult, RunResult, ScoreResult


def case(case_id: str, passed: bool, score: float | None = None) -> CaseResult:
    value = score if score is not None else (1.0 if passed else 0.0)
    return CaseResult(
        case_id=case_id,
        prompt="p",
        expected=None,
        output="o",
        scores=[
            ScoreResult(
                scorer="exact_match", score=value, passed=passed, reason="" if passed else "bad"
            )
        ],
    )


def run(*cases: CaseResult, threshold: float = 0.0) -> RunResult:
    return RunResult(
        suite="s",
        provider="fake",
        model=None,
        threshold=threshold,
        started_at=datetime(2026, 1, 1, tzinfo=UTC),
        duration_s=0.1,
        cases=list(cases),
    )


BASELINE = run(case("keep", True), case("breaks", True), case("heals", False), case("gone", True))
CURRENT = run(case("keep", True), case("breaks", False), case("heals", True), case("fresh", False))


def test_classifies_every_case() -> None:
    comparison = compare_runs(BASELINE, CURRENT)
    assert [(c.case_id, c.change) for c in comparison.changes] == [
        ("keep", Change.STILL_PASSING),
        ("breaks", Change.REGRESSED),
        ("heals", Change.FIXED),
        ("fresh", Change.NEW),
        ("gone", Change.REMOVED),
    ]
    assert comparison.regressions == 1 and comparison.fixes == 1
    assert comparison.pass_rate_delta == pytest.approx(0.5 - 0.75)
    breaks = comparison.by_change(Change.REGRESSED)[0]
    assert (breaks.baseline_score, breaks.current_score) == (1.0, 0.0)
    assert breaks.detail == "exact_match: bad"
    gone = comparison.by_change(Change.REMOVED)[0]
    assert gone.current_score is None


def test_still_failing_and_errors() -> None:
    errored = CaseResult(case_id="a", prompt="p", expected=None, output=None, error="boom")
    comparison = compare_runs(run(case("a", False)), run(errored))
    (change,) = comparison.changes
    assert change.change is Change.STILL_FAILING and change.detail == "boom"
    assert comparison.regressions == 0
    assert mean_score(errored) == 0.0
    assert mean_score(case("x", False, score=0.4)) == pytest.approx(0.4)


def test_load_run_round_trips_json_report(tmp_path: Path) -> None:
    path = write_report(CURRENT, tmp_path / "run.json")
    loaded = load_run(path)
    assert loaded.cases == CURRENT.cases
    assert loaded.pass_rate == CURRENT.pass_rate


def test_load_run_errors(tmp_path: Path) -> None:
    with pytest.raises(EvalkitError, match="cannot read"):
        load_run(tmp_path / "missing.json")
    bad = tmp_path / "bad.json"
    bad.write_text('{"not": "a report"}')
    with pytest.raises(EvalkitError, match="not an evalkit JSON report"):
        load_run(bad)


def test_reports_include_comparison() -> None:
    comparison = compare_runs(BASELINE, CURRENT)
    data = json.loads(render_json(CURRENT, comparison))
    assert data["comparison"]["regressions"] == 1
    assert "comparison" not in json.loads(render_json(CURRENT))

    html = render_html(CURRENT, comparison)
    assert "Compared with baseline" in html and "REGRESSED" in html and "FIXED" in html
    assert "1 regressed" in html
    assert "Compared with baseline" not in render_html(CURRENT)


def test_regressions_fail_the_html_gate_even_when_pass_rate_clears() -> None:
    assert CURRENT.passed  # threshold 0
    assert "GATE FAILED" in render_html(CURRENT, compare_runs(BASELINE, CURRENT))
    assert "GATE PASSED" in render_html(CURRENT, compare_runs(CURRENT, CURRENT))


def cli(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(list(argv), out=out), out.getvalue()


def test_compare_command(tmp_path: Path) -> None:
    base = write_report(BASELINE, tmp_path / "base.json")
    cur = write_report(CURRENT, tmp_path / "cur.json")
    html = tmp_path / "diff.html"

    code, out = cli("compare", str(base), str(cur), "--report", str(html))
    assert code == EXIT_GATE_FAILED
    assert "1 regressed, 1 fixed" in out
    assert "REGRESSED  breaks  - exact_match: bad" in out
    assert "REMOVED    gone" in out
    assert "keep" not in out  # unchanged cases are not listed
    assert "Compared with baseline" in html.read_text()

    code, out = cli("compare", str(base), str(base))
    assert code == EXIT_OK
    assert "0 regressed, 0 fixed" in out


def test_run_with_baseline_fails_on_regression(example_suite_path: Path, tmp_path: Path) -> None:
    base = tmp_path / "base.json"
    assert cli("run", str(example_suite_path), "-q", "-r", str(base))[0] == EXIT_OK

    # Break exactly one case; 4/5 still clears the suite's 0.8 threshold.
    responses = (
        "responses={capital of France: Lyon, capital of Japan: Tokyo, "
        'ISO date: "2024-03-01", photosynthesis: glucose and oxygen, haiku: "a\\nb\\nc"}'
    )
    code, out = cli("run", str(example_suite_path), "-q", "-O", responses)
    assert code == EXIT_OK
    code, out = cli("run", str(example_suite_path), "-q", "-O", responses, "-b", str(base))
    assert code == EXIT_GATE_FAILED
    assert "-> PASSED" in out  # the pass-rate gate alone is satisfied
    assert "REGRESSED  capital-france" in out

    assert cli("run", str(example_suite_path), "-q", "-b", str(base))[0] == EXIT_OK


def test_run_with_unreadable_baseline_is_a_usage_error(
    example_suite_path: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code, out = cli("run", str(example_suite_path), "-b", str(tmp_path / "nope.json"))
    assert code == EXIT_USAGE
    assert "Running" not in out  # failed before any provider calls
    assert "cannot read run" in capsys.readouterr().err

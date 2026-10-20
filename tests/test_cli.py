from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from evalkit.cli import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, main


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    code = main(list(argv), out=out)
    return code, out.getvalue()


def test_run_example_with_html_and_json_reports(example_suite_path: Path, tmp_path: Path) -> None:
    html, js = tmp_path / "out.html", tmp_path / "out.json"
    code, out = run(
        "run", str(example_suite_path), "--provider", "fake", "--report", str(html), "-r", str(js)
    )
    assert code == EXIT_OK, out
    assert "5/5 passed" in out and "PASSED" in out
    assert "GATE PASSED" in html.read_text()
    assert json.loads(js.read_text())["pass_rate"] == 1.0


def test_gate_failure_exits_nonzero(example_suite_path: Path) -> None:
    # Clearing the canned replies makes the fake echo prompts, so most cases fail.
    code, out = run("run", str(example_suite_path), "-O", "responses={}", "--quiet")
    assert code == EXIT_GATE_FAILED
    assert "FAILED" in out
    assert "  FAIL " not in out  # --quiet hides per-case lines


def test_threshold_override(example_suite_path: Path) -> None:
    code, _ = run("run", str(example_suite_path), "-O", "responses={}", "--threshold", "0")
    assert code == EXIT_OK


def test_tag_filter_and_model_override(example_suite_path: Path) -> None:
    code, out = run("run", str(example_suite_path), "--tag", "geography", "--model", "fake-2")
    assert code == EXIT_OK
    assert "2 cases on fake (fake-2)" in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["run", "missing.yaml"], "cannot read"),
        (["run", "{suite}", "--provider", "nope"], "unknown provider"),
        (["run", "{suite}", "--tag", "nope"], "no cases match"),
        (["run", "{suite}", "-O", "bogus=1"], "invalid config"),
        (["run", "{suite}", "--report", "{tmp}/out.txt"], "unsupported report extension"),
        (["run", "{suite}", "-j", "0"], "concurrency"),
    ],
)
def test_usage_errors_exit_2(
    argv: list[str],
    message: str,
    example_suite_path: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    filled = [a.format(suite=example_suite_path, tmp=tmp_path) for a in argv]
    code, _ = run(*filled)
    assert code == EXIT_USAGE
    assert message in capsys.readouterr().err


def test_no_provider_configured(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    suite = tmp_path / "s.yaml"
    suite.write_text(
        "name: s\ncases:\n  - {id: a, input: x, expected: x, scorers: [exact_match]}\n"
    )
    assert run("run", str(suite))[0] == EXIT_USAGE
    assert "no provider" in capsys.readouterr().err


def test_validate_and_list(example_suite_path: Path) -> None:
    code, out = run("validate", str(example_suite_path))
    assert code == EXIT_OK and "OK (5 cases)" in out
    code, out = run("list")
    assert code == EXIT_OK
    assert "anthropic" in out and "llm_judge" in out


def test_bad_option_syntax_is_an_argparse_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["run", "s.yaml", "-O", "novalue"])
    assert exc.value.code == 2
    assert "KEY=VALUE" in capsys.readouterr().err

from __future__ import annotations

import io
import json
from pathlib import Path

import pytest

from lectureiq.cli import main


def run(*argv: str) -> tuple[int, str]:
    out = io.StringIO()
    return main(list(argv), out=out), out.getvalue()


def test_analyze_sample_writes_all_outputs(tmp_path: Path) -> None:
    js, md, cards = tmp_path / "pack.json", tmp_path / "guide.md", tmp_path / "cards.csv"
    code, out = run(
        "analyze", "@sample", "--json", str(js), "--markdown", str(md), "--csv", str(cards)
    )
    assert code == 0
    assert "Introduction to Machine Learning" in out and "(offline engine)" in out
    assert "1. Learning · Spam · Task [0:02]" in out
    assert json.loads(js.read_text())["engine"] == "offline"
    assert md.read_text().startswith("# Introduction to Machine Learning")
    assert cards.read_text().count("\n") == 20


def test_analyze_file_uses_its_name_as_title(tmp_path: Path) -> None:
    path = tmp_path / "week_3_entropy.txt"
    path.write_text("Entropy is a measure of uncertainty in a random variable. " * 3)
    code, out = run("analyze", str(path))
    assert code == 0 and "week 3 entropy" in out


def test_analyze_reads_stdin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO("Entropy is a measure of uncertainty in data."))
    code, out = run("analyze", "-", "--title", "Piped")
    assert code == 0 and "Piped" in out


def test_ask_prints_answer_and_citation() -> None:
    code, out = run("ask", "@sample", "What does the learning rate control?")
    assert code == 0
    assert out.startswith("The learning rate controls how big each nudge is.")
    assert "↳ part" in out and " at 6:12" in out


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["analyze", "missing.vtt"], "cannot read"),
        (["analyze", "@sample", "--model", "x"], "only apply to LLM providers"),
        (["analyze", "@sample", "--provider", "nope"], "unknown provider"),
        (["ask", "@sample", "  "], "question is empty"),
    ],
)
def test_errors_exit_2(argv: list[str], message: str, capsys: pytest.CaptureFixture[str]) -> None:
    code, _ = run(*argv)
    assert code == 2
    assert message in capsys.readouterr().err


def test_llm_provider_path_runs_through_the_cache(tmp_path: Path) -> None:
    # The fake echoes each prompt back; the JSON template inside the prompt happens to be
    # valid output, so the whole LLM pipeline runs end to end without a network.
    code, out = run("analyze", "@sample", "--provider", "fake", "--cache", str(tmp_path))
    assert code == 0 and "(fake engine)" in out
    assert len(list(tmp_path.rglob("*.json"))) == 4  # 3 chunks + the outline call
    code, again = run("analyze", "@sample", "--provider", "fake", "--cache", str(tmp_path))
    assert code == 0 and again == out

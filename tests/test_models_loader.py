from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evalkit import Message, Suite, SuiteError, TestCase, load_suite, parse_suite


def minimal(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "s",
        "defaults": {"scorers": [{"type": "exact_match"}]},
        "cases": [{"id": "a", "input": "hi", "expected": "hello"}],
    }
    data.update(overrides)
    return data


def test_example_suite_loads(example_suite_path: Path) -> None:
    suite = load_suite(example_suite_path)
    assert suite.name == "geography-and-format"
    assert suite.threshold == 0.8
    assert len(suite.cases) == 5
    assert suite.resolve_provider("judge")[0] == "fake"


def test_defaults_apply_when_case_has_no_scorers() -> None:
    suite = parse_suite(minimal())
    assert [s.type for s in suite.scorers_for(suite.cases[0])] == ["exact_match"]


def test_scorer_spec_collects_params_and_accepts_bare_strings() -> None:
    suite = parse_suite(
        minimal(
            cases=[
                {
                    "id": "a",
                    "input": "x",
                    "expected": "x",
                    "scorers": ["exact_match", {"type": "regex", "pattern": "x"}],
                }
            ]
        )
    )
    specs = suite.scorers_for(suite.cases[0])
    assert specs[0].type == "exact_match" and specs[0].params == {}
    assert specs[1].params == {"pattern": "x"}


@pytest.mark.parametrize(
    ("data", "message"),
    [
        (minimal(cases=[]), "cases"),
        (minimal(threshold=1.5), "threshold"),
        (minimal(cases=[{"id": "a", "input": "x"}, {"id": "a", "input": "y"}]), "duplicate"),
        (minimal(defaults={}), "no scorers"),
        (minimal(defaults={"scorers": [{"type": "nope"}]}), "unknown scorer 'nope'"),
        (minimal(defaults={"scorers": [{"type": "regex"}]}), "pattern"),
        (minimal(defaults={"scorers": [{"type": "regex", "pattern": "("}]}), "invalid regex"),
        (minimal(defaults={"scorers": [{"type": "exact_match", "typo": 1}]}), "typo"),
        (minimal(surprise=True), "surprise"),
        ([1, 2], "mapping"),
    ],
)
def test_invalid_suites_are_rejected(data: Any, message: str) -> None:
    with pytest.raises(SuiteError, match=message):
        parse_suite(data)


def test_load_suite_reports_missing_file_and_bad_yaml(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="cannot read"):
        load_suite(tmp_path / "missing.yaml")
    bad = tmp_path / "bad.yaml"
    bad.write_text("name: [unclosed\n")
    with pytest.raises(SuiteError, match="invalid YAML"):
        load_suite(bad)


def test_messages_prepend_system_prompt() -> None:
    case = TestCase(id="a", input="hi")
    assert case.messages("be brief") == [
        Message(role="system", content="be brief"),
        Message(role="user", content="hi"),
    ]
    own = TestCase(id="b", input="hi", system="mine")
    assert own.messages("default")[0].content == "mine"


def test_chat_input_keeps_existing_system_and_reports_last_user_turn() -> None:
    case = TestCase.model_validate(
        {
            "id": "a",
            "input": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "first"},
                {"role": "assistant", "content": "ok"},
                {"role": "user", "content": "second"},
            ],
        }
    )
    assert [m.role for m in case.messages("ignored")] == ["system", "user", "assistant", "user"]
    assert case.prompt_text() == "second"


def test_filter_tags() -> None:
    suite = Suite.model_validate(
        minimal(
            cases=[
                {"id": "a", "input": "x", "tags": ["fast"]},
                {"id": "b", "input": "y", "tags": ["slow"]},
            ]
        )
    )
    assert [c.id for c in suite.filter_tags(["fast"]).cases] == ["a"]
    assert suite.filter_tags([]) is suite
    with pytest.raises(ValueError, match="no cases"):
        suite.filter_tags(["missing"])


def test_resolve_provider_alias_and_plain_config() -> None:
    suite = Suite.model_validate(
        minimal(providers={"openai": {"model": "m"}, "judge": {"type": "anthropic", "model": "j"}})
    )
    assert suite.resolve_provider("openai") == ("openai", {"model": "m"})
    assert suite.resolve_provider("judge") == ("anthropic", {"model": "j"})
    assert suite.resolve_provider("fake") == ("fake", {})

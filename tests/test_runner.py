from __future__ import annotations

import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

from evalkit import (
    CaseResult,
    Completion,
    FakeProvider,
    Message,
    Provider,
    create_provider,
    load_suite,
    parse_suite,
    run_suite,
)


def suite(cases: list[dict[str, Any]], **extra: Any) -> Any:
    return parse_suite(
        {"name": "t", "defaults": {"scorers": ["exact_match"]}, "cases": cases, **extra}
    )


def test_example_suite_passes_with_fake(example_suite_path: Path) -> None:
    s = load_suite(example_suite_path)
    _, config = s.resolve_provider("fake")
    result = run_suite(s, FakeProvider.from_config(config))
    assert [c.case_id for c in result.cases] == [c.id for c in s.cases]
    failed = [(c.case_id, c.error, c.scores) for c in result.cases if not c.passed]
    assert failed == []
    assert result.pass_rate == 1.0 and result.passed


def test_pass_rate_and_threshold() -> None:
    s = suite(
        [
            {"id": "a", "input": "yes", "expected": "yes"},
            {"id": "b", "input": "no", "expected": "yes"},
        ],
        threshold=0.5,
    )
    result = run_suite(s, FakeProvider.from_config({}))
    assert result.passed_count == 1 and result.pass_rate == 0.5
    assert result.passed
    assert not run_suite(s, FakeProvider.from_config({}), threshold=0.6).passed


def test_provider_errors_fail_the_case_not_the_run() -> None:
    s = suite(
        [{"id": "a", "input": "boom", "expected": "x"}, {"id": "b", "input": "x", "expected": "x"}]
    )
    result = run_suite(s, FakeProvider.from_config({"fail_on": ["boom"]}))
    a, b = result.cases
    assert not a.passed and a.error is not None and "boom" in a.error
    assert a.output is None and a.scores == []
    assert b.passed
    assert result.error_count == 1


def test_scorer_errors_are_recorded() -> None:
    s = suite([{"id": "a", "input": "x"}])  # exact_match without `expected`
    case = run_suite(s, FakeProvider.from_config({})).cases[0]
    assert not case.passed and case.error is not None and "expected" in case.error


def test_unexpected_scorer_exceptions_are_recorded() -> None:
    s = suite(
        [
            {
                "id": "a",
                "input": "x",
                "expected": "x",
                "scorers": [{"type": "embedding_similarity", "provider": "anthropic"}],
            }
        ]
    )
    case = run_suite(s, FakeProvider.from_config({})).cases[0]
    assert case.error is not None and "does not support embeddings" in case.error


def test_defaults_are_sent_to_provider() -> None:
    seen: list[tuple[list[Message], float | None, int | None]] = []

    class Spy(FakeProvider):
        def complete(
            self,
            messages: Sequence[Message],
            *,
            temperature: float | None = None,
            max_tokens: int | None = None,
        ) -> Completion:
            seen.append((list(messages), temperature, max_tokens))
            return super().complete(messages)

    s = suite(
        [{"id": "a", "input": "x", "expected": "x"}],
        defaults={"system": "sys", "temperature": 0.3, "max_tokens": 9, "scorers": ["exact_match"]},
    )
    run_suite(s, Spy.from_config({}))
    messages, temperature, max_tokens = seen[0]
    assert [m.role for m in messages] == ["system", "user"]
    assert (temperature, max_tokens) == (0.3, 9)


def test_concurrency_preserves_order_and_reports_progress() -> None:
    barrier = threading.Barrier(3, timeout=5)

    class Slow(FakeProvider):
        def complete(self, messages: Sequence[Message], **kwargs: Any) -> Completion:
            barrier.wait()  # proves three cases are in flight at once
            return super().complete(messages)

    s = suite([{"id": str(i), "input": str(i), "expected": str(i)} for i in range(3)])
    progress: list[CaseResult] = []
    result = run_suite(s, Slow.from_config({}), concurrency=3, on_result=progress.append)
    assert [c.case_id for c in result.cases] == ["0", "1", "2"]
    assert sorted(c.case_id for c in progress) == ["0", "1", "2"]


def test_named_providers_are_built_from_suite_config_and_cached(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[tuple[str, dict[str, Any]]] = []

    def tracking(name: str, config: Any = None) -> Provider:
        created.append((name, dict(config or {})))
        return create_provider(name, config)

    monkeypatch.setattr("evalkit.runner.create_provider", tracking)
    judge = {"type": "llm_judge", "criteria": "ok", "provider": "judge"}
    s = suite(
        [{"id": str(i), "input": "x", "scorers": [judge]} for i in range(3)],
        providers={"judge": {"type": "fake", "default": '{"score": 5}'}},
    )
    result = run_suite(s, FakeProvider.from_config({}), concurrency=1)
    assert result.pass_rate == 1.0
    assert created == [("fake", {"default": '{"score": 5}'})]


def test_rejects_bad_concurrency() -> None:
    with pytest.raises(ValueError, match="concurrency"):
        run_suite(suite([{"id": "a", "input": "x"}]), FakeProvider.from_config({}), concurrency=0)

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from pathlib import Path
from typing import Any

import pytest

from evalkit import FakeProvider, Provider, ScoreContext, TestCase

EXAMPLE_SUITE = Path(__file__).resolve().parent.parent / "examples" / "suite.yaml"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Fail loudly if any test tries to open a real socket."""
    import socket

    def guard(*args: object, **kwargs: object) -> None:
        raise RuntimeError("tests must not touch the network")

    monkeypatch.setattr(socket.socket, "connect", guard)
    yield


@pytest.fixture
def example_suite_path() -> Path:
    return EXAMPLE_SUITE


MakeContext = Callable[..., ScoreContext]


@pytest.fixture
def make_ctx() -> MakeContext:
    """Build a ScoreContext around a FakeProvider (or a lookup you supply)."""

    def factory(
        output: str,
        expected: str | None = None,
        *,
        input: str = "prompt",
        provider: Provider | None = None,
        named: Mapping[str, Provider] | None = None,
    ) -> ScoreContext:
        primary = provider or FakeProvider.from_config({})
        named = dict(named or {})

        def lookup(name: str | None, overrides: Mapping[str, Any]) -> Provider:
            return primary if name is None else named[name]

        case = TestCase(id="c1", input=input, expected=expected)
        return ScoreContext(case=case, output=output, get_provider=lookup)

    return factory


@pytest.fixture
def sample_vtt() -> str:
    from lectureiq.sample import sample_transcript

    return sample_transcript()

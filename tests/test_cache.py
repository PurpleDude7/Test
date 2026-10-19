from __future__ import annotations

import io
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import httpx
import pytest

from evalkit import (
    CachingProvider,
    Completion,
    FakeProvider,
    Message,
    ProviderError,
    ResponseCache,
    load_run,
    load_suite,
    parse_suite,
    run_suite,
)
from evalkit.cli import EXIT_OK, main
from evalkit.providers.openai_compat import OpenAICompatibleConfig, OpenAICompatibleProvider
from evalkit.results import Usage

HELLO = [Message(role="user", content="hello")]


def make(tmp_path: Path, **config: Any) -> tuple[FakeProvider, CachingProvider]:
    inner = FakeProvider.from_config(config)
    return inner, CachingProvider(inner, ResponseCache(tmp_path))


def test_second_identical_request_is_served_from_cache(tmp_path: Path) -> None:
    inner, cached = make(tmp_path, default="hi")
    first = cached.complete(HELLO, temperature=0.0)
    second = cached.complete(HELLO, temperature=0.0)
    assert (first.text, first.cached) == ("hi", False)
    assert (second.text, second.cached) == ("hi", True)
    assert second.usage == first.usage
    assert len(inner.calls) == 1
    assert cached.name == "fake" and cached.model == "fake-1"


def test_cache_survives_new_provider_instances(tmp_path: Path) -> None:
    make(tmp_path, default="hi")[1].complete(HELLO)
    inner, cached = make(tmp_path, default="hi")
    assert cached.complete(HELLO).cached
    assert inner.calls == []


@pytest.mark.parametrize(
    ("messages", "kwargs"),
    [
        ([Message(role="user", content="hello!")], {}),
        ([Message(role="system", content="s"), *HELLO], {}),
        (HELLO, {"temperature": 0.7}),
        (HELLO, {"max_tokens": 5}),
    ],
)
def test_request_changes_miss(tmp_path: Path, messages: list[Message], kwargs: Any) -> None:
    inner, cached = make(tmp_path)
    cached.complete(HELLO)
    assert not cached.complete(messages, **kwargs).cached
    assert len(inner.calls) == 2


def test_provider_identity_is_part_of_the_key(tmp_path: Path) -> None:
    make(tmp_path, default="one")[1].complete(HELLO)
    # A different fake config, model, or endpoint must not reuse the reply.
    assert make(tmp_path, default="two")[1].complete(HELLO).text == "two"
    assert not make(tmp_path, default="one", model="other")[1].complete(HELLO).cached

    a = OpenAICompatibleProvider(OpenAICompatibleConfig(base_url="http://a/v1"))
    b = OpenAICompatibleProvider(OpenAICompatibleConfig(base_url="http://b/v1/"))
    assert a.cache_identity() != b.cache_identity()
    assert "api_key" not in json.dumps(
        OpenAICompatibleProvider(OpenAICompatibleConfig(api_key="sk-secret")).cache_identity()
    )


def test_errors_are_not_cached(tmp_path: Path) -> None:
    inner, cached = make(tmp_path, fail_on=["hello"])
    for _ in range(2):
        with pytest.raises(ProviderError):
            cached.complete(HELLO)
    assert len(inner.calls) == 2
    assert not any(tmp_path.rglob("*.json"))


def test_corrupt_entries_are_misses_and_get_rewritten(tmp_path: Path) -> None:
    cache = ResponseCache(tmp_path)
    inner, cached = make(tmp_path, default="ok")
    cached.complete(HELLO)
    (entry,) = tmp_path.rglob("*.json")
    entry.write_text("{not json")
    assert not cached.complete(HELLO).cached
    assert cached.complete(HELLO).cached
    key = cache.key(inner.cache_identity(), HELLO, None, None)
    assert cache.get(key) == Completion(
        text="ok", model="fake-1", usage=Usage(input_tokens=1, output_tokens=1)
    )


def test_concurrent_writers_leave_no_temp_files(tmp_path: Path) -> None:
    _, cached = make(tmp_path, default="x")
    threads = [threading.Thread(target=cached.complete, args=(HELLO,)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not [p for p in tmp_path.rglob(".tmp-*")]
    assert len(list(tmp_path.rglob("*.json"))) == 1


def test_embeddings_pass_through(tmp_path: Path) -> None:
    inner, cached = make(tmp_path)
    assert cached.embed(["a b"]) == inner.embed(["a b"])


def test_run_suite_caches_model_and_judge_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    suite = parse_suite(
        {
            "name": "s",
            "providers": {"judge": {"type": "fake", "default": '{"score": 5}'}},
            "cases": [
                {
                    "id": "a",
                    "input": "q",
                    "expected": "q",
                    "scorers": [
                        "exact_match",
                        {"type": "llm_judge", "criteria": "ok", "provider": "judge"},
                    ],
                }
            ],
        }
    )
    judge_calls: list[int] = []
    original = FakeProvider.complete

    def spy(self: FakeProvider, messages: Sequence[Message], **kwargs: Any) -> Completion:
        if self.config.default == '{"score": 5}':
            judge_calls.append(1)
        return original(self, messages, **kwargs)

    monkeypatch.setattr(FakeProvider, "complete", spy)
    cache = ResponseCache(tmp_path)
    primary = FakeProvider.from_config({})
    first = run_suite(suite, primary, cache=cache)
    second = run_suite(suite, primary, cache=cache)

    assert first.cached_count == 0 and not first.cases[0].cached
    assert second.cached_count == 1 and second.cases[0].cached
    assert first.pass_rate == second.pass_rate == 1.0
    assert len(primary.calls) == 1
    assert len(judge_calls) == 1  # the judge reply was cached too
    assert second.provider == "fake" and second.model == "fake-1"


def test_cli_cache_flag(example_suite_path: Path, tmp_path: Path) -> None:
    cache_dir = tmp_path / "cache"
    report = tmp_path / "out.html"
    args = ["run", str(example_suite_path), "--cache", str(cache_dir)]

    first = io.StringIO()
    assert main(args, out=first) == EXIT_OK
    assert "from cache" not in first.getvalue()

    second = io.StringIO()
    assert main([*args, "-r", str(report)], out=second) == EXIT_OK
    assert "5/5 replies served from cache" in second.getvalue()
    assert "PASS  capital-france (cached)" in second.getvalue()
    assert report.read_text().count(">cached<") == 5


def test_old_reports_without_cached_field_still_load(
    example_suite_path: Path, tmp_path: Path
) -> None:
    suite = load_suite(example_suite_path)
    result = run_suite(suite, FakeProvider.from_config(suite.resolve_provider("fake")[1]))
    data = json.loads(result.model_dump_json())
    for case in data["cases"]:
        del case["cached"]
    del data["cached_count"]
    path = tmp_path / "old.json"
    path.write_text(json.dumps(data))
    assert load_run(path).cached_count == 0


def test_no_network_used_by_http_identity() -> None:
    # cache_identity must be computable without a client or an API key.
    provider = OpenAICompatibleProvider(
        OpenAICompatibleConfig(), transport=httpx.MockTransport(lambda r: httpx.Response(500))
    )
    assert provider.cache_identity()["base_url"] == "https://api.openai.com/v1"

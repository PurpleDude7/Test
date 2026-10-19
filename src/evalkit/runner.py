"""Run a suite against a provider and collect results."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from evalkit.cache import CachingProvider, ResponseCache
from evalkit.errors import EvalkitError
from evalkit.models import Suite, TestCase
from evalkit.providers.base import Provider, create_provider
from evalkit.results import CaseResult, RunResult
from evalkit.scorers import ScoreContext, build_scorer

ProgressCallback = Callable[[CaseResult], None]


class _ProviderPool:
    """Resolves providers for scorers: the one under test, or named ones built on demand.

    Named providers are resolved through the suite's ``providers`` section, merged with
    per-scorer overrides, and cached so each distinct configuration is created once.
    """

    def __init__(self, primary: Provider, suite: Suite, cache: ResponseCache | None) -> None:
        self._primary = primary
        self._suite = suite
        self._response_cache = cache
        self._cache: dict[str, Provider] = {}
        self._lock = threading.Lock()

    def get(self, name: str | None, overrides: Mapping[str, Any]) -> Provider:
        if name is None and not overrides:
            return self._primary
        name = name or self._primary.name
        key = json.dumps([name, dict(overrides)], sort_keys=True, default=str)
        with self._lock:
            if key not in self._cache:
                kind, config = self._suite.resolve_provider(name)
                provider = create_provider(kind, {**config, **overrides})
                if self._response_cache is not None:
                    provider = CachingProvider(provider, self._response_cache)
                self._cache[key] = provider
            return self._cache[key]

    def close(self) -> None:
        for provider in self._cache.values():
            provider.close()


def _run_case(case: TestCase, suite: Suite, provider: Provider, pool: _ProviderPool) -> CaseResult:
    result = CaseResult(
        case_id=case.id,
        prompt=case.prompt_text(),
        expected=case.expected,
        output=None,
        tags=case.tags,
    )
    start = time.perf_counter()
    try:
        completion = provider.complete(
            case.messages(suite.defaults.system),
            temperature=suite.defaults.temperature,
            max_tokens=suite.defaults.max_tokens,
        )
    except Exception as exc:  # any provider failure fails the case, not the run
        result.error = f"{type(exc).__name__}: {exc}"
        result.latency_s = time.perf_counter() - start
        return result
    result.latency_s = time.perf_counter() - start
    result.output = completion.text
    result.usage = completion.usage
    result.cached = completion.cached

    ctx = ScoreContext(case=case, output=completion.text, get_provider=pool.get)
    for spec in suite.scorers_for(case):
        try:
            result.scores.append(build_scorer(spec).score(ctx))
        except EvalkitError as exc:
            result.error = f"scorer {spec.type!r}: {exc}"
            break
        except Exception as exc:
            result.error = f"scorer {spec.type!r}: {type(exc).__name__}: {exc}"
            break
    return result


def run_suite(
    suite: Suite,
    provider: Provider,
    *,
    threshold: float | None = None,
    concurrency: int = 4,
    on_result: ProgressCallback | None = None,
    cache: ResponseCache | None = None,
) -> RunResult:
    """Run every case in ``suite`` and score the outputs.

    ``threshold`` overrides the suite's own pass-rate gate. Provider and scorer errors are
    recorded on the case (which then fails) rather than aborting the run. With ``cache``,
    completions from the provider under test and from named providers (e.g. a judge) are
    reused across runs.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be >= 1")
    started_at = datetime.now(UTC)
    start = time.perf_counter()
    if cache is not None:
        provider = CachingProvider(provider, cache)
    pool = _ProviderPool(provider, suite, cache)

    def task(case: TestCase) -> CaseResult:
        res = _run_case(case, suite, provider, pool)
        if on_result is not None:
            on_result(res)
        return res

    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            cases = list(executor.map(task, suite.cases))
    finally:
        pool.close()

    return RunResult(
        suite=suite.name,
        provider=provider.name,
        model=provider.model,
        threshold=suite.threshold if threshold is None else threshold,
        started_at=started_at,
        duration_s=time.perf_counter() - start,
        cases=cases,
    )

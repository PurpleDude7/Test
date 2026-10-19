# CLAUDE.md

Guidance for AI assistants (and humans) working on **evalkit**, a Python library and CLI for
evaluating LLM outputs. User-facing docs live in [README.md](README.md); this file covers
how to work on the code.

## Quick reference

```bash
pip install -e ".[dev]"            # editable install with pytest, ruff, mypy, stubs
pytest                             # all tests; offline, takes about a second
pytest tests/test_scorers.py -k judge   # a subset
ruff check . && ruff format --check .   # lint + format check (ruff format . to fix)
mypy                               # strict type check of src/ and tests/
evalkit run examples/suite.yaml --provider fake --report out.html   # end-to-end smoke test
```

CI (`.github/workflows/ci.yml`) runs ruff, ruff format, and mypy on 3.11, then pytest and
the CLI smoke test on 3.11, 3.12 and 3.13. Run all four commands above before committing.

## Layout

```text
src/evalkit/
  models.py        YAML schema: Suite, TestCase, Message, ScorerSpec, Defaults
  loader.py        load_suite / parse_suite: YAML -> validated Suite (builds every scorer once)
  runner.py        run_suite: thread-pooled execution, _ProviderPool for named providers
  results.py       ScoreResult, CaseResult, RunResult (pass_rate, passed gate, summaries)
  registry.py      generic name -> class Registry with entry-point discovery
  errors.py        EvalkitError > SuiteError, RegistryError, ProviderError > NotSupportedError
  providers/       base.py (Provider ABC, registry), fake.py, openai_compat.py, anthropic.py,
                   _http.py (shared httpx client + retry/backoff)
  scorers/         base.py (Scorer ABC, ScoreContext, registry), basic.py, embedding.py, judge.py
  report/          json_report.py, html_report.py, write_report() picks format by extension
  cli.py           argparse; exit codes 0 = gate passed, 1 = gate failed, 2 = usage/config error
tests/             one module per area; conftest.py has the network guard and make_ctx fixture
examples/suite.yaml  runs green offline with the fake provider; used by tests and CI
```

## Conventions

- **Python 3.11+.** Do not use 3.12-only syntax (PEP 695 `class Foo[T]`, `type X = ...`).
  ruff targets py311 and mypy checks as 3.11.
- **Full type hints, mypy strict.** No untyped defs; avoid `Any` except at YAML/JSON
  boundaries. Use `# type: ignore[code]` only with a specific code.
- **pydantic v2 for anything parsed from user input** (suites, scorer params, provider
  configs) with `extra="forbid"` so typos fail at load time. Plain classes elsewhere.
- **Errors:** raise subclasses of `EvalkitError`. The CLI turns them into exit code 2 with a
  one-line message; never let a traceback reach CLI users for a config mistake.
- **Failures are results.** Provider or scorer errors during a run are recorded on the
  `CaseResult` (the case fails); they must not abort the run.
- **Thread safety.** The runner calls `Provider.complete()` from a thread pool. Guard
  shared mutable state (see `FakeProvider.calls`, `HTTPProvider.client`).
- **Style:** ruff (line length 100), double quotes, docstrings on public classes and
  functions. Comments explain why, not what. Keep dependencies minimal: core is httpx,
  pydantic, PyYAML; anything heavy goes in an optional extra and is imported lazily.
- **Security:** API keys come from environment variables (`api_key_env`); never put keys in
  suites, tests, or fixtures. The HTML report must escape every user-controlled string.

## Adding things

**A scorer:** subclass `Scorer` in `src/evalkit/scorers/`, declare options as pydantic
fields, set `name`, implement `score(ctx) -> ScoreResult` (use `self.result(...)`, which
clamps the score to [0, 1]), decorate with `@register_scorer("name")`, import it in
`scorers/__init__.py`, add tests in `tests/test_scorers.py`, and add a row to the scorer
table in README.md.

**A provider:** subclass `Provider` (or `HTTPProvider` for JSON-over-HTTPS APIs), define a
`ProviderConfig` subclass as `Config`, implement `complete()` (and `embed()` if supported),
register with `@register_provider("name")`, import it in `providers/__init__.py`, and test it
with `httpx.MockTransport` in `tests/test_providers.py`. Update the README provider table.

## Testing rules

- **No network, ever.** `tests/conftest.py` patches `socket.connect` to raise. Use
  `FakeProvider` for behaviour and `httpx.MockTransport` (passed as `transport=`) to test
  HTTP providers' request and response handling. No real API keys in tests.
- Use the `make_ctx` fixture to build a `ScoreContext` for scorer tests.
- `filterwarnings = ["error"]`: warnings fail the suite; fix the cause rather than
  filtering it.
- Patch `evalkit.providers._http.time.sleep` in retry tests so they stay fast.
- If you change `examples/suite.yaml`, keep it passing with `--provider fake`; tests and CI
  depend on it.
- Bug fixes come with a test that fails without the fix.

## Git workflow

- Work on a branch prefixed `claude/`; never push directly to the default branch.
- Conventional commit messages: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`.
- Push with `git push -u origin <branch>`; on network errors retry up to 4 times with
  exponential backoff (2s, 4s, 8s, 16s).
- Before committing: tests, ruff, ruff format, and mypy all pass; no secrets; README and
  this file updated if behaviour or conventions changed.

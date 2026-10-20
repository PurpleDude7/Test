# CLAUDE.md

Guidance for AI assistants (and humans) working on this repo. It holds two packages, built
and tested together:

- **evalkit** (`src/evalkit/`): a library and CLI for evaluating LLM outputs.
- **LectureIQ** (`src/lectureiq/`): an app that turns lecture transcripts and recordings into
  study material. It uses evalkit's providers and response cache, and its web dependencies
  (Starlette, uvicorn, python-multipart) are the optional `lectureiq` extra. It is also an
  evalkit provider (`lectureiq`, via an entry point), and `examples/lectureiq_qa.yaml` grades it.

User-facing docs live in [README.md](README.md); this file covers how to work on the code.

## Quick reference

```bash
pip install -e ".[dev]"            # editable install with pytest, ruff, mypy, stubs
pytest                             # all tests; offline, takes about a second
pytest tests/test_scorers.py -k judge   # a subset
ruff check . && ruff format --check .   # lint + format check (ruff format . to fix)
mypy                               # strict type check of src/ and tests/
evalkit run examples/suite.yaml --provider fake --report out.html   # end-to-end smoke test
lectureiq analyze @sample          # LectureIQ smoke test (offline engine, bundled lecture)
evalkit run examples/lectureiq_qa.yaml   # grade LectureIQ's answers (needs the entry point:
                                         # pip install -e ".[dev]")
lectureiq serve --db lectures.db   # web app on http://127.0.0.1:8000, lectures kept in SQLite
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
  compare.py       compare_runs / load_run: regressed/fixed/new/removed vs a baseline JSON report
  cache.py         ResponseCache (atomic JSON files) + CachingProvider wrapper; keyed on
                   Provider.cache_identity() + messages + sampling params
  registry.py      generic name -> class Registry with entry-point discovery
  errors.py        EvalkitError > SuiteError, RegistryError, ProviderError > NotSupportedError
  providers/       base.py (Provider ABC, registry), fake.py, openai_compat.py, anthropic.py,
                   _http.py (shared httpx client + retry/backoff)
  scorers/         base.py (Scorer ABC, ScoreContext, registry), basic.py, embedding.py, judge.py
  report/          json_report.py, html_report.py, write_report() picks format by extension
  cli.py           argparse (run, compare, validate, list); exit codes 0 = gate passed,
                   1 = below threshold or regressed vs --baseline, 2 = usage/config error
src/lectureiq/
  transcript.py    parse SRT/WebVTT/plain text -> Lecture; chunk_lecture -> timestamped Chunks;
                   lecture_to_dict / lecture_from_dict for persistence
  audio.py         WhisperTranscriber (OpenAI-compatible /audio/transcriptions, verbose_json)
  engines.py       Engine protocol (study_pack, answer) + PackOptions caps
  offline.py       OfflineEngine: extractive heuristics, deterministic, no model
  llm.py           LLMEngine: per-chunk JSON notes (map) + outline synthesis (reduce), with
                   pydantic validation, one repair retry, and skip-with-warning on failure
  retrieval.py     BM25Index over chunks (used by ask)
  pipeline.py      analyze(), ask(), make_engine() ("offline" or any evalkit provider)
  models.py        StudyPack, Section, KeyPoint, Term, Flashcard, QuizQuestion, Answer, Citation
  export.py        to_markdown(), flashcards_csv(), to_vtt()
  store.py         Store protocol; MemoryStore (bounded LRU) and SQLiteStore (one row per lecture)
  evalkit_provider.py  LectureIQProvider: answers evalkit cases from a lecture, adds "Sources:"
  web/app.py       Starlette app factory create_app(engine); JSON API under /api
  web/static/      index.html, app.js, style.css (vanilla JS, no build step)
  samples/         intro_to_ml.vtt, the bundled demo lecture (`@sample` in the CLI)
  cli.py           lectureiq analyze | ask | transcribe | serve
tests/             one module per area (test_lectureiq_* for LectureIQ); conftest.py has the
                   network guard, make_ctx, and sample_vtt fixtures
examples/suite.yaml  runs green offline with the fake provider; used by tests and CI
examples/lectureiq_qa.yaml  grades LectureIQ (offline engine) on the sample lecture; CI gate
demo/              the hosted single-page demo: lectureiq-core.js (JS port of transcript parsing,
                   chunking, BM25, OfflineEngine, exports), page.html, build.py -> demo/dist/
```

**The demo is a port.** If you change `transcript.py`, `text.py`, `retrieval.py`, `offline.py`
or `export.py`, make the same change in `demo/lectureiq-core.js`. `tests/test_demo_port.py`
(needs Node) fails if the two engines' study packs differ or the port fails the QA suite.
Rebuild with `python demo/build.py`. The live artifact is republished from that output.

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
- **LectureIQ frontend:** insert lecture-derived text with `textContent` (the `h()` helper in
  `app.js`), never `innerHTML`; transcripts and model output are untrusted. No build step and
  no CDN dependencies: keep it plain HTML, CSS, and JS.
- **LectureIQ engines** implement both `study_pack()` and `answer()`. Every generated item
  carries a `chunk_id` (and a precise `start` when the engine knows the source sentence) so
  the UI can link it to the transcript. Model output is untrusted
  JSON: validate it with pydantic, drop bad items, and record a warning instead of failing.

## Adding things

**A scorer:** subclass `Scorer` in `src/evalkit/scorers/`, declare options as pydantic
fields, set `name`, implement `score(ctx) -> ScoreResult` (use `self.result(...)`, which
clamps the score to [0, 1]), decorate with `@register_scorer("name")`, import it in
`scorers/__init__.py`, add tests in `tests/test_scorers.py`, and add a row to the scorer
table in README.md.

**A provider:** subclass `Provider` (or `HTTPProvider` for JSON-over-HTTPS APIs), define a
`ProviderConfig` subclass as `Config`, implement `complete()` (and `embed()` if supported),
override `cache_identity()` if anything beyond name and model changes replies (never include
secrets), register with `@register_provider("name")`, import it in `providers/__init__.py`,
and test it with `httpx.MockTransport` in `tests/test_providers.py`. Update the README
provider table.

## Testing rules

- **No network, ever.** `tests/conftest.py` patches `socket.connect` to raise. Use
  `FakeProvider` for behaviour and `httpx.MockTransport` (passed as `transport=`) to test
  HTTP providers' request and response handling. No real API keys in tests.
- Use the `make_ctx` fixture to build a `ScoreContext` for scorer tests.
- `filterwarnings = ["error"]`: warnings fail the suite; fix the cause rather than
  filtering it.
- Patch `evalkit.providers._http.time.sleep` in retry tests so they stay fast.
- LectureIQ: drive `LLMEngine` with `FakeProvider` (`regex: true` responses keyed on
  `TASK: chunk_notes` / `TASK: synthesis` / `TASK: answer` or `excerpt id="N"`), and the web
  API with `starlette.testclient.TestClient`. The pytest config ignores two deprecation
  warnings raised inside Starlette's test client (anyio alias, httpx -> httpx2); don't widen
  those filters.
- Audio: never call a real transcription API in tests. Use a stub with a `transcribe()` method
  for the web API and CLI (monkeypatch `lectureiq.cli.make_transcriber`), and
  `httpx.MockTransport` for `WhisperTranscriber` itself.
- If you change the offline engine or the sample lecture, run `examples/lectureiq_qa.yaml`.
  `tests/test_lectureiq_features.py` runs it too, and checks that it still catches a known
  regression.
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

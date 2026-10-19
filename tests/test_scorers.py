from __future__ import annotations

import builtins
from typing import Any

import pytest

from evalkit import FakeProvider, ScoreContext, Scorer, SuiteError, build_scorer, register_scorer
from evalkit.results import ScoreResult
from evalkit.scorers import scorer_registry
from evalkit.scorers.embedding import cosine
from evalkit.scorers.judge import parse_verdict
from tests.conftest import MakeContext


def score(spec: dict[str, Any], ctx: ScoreContext) -> ScoreResult:
    return build_scorer(spec).score(ctx)


class TestExactMatch:
    def test_matches_expected_with_whitespace_normalisation(self, make_ctx: MakeContext) -> None:
        result = score({"type": "exact_match"}, make_ctx("  Paris \n", "Paris"))
        assert result.passed and result.score == 1.0

    def test_case_sensitivity(self, make_ctx: MakeContext) -> None:
        assert not score({"type": "exact_match"}, make_ctx("paris", "Paris")).passed
        spec = {"type": "exact_match", "case_sensitive": False}
        assert score(spec, make_ctx("paris", "Paris")).passed

    def test_explicit_value_overrides_expected(self, make_ctx: MakeContext) -> None:
        result = score({"type": "exact_match", "value": "Lyon"}, make_ctx("Paris", "Paris"))
        assert not result.passed and "Lyon" in result.reason

    def test_needs_a_reference(self, make_ctx: MakeContext) -> None:
        with pytest.raises(SuiteError, match="expected"):
            score({"type": "exact_match"}, make_ctx("Paris"))


class TestContains:
    def test_defaults_to_expected_case_insensitive(self, make_ctx: MakeContext) -> None:
        assert score({"type": "contains"}, make_ctx("It is TOKYO.", "tokyo")).passed

    def test_all_mode_gives_partial_credit(self, make_ctx: MakeContext) -> None:
        result = score({"type": "contains", "value": ["a", "b"]}, make_ctx("a only"))
        assert not result.passed
        assert result.score == 0.5
        assert "'b'" in result.reason

    def test_any_mode(self, make_ctx: MakeContext) -> None:
        spec = {"type": "contains", "value": ["x", "only"], "mode": "any"}
        assert score(spec, make_ctx("a only")).passed

    def test_negate(self, make_ctx: MakeContext) -> None:
        spec = {"type": "contains", "value": ["Kyoto", "Osaka"], "mode": "any", "negate": True}
        assert score(spec, make_ctx("Tokyo")).passed
        failed = score(spec, make_ctx("Kyoto, maybe"))
        assert not failed.passed and "forbidden" in failed.reason


class TestRegex:
    def test_search(self, make_ctx: MakeContext) -> None:
        assert score({"type": "regex", "pattern": r"\d+"}, make_ctx("answer: 42")).passed

    def test_fullmatch_strips_surrounding_whitespace(self, make_ctx: MakeContext) -> None:
        spec = {"type": "regex", "pattern": r"\d{4}-\d{2}-\d{2}", "fullmatch": True}
        assert score(spec, make_ctx(" 2024-03-01\n")).passed
        assert not score(spec, make_ctx("Date: 2024-03-01")).passed

    def test_flags_and_negate(self, make_ctx: MakeContext) -> None:
        assert score({"type": "regex", "pattern": "yes", "ignore_case": True}, make_ctx("YES"))
        spec = {"type": "regex", "pattern": "sorry", "ignore_case": True, "negate": True}
        assert not score(spec, make_ctx("Sorry, I can't")).passed


class TestEmbeddingSimilarity:
    def test_uses_provider_embeddings(self, make_ctx: MakeContext) -> None:
        spec = {"type": "embedding_similarity", "threshold": 0.5}
        close = score(spec, make_ctx("glucose and oxygen", "it makes glucose and oxygen"))
        far = score(spec, make_ctx("bananas", "it makes glucose and oxygen"))
        assert close.passed and close.score > far.score
        assert not far.passed

    def test_named_provider_and_reference(self, make_ctx: MakeContext) -> None:
        other = FakeProvider.from_config({})
        spec = {"type": "embedding_similarity", "reference": "same words", "provider": "emb"}
        ctx = make_ctx("same words", named={"emb": other})
        assert score(spec, ctx).score == pytest.approx(1.0)

    def test_local_backend_without_extra_explains_install(
        self, make_ctx: MakeContext, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        real_import = builtins.__import__

        def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "sentence_transformers":
                raise ImportError(name)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        spec = {"type": "embedding_similarity", "backend": "sentence-transformers"}
        with pytest.raises(SuiteError, match=r"evalkit\[embeddings\]"):
            score(spec, make_ctx("a", "b"))

    def test_cosine(self) -> None:
        assert cosine([1, 0], [1, 0]) == 1.0
        assert cosine([1, 0], [0, 1]) == 0.0
        assert cosine([0, 0], [1, 1]) == 0.0
        with pytest.raises(ValueError, match="mismatch"):
            cosine([1], [1, 2])


class TestLLMJudge:
    def test_grades_with_named_judge(self, make_ctx: MakeContext) -> None:
        judge = FakeProvider.from_config({"default": '{"score": 4, "reasoning": "good"}'})
        spec = {"type": "llm_judge", "criteria": "Is correct", "provider": "judge"}
        result = score(spec, make_ctx("Paris", "Paris", named={"judge": judge}))
        assert result.passed
        assert result.score == pytest.approx(0.75)
        assert result.reason == "4/5: good"
        prompt = judge.calls[0][0].content
        assert "<response>\nParis\n</response>" in prompt
        assert "<expected>" in prompt and "Is correct" in prompt

    def test_defaults_to_provider_under_test(self, make_ctx: MakeContext) -> None:
        primary = FakeProvider.from_config({"default": "Score: 2"})
        result = score({"type": "llm_judge", "criteria": "x"}, make_ctx("out", provider=primary))
        assert not result.passed and result.score == pytest.approx(0.25)
        assert len(primary.calls) == 1

    def test_unparseable_reply_fails(self, make_ctx: MakeContext) -> None:
        primary = FakeProvider.from_config({"default": "I refuse"})
        result = score({"type": "llm_judge", "criteria": "x"}, make_ctx("out", provider=primary))
        assert not result.passed and "could not parse" in result.reason

    def test_custom_scale_threshold_and_clamping(self, make_ctx: MakeContext) -> None:
        primary = FakeProvider.from_config({"default": '{"score": 99}'})
        spec = {
            "type": "llm_judge",
            "criteria": "x",
            "min_score": 0,
            "max_score": 10,
            "threshold": 0.9,
        }
        result = score(spec, make_ctx("out", provider=primary))
        assert result.passed and result.score == 1.0

    def test_invalid_scale_rejected(self) -> None:
        with pytest.raises(SuiteError, match="max_score"):
            build_scorer({"type": "llm_judge", "criteria": "x", "min_score": 5, "max_score": 5})

    @pytest.mark.parametrize(
        ("reply", "expected"),
        [
            ('{"score": 5, "reasoning": "great"}', (5.0, "great")),
            ('Sure! ```json\n{"score": 3.5}\n```', (3.5, "")),
            ("score = 2", (2.0, "score = 2")),
            ('{"score": "high"} score: 1', (1.0, '{"score": "high"} score: 1')),
            ("no idea", None),
        ],
    )
    def test_parse_verdict(self, reply: str, expected: tuple[float, str] | None) -> None:
        assert parse_verdict(reply) == expected


def test_custom_scorer_plugin(make_ctx: MakeContext) -> None:
    @register_scorer("test_length_under")
    class LengthUnder(Scorer):
        name = "test_length_under"
        max_chars: int

        def score(self, ctx: ScoreContext) -> ScoreResult:
            ok = len(ctx.output) < self.max_chars
            return self.result(1.0 if ok else 0.0, ok)

    try:
        assert score({"type": "test_length_under", "max_chars": 5}, make_ctx("abc")).passed
        assert "test_length_under" in scorer_registry.names()
    finally:
        scorer_registry._items.pop("test_length_under")

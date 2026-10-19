"""evalkit: a small, pluggable library and CLI for evaluating LLM outputs."""

from evalkit.compare import Comparison, compare_runs, load_run
from evalkit.errors import (
    EvalkitError,
    NotSupportedError,
    ProviderError,
    RegistryError,
    SuiteError,
)
from evalkit.loader import load_suite, parse_suite
from evalkit.models import Message, ScorerSpec, Suite, TestCase
from evalkit.providers import (
    AnthropicProvider,
    Completion,
    FakeProvider,
    OpenAICompatibleProvider,
    Provider,
    ProviderConfig,
    create_provider,
    register_provider,
)
from evalkit.results import CaseResult, RunResult, ScoreResult
from evalkit.runner import run_suite
from evalkit.scorers import ScoreContext, Scorer, build_scorer, register_scorer

__version__ = "0.1.0"

__all__ = [
    "AnthropicProvider",
    "CaseResult",
    "Comparison",
    "Completion",
    "EvalkitError",
    "FakeProvider",
    "Message",
    "NotSupportedError",
    "OpenAICompatibleProvider",
    "Provider",
    "ProviderConfig",
    "ProviderError",
    "RegistryError",
    "RunResult",
    "ScoreContext",
    "ScoreResult",
    "Scorer",
    "ScorerSpec",
    "Suite",
    "SuiteError",
    "TestCase",
    "__version__",
    "build_scorer",
    "compare_runs",
    "create_provider",
    "load_run",
    "load_suite",
    "parse_suite",
    "register_provider",
    "register_scorer",
    "run_suite",
]

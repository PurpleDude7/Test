"""Suite and test-case models, validated with pydantic when loaded from YAML."""

from __future__ import annotations

from typing import Any, ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Role = Literal["system", "user", "assistant"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Message(_Strict):
    """A single chat message."""

    role: Role
    content: str


class ScorerSpec(_Strict):
    """A scorer reference as written in YAML: ``{type: regex, pattern: "..."}``.

    Everything except ``type`` is collected into ``params`` and validated by the scorer
    itself when the suite is loaded.
    """

    type: str
    params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _collect_params(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"type": data}
        if isinstance(data, dict) and "params" not in data:
            rest = {k: v for k, v in data.items() if k != "type"}
            return {"type": data.get("type"), "params": rest}
        return data


class Defaults(_Strict):
    """Values applied to every case that does not set them itself."""

    system: str | None = None
    scorers: list[ScorerSpec] = Field(default_factory=list)
    temperature: float | None = Field(default=None, ge=0)
    max_tokens: int | None = Field(default=None, gt=0)


class TestCase(_Strict):
    """One prompt plus the scorers that decide whether the model's answer passes."""

    __test__: ClassVar[bool] = False  # keep pytest from trying to collect this class

    id: str = Field(min_length=1)
    input: str | list[Message]
    expected: str | None = None
    system: str | None = None
    scorers: list[ScorerSpec] | None = None
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def messages(self, default_system: str | None = None) -> list[Message]:
        """The chat transcript to send, with the system prompt (if any) first."""
        system = self.system if self.system is not None else default_system
        turns = (
            [Message(role="user", content=self.input)]
            if isinstance(self.input, str)
            else list(self.input)
        )
        if system and not any(m.role == "system" for m in turns):
            turns.insert(0, Message(role="system", content=system))
        return turns

    def prompt_text(self) -> str:
        """The last user turn: a human-readable stand-in for the prompt."""
        if isinstance(self.input, str):
            return self.input
        users = [m.content for m in self.input if m.role == "user"]
        return users[-1] if users else ""


class Suite(_Strict):
    """A named collection of test cases plus the pass-rate gate for CI."""

    name: str
    description: str | None = None
    threshold: float = Field(default=1.0, ge=0, le=1)
    provider: str | None = None
    providers: dict[str, dict[str, Any]] = Field(default_factory=dict)
    defaults: Defaults = Field(default_factory=Defaults)
    cases: list[TestCase] = Field(min_length=1)

    @model_validator(mode="after")
    def _unique_ids(self) -> Suite:
        seen: set[str] = set()
        for case in self.cases:
            if case.id in seen:
                raise ValueError(f"duplicate case id {case.id!r}")
            seen.add(case.id)
        return self

    def resolve_provider(self, name: str) -> tuple[str, dict[str, Any]]:
        """Map a provider name to ``(registry name, config)``.

        Entries under ``providers`` are either config for a registered provider
        (``openai: {model: ...}``) or a named alias with an explicit ``type``
        (``judge: {type: anthropic, model: ...}``).
        """
        config = dict(self.providers.get(name, {}))
        kind = config.pop("type", name)
        if not isinstance(kind, str):
            raise ValueError(f"providers.{name}.type must be a string")
        return kind, config

    def scorers_for(self, case: TestCase) -> list[ScorerSpec]:
        return case.scorers if case.scorers is not None else self.defaults.scorers

    def filter_tags(self, tags: list[str]) -> Suite:
        """A copy keeping only cases that carry at least one of ``tags``."""
        if not tags:
            return self
        wanted = set(tags)
        kept = [c for c in self.cases if wanted.intersection(c.tags)]
        if not kept:
            raise ValueError(f"no cases match tags {sorted(wanted)}")
        return self.model_copy(update={"cases": kept})

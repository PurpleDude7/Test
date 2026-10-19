"""Provider tests. HTTP providers run against httpx.MockTransport: no sockets are opened."""

from __future__ import annotations

import json
from collections.abc import Callable

import httpx
import pytest

from evalkit import (
    AnthropicProvider,
    FakeProvider,
    Message,
    NotSupportedError,
    OpenAICompatibleProvider,
    ProviderError,
    RegistryError,
    create_provider,
)
from evalkit.providers.anthropic import AnthropicConfig
from evalkit.providers.openai_compat import OpenAICompatibleConfig

USER = [Message(role="system", content="sys"), Message(role="user", content="Hello there")]

Handler = Callable[[httpx.Request], httpx.Response]


class TestFakeProvider:
    def test_echoes_by_default_and_records_calls(self) -> None:
        fake = FakeProvider.from_config({})
        assert fake.complete(USER).text == "Hello there"
        assert fake.calls == [USER]
        assert fake.model == "fake-1"

    def test_first_matching_response_wins(self) -> None:
        fake = FakeProvider.with_responses({"Hello": "first", "there": "second"})
        assert fake.complete(USER).text == "first"

    def test_regex_default_and_failures(self) -> None:
        fake = FakeProvider.from_config(
            {"responses": {r"^H\w+": "rx"}, "regex": True, "default": "dflt", "fail_on": ["boom"]}
        )
        assert fake.complete(USER).text == "rx"
        assert fake.complete([Message(role="user", content="nope")]).text == "dflt"
        with pytest.raises(ProviderError, match="boom"):
            fake.complete([Message(role="user", content="go boom")])

    def test_embeddings_are_deterministic_and_normalised(self) -> None:
        fake = FakeProvider.from_config({"embedding_dim": 16})
        a, b = fake.embed(["Hello world", "hello, WORLD"])
        assert a == b
        assert len(a) == 16
        assert sum(x * x for x in a) == pytest.approx(1.0)
        assert fake.embed([""]) == [[0.0] * 16]

    def test_rejects_unknown_config(self) -> None:
        with pytest.raises(ProviderError, match="invalid config"):
            FakeProvider.from_config({"respones": {}})


def test_registry_lookup() -> None:
    assert isinstance(create_provider("fake"), FakeProvider)
    assert isinstance(create_provider("openai-compatible"), OpenAICompatibleProvider)
    with pytest.raises(RegistryError, match="unknown provider 'nope'"):
        create_provider("nope")


def recorder(
    response: httpx.Response | list[httpx.Response],
) -> tuple[list[httpx.Request], Handler]:
    requests: list[httpx.Request] = []
    queue = response if isinstance(response, list) else None

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if queue is not None:
            return queue.pop(0)
        assert isinstance(response, httpx.Response)
        return response

    return requests, handler


class TestOpenAICompatible:
    def make(self, handler: Handler, **config: object) -> OpenAICompatibleProvider:
        cfg = OpenAICompatibleConfig.model_validate({"api_key": "sk-test", **config})
        return OpenAICompatibleProvider(cfg, transport=httpx.MockTransport(handler))

    def test_chat_request_and_response(self) -> None:
        body = {
            "model": "gpt-x",
            "choices": [{"message": {"role": "assistant", "content": "Hi!"}}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 2},
        }
        requests, handler = recorder(httpx.Response(200, json=body))
        provider = self.make(handler, base_url="http://local/v1/", model="m")
        result = provider.complete(USER, temperature=0.0, max_tokens=5)

        assert result.text == "Hi!" and result.model == "gpt-x"
        assert result.usage is not None and result.usage.output_tokens == 2
        req = requests[0]
        assert str(req.url) == "http://local/v1/chat/completions"
        assert req.headers["authorization"] == "Bearer sk-test"
        sent = json.loads(req.content)
        assert sent == {
            "model": "m",
            "messages": [{"role": "system", "content": "sys"}, USER[1].model_dump()],
            "temperature": 0.0,
            "max_tokens": 5,
        }

    def test_embeddings_sorted_by_index(self) -> None:
        body = {"data": [{"index": 1, "embedding": [0, 1]}, {"index": 0, "embedding": [1, 0]}]}
        requests, handler = recorder(httpx.Response(200, json=body))
        provider = self.make(handler)
        assert provider.embed(["a", "b"]) == [[1.0, 0.0], [0.0, 1.0]]
        assert json.loads(requests[0].content)["model"] == "text-embedding-3-small"

    def test_retries_transient_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("evalkit.providers._http.time.sleep", lambda s: None)
        ok = {"choices": [{"message": {"content": "ok"}}]}
        requests, handler = recorder(
            [httpx.Response(429, text="slow down"), httpx.Response(200, json=ok)]
        )
        assert self.make(handler).complete(USER).text == "ok"
        assert len(requests) == 2

    def test_gives_up_after_max_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("evalkit.providers._http.time.sleep", lambda s: None)
        requests, handler = recorder(httpx.Response(503, text="down"))
        with pytest.raises(ProviderError, match="HTTP 503"):
            self.make(handler, max_retries=1).complete(USER)
        assert len(requests) == 2

    def test_client_errors_are_not_retried(self) -> None:
        requests, handler = recorder(httpx.Response(401, text="bad key"))
        with pytest.raises(ProviderError, match="HTTP 401: bad key"):
            self.make(handler).complete(USER)
        assert len(requests) == 1

    def test_transport_errors_become_provider_errors(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("evalkit.providers._http.time.sleep", lambda s: None)

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("refused", request=request)

        with pytest.raises(ProviderError, match="request failed"):
            self.make(handler, max_retries=0).complete(USER)

    def test_bad_response_shape(self) -> None:
        _, handler = recorder(httpx.Response(200, json={"choices": []}))
        with pytest.raises(ProviderError, match="unexpected response shape"):
            self.make(handler).complete(USER)

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        provider = OpenAICompatibleProvider.from_config({})
        with pytest.raises(ProviderError, match=r"\$OPENAI_API_KEY"):
            provider.complete(USER)

    def test_api_key_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("MY_KEY", "from-env")
        cfg = OpenAICompatibleConfig(api_key_env="MY_KEY")
        assert OpenAICompatibleProvider(cfg).api_key == "from-env"


class TestAnthropic:
    def make(self, handler: Handler, **config: object) -> AnthropicProvider:
        cfg = AnthropicConfig.model_validate({"api_key": "ak-test", **config})
        return AnthropicProvider(cfg, transport=httpx.MockTransport(handler))

    def test_messages_request_and_response(self) -> None:
        body = {
            "model": "claude-x",
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"},
            ],
            "usage": {"input_tokens": 9, "output_tokens": 3},
        }
        requests, handler = recorder(httpx.Response(200, json=body))
        result = self.make(handler).complete(USER)

        assert result.text == "Hello world"
        assert result.usage is not None and result.usage.input_tokens == 9
        req = requests[0]
        assert str(req.url) == "https://api.anthropic.com/v1/messages"
        assert req.headers["x-api-key"] == "ak-test"
        assert req.headers["anthropic-version"] == "2023-06-01"
        sent = json.loads(req.content)
        assert sent == {
            "model": "claude-opus-5-5",
            "max_tokens": 1024,
            "system": "sys",
            "messages": [{"role": "user", "content": "Hello there"}],
        }

    def test_passes_sampling_options(self) -> None:
        requests, handler = recorder(httpx.Response(200, json={"content": []}))
        self.make(handler).complete(USER[1:], temperature=0.2, max_tokens=50)
        sent = json.loads(requests[0].content)
        assert sent["max_tokens"] == 50 and sent["temperature"] == 0.2
        assert "system" not in sent

    def test_bad_response_shape(self) -> None:
        _, handler = recorder(httpx.Response(200, json={"type": "error"}))
        with pytest.raises(ProviderError, match="unexpected response shape"):
            self.make(handler).complete(USER)

    def test_embeddings_not_supported(self) -> None:
        with pytest.raises(NotSupportedError):
            AnthropicProvider.from_config({}).embed(["x"])

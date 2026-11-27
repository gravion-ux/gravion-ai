import io
import json
import urllib.error

import pytest

from gravion_ai.config import ProviderConfig
from gravion_ai.errors import ProviderError
from gravion_ai.providers import EndpointProvider, GravionCoreProvider, build_provider

MESSAGES = [
    {"role": "system", "content": "You are Gravion."},
    {"role": "user", "content": "hello"},
]


def test_gravion_core_is_deterministic_and_offline():
    provider = GravionCoreProvider()
    out = provider.complete(MESSAGES)
    assert "hello" in out
    assert provider.complete(MESSAGES) == out  # deterministic


def test_build_provider_selects_engine():
    assert isinstance(
        build_provider(ProviderConfig(type="gravion-core")), GravionCoreProvider
    )
    assert isinstance(
        build_provider(
            ProviderConfig(type="endpoint", base_url="http://x.test/v1")
        ),
        EndpointProvider,
    )


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def test_endpoint_provider_wire_format():
    captured = {}

    def fake_opener(request, timeout=None):
        captured["url"] = request.full_url
        captured["auth"] = request.get_header("Authorization")
        captured["payload"] = json.loads(request.data.decode("utf-8"))
        body = {"choices": [{"message": {"content": "endpoint says hi"}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    config = ProviderConfig(
        type="endpoint",
        base_url="http://localhost:9999/v1/",
        model="gravion-chat",
        api_key="test-key",
    )
    provider = EndpointProvider(config, opener=fake_opener)
    out = provider.complete(MESSAGES)
    assert out == "endpoint says hi"
    assert captured["url"] == "http://localhost:9999/v1/chat/completions"
    assert captured["auth"] == "Bearer test-key"
    assert captured["payload"]["model"] == "gravion-chat"
    assert captured["payload"]["messages"] == MESSAGES


def test_endpoint_provider_model_override():
    def fake_opener(request, timeout=None):
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["model"] == "gravion-code"
        body = {"choices": [{"message": {"content": "ok"}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    config = ProviderConfig(type="endpoint", base_url="http://x.test", model="gravion-chat")
    provider = EndpointProvider(config, opener=fake_opener)
    assert provider.complete(MESSAGES, model="gravion-code") == "ok"


def test_endpoint_provider_rejects_bad_messages_and_non_text_content():
    config = ProviderConfig(type="endpoint", base_url="http://x.test", model="gravion-chat")
    provider = EndpointProvider(config, opener=lambda *a, **k: None)
    with pytest.raises(ProviderError):
        provider.complete([{"role": "tool", "content": "hello"}])

    def fake_opener(request, timeout=None):
        body = {"choices": [{"message": {"content": {"not": "text"}}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    provider = EndpointProvider(config, opener=fake_opener)
    with pytest.raises(ProviderError):
        provider.complete(MESSAGES)


def test_endpoint_provider_http_error_does_not_leak_body():
    def fake_opener(request, timeout=None):
        raise urllib.error.HTTPError(
            request.full_url,
            500,
            "server error",
            hdrs=None,
            fp=io.BytesIO(b"secret upstream body"),
        )

    config = ProviderConfig(
        type="endpoint", base_url="http://x.test", model="gravion-chat", max_retries=0
    )
    provider = EndpointProvider(config, opener=fake_opener)
    with pytest.raises(ProviderError) as excinfo:
        provider.complete(MESSAGES)
    assert "HTTP 500" in str(excinfo.value)
    assert "secret upstream body" not in str(excinfo.value)


class _ListLog:
    def __init__(self):
        self.records = []

    def _record(self, level, message, *args):
        self.records.append((level, message % args if args else message))

    def info(self, message, *args):
        self._record("info", message, *args)

    def warning(self, message, *args):
        self._record("warning", message, *args)


def test_endpoint_provider_retries_transient_http_and_logs_safely():
    attempts = {"count": 0}
    delays = []
    log = _ListLog()

    def fake_opener(request, timeout=None):
        attempts["count"] += 1
        assert timeout == 7
        assert request.get_header("X-gravion-request-id") == "req-test"
        if attempts["count"] == 1:
            raise urllib.error.HTTPError(
                request.full_url,
                503,
                "busy",
                hdrs=None,
                fp=io.BytesIO(b"secret upstream body"),
            )
        payload = json.loads(request.data.decode("utf-8"))
        assert payload["temperature"] == 0.2
        assert payload["max_tokens"] == 123
        body = {"choices": [{"message": {"content": "recovered"}}]}
        return _FakeResponse(json.dumps(body).encode("utf-8"))

    config = ProviderConfig(
        type="endpoint",
        base_url="http://x.test",
        model="gravion-chat",
        api_key="test-key",
        timeout_seconds=7,
        max_retries=2,
        retry_backoff_seconds=0.25,
        temperature=0.2,
        max_tokens=123,
    )
    provider = EndpointProvider(
        config,
        opener=fake_opener,
        sleeper=delays.append,
        request_id_factory=lambda: "req-test",
        log=log,
    )
    assert provider.complete(MESSAGES) == "recovered"
    assert attempts["count"] == 2
    assert delays == [0.25]
    rendered = "\n".join(message for _, message in log.records)
    assert "provider retry" in rendered
    assert "provider completed" in rendered
    assert "test-key" not in rendered
    assert "hello" not in rendered
    assert "secret upstream body" not in rendered


def test_endpoint_provider_does_not_retry_client_error():
    attempts = {"count": 0}

    def fake_opener(request, timeout=None):
        attempts["count"] += 1
        raise urllib.error.HTTPError(
            request.full_url,
            400,
            "bad request",
            hdrs=None,
            fp=io.BytesIO(b"bad"),
        )

    config = ProviderConfig(
        type="endpoint", base_url="http://x.test", model="gravion-chat", max_retries=3
    )
    provider = EndpointProvider(config, opener=fake_opener, sleeper=lambda _: None)
    with pytest.raises(ProviderError, match="HTTP 400"):
        provider.complete(MESSAGES)
    assert attempts["count"] == 1


def test_endpoint_provider_retries_timeout_then_raises_without_body():
    attempts = {"count": 0}
    delays = []

    def fake_opener(request, timeout=None):
        attempts["count"] += 1
        raise TimeoutError("slow")

    config = ProviderConfig(
        type="endpoint",
        base_url="http://x.test",
        model="gravion-chat",
        max_retries=1,
        retry_backoff_seconds=0.1,
    )
    provider = EndpointProvider(
        config, opener=fake_opener, sleeper=delays.append, log=_ListLog()
    )
    with pytest.raises(ProviderError, match="TimeoutError"):
        provider.complete(MESSAGES)
    assert attempts["count"] == 2
    assert delays == [0.1]


def test_providers_reject_empty_user_content():
    # Regression: an empty/whitespace user prompt used to echo back
    # "[gravion-core] You said: " instead of failing fast.
    core = GravionCoreProvider()
    for content in ("", "   ", "\n\n"):
        with pytest.raises(ProviderError, match="non-empty content"):
            core.complete(
                [
                    {"role": "system", "content": "You are Gravion."},
                    {"role": "user", "content": content},
                ]
            )


def test_providers_reject_messages_without_user_turn():
    core = GravionCoreProvider()
    with pytest.raises(ProviderError, match="non-empty content"):
        core.complete([{"role": "system", "content": "You are Gravion."}])


def test_endpoint_provider_rejects_empty_user_content_without_calling():
    calls = {"count": 0}

    def fake_opener(request, timeout=None):
        calls["count"] += 1
        raise AssertionError("endpoint must not be called for an empty prompt")

    config = ProviderConfig(type="endpoint", base_url="http://x.test", model="gravion-chat")
    provider = EndpointProvider(config, opener=fake_opener)
    with pytest.raises(ProviderError, match="non-empty content"):
        provider.complete([{"role": "user", "content": "  "}])
    assert calls["count"] == 0


def test_provider_uses_last_user_message_for_empty_check():
    # A real request followed by an empty user turn is still an
    # empty request: the last user message is what gets answered.
    core = GravionCoreProvider()
    with pytest.raises(ProviderError, match="non-empty content"):
        core.complete(
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi"},
                {"role": "user", "content": ""},
            ]
        )
    out = core.complete(
        [
            {"role": "user", "content": ""},
            {"role": "user", "content": "hello"},
        ]
    )
    assert "hello" in out


def test_gravion_core_logs_weight_load_failure_and_falls_back(monkeypatch, caplog):
    import gravion_ai.providers as providers

    def fail_load(*_args, **_kwargs):
        raise ValueError("corrupt weights")

    monkeypatch.setattr(providers, "load_weights", fail_load)
    with caplog.at_level("WARNING", logger="gravion_ai.providers"):
        provider = GravionCoreProvider(weights_path="weights.json")
    assert provider.parameter_count == 0
    assert provider.complete(MESSAGES).endswith("You said: hello")
    assert "weights load failed" in caplog.text
    assert "corrupt weights" not in caplog.text


def test_gravion_core_logs_specialist_weight_load_failure(monkeypatch, caplog):
    import gravion_ai.providers as providers

    def fail_load(_model_id=None):
        raise ValueError("corrupt specialist weights")

    monkeypatch.setattr(providers, "load_bundled_model", fail_load)
    provider = GravionCoreProvider(model=object())
    with caplog.at_level("WARNING", logger="gravion_ai.providers"):
        assert provider.parameter_count_for("gravion-chat") == 0
    assert "specialist weights load failed" in caplog.text
    assert "corrupt specialist weights" not in caplog.text

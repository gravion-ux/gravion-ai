import io
import json

from gravion_ai.config import ProviderConfig
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

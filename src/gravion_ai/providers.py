"""How a deployment reaches a model.

Gravion ships with its own built-in engine — **Gravion Core** — which
runs fully offline with no network and no API key. Deployments that
outgrow it point Gravion at any inference endpoint they operate or
rent: a base URL plus an API key in the deployment config, speaking a
simple chat-completions JSON protocol. Which *model* an endpoint is
asked for comes from the Gravion model registry (catalog.py) — the
registry is the product surface; this layer is just plumbing.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Protocol, Sequence

from .config import ProviderConfig
from .errors import ProviderError

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}


class ChatProvider(Protocol):
    def complete(self, messages: Sequence[Message], model: str | None = None) -> str:
        ...


class GravionCoreProvider:
    """The built-in Gravion Core engine.

    Piece 1's Core is a deterministic offline responder: it proves the
    full module path (config → context → registry → provider → gate)
    with zero infrastructure, and it is the engine development mode
    runs on. Trained Core weights land in later pieces; the interface
    they plug into is this one.
    """

    name = "gravion-core"

    def complete(self, messages: Sequence[Message], model: str | None = None) -> str:
        user = next(
            (m["content"] for m in reversed(messages) if m.get("role") == "user"),
            "",
        )
        return f"[gravion-core] You said: {user}"


class EndpointProvider:
    """Chat completions against a configurable inference endpoint.

    POSTs {"model", "messages"} to {base_url}/chat/completions with a
    bearer API key when one is configured, and reads the reply from
    choices[0].message.content — a deliberately small, generic wire
    contract that any inference server can implement.
    """

    name = "endpoint"

    def __init__(self, config: ProviderConfig, opener=None):
        self._config = config
        # `opener` is injectable so tests can exercise the wire format
        # without a network.
        self._opener = opener or urllib.request.urlopen

    def complete(self, messages: Sequence[Message], model: str | None = None) -> str:
        cfg = self._config
        if not cfg.base_url:
            raise ProviderError(
                "provider.type is 'endpoint' but no base_url is configured"
            )
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": model or cfg.model,
            "messages": list(messages),
        }
        headers = {"Content-Type": "application/json"}
        key = cfg.resolved_api_key()
        if key:
            headers["Authorization"] = f"Bearer {key}"
        request = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with self._opener(request, timeout=cfg.timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
            raise ProviderError(f"endpoint call to {url} failed: {exc}") from exc
        try:
            return body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderError(f"unexpected endpoint response shape: {exc}") from exc


def build_provider(config: ProviderConfig) -> ChatProvider:
    if config.type == "gravion-core":
        return GravionCoreProvider()
    if config.type == "endpoint":
        return EndpointProvider(config)
    raise ProviderError(f"unknown provider type {config.type!r}")

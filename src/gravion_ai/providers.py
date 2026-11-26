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
import logging
import time
import urllib.error
import urllib.request
import uuid
from typing import Protocol, Sequence

from .config import ProviderConfig
from .core_model import (
    CORE_MODEL_ID,
    SPECIALIST_MODEL_IDS,
    GravionCoreLocalModel,
    deployment_weights_path,
    load_bundled_model,
    load_weights,
)
from .errors import ProviderError

Message = dict  # {"role": "system"|"user"|"assistant", "content": str}

MAX_RESPONSE_BYTES = 4 * 1024 * 1024
_ALLOWED_ROLES = frozenset({"system", "user", "assistant"})
_RETRYABLE_HTTP_STATUSES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})
logger = logging.getLogger(__name__)


def _validate_messages(messages: Sequence[Message]) -> list[Message]:
    if isinstance(messages, (str, bytes)) or not messages:
        raise ProviderError("messages must be a non-empty sequence")
    validated: list[Message] = []
    for index, message in enumerate(messages):
        if not isinstance(message, dict):
            raise ProviderError(f"message {index} must be an object")
        role = message.get("role")
        content = message.get("content")
        if role not in _ALLOWED_ROLES:
            raise ProviderError(f"message {index} has unsupported role {role!r}")
        if not isinstance(content, str):
            raise ProviderError(f"message {index} content must be a string")
        validated.append({"role": role, "content": content})
    # The last user message is the request being answered. An empty
    # or whitespace-only one is a caller bug, not a question: fail
    # fast with a clear error instead of echoing nonsense (local
    # engine) or spending an endpoint call on nothing.
    last_user = next(
        (m["content"] for m in reversed(validated) if m["role"] == "user"),
        None,
    )
    if last_user is None or not last_user.strip():
        raise ProviderError(
            "messages must include a user message with non-empty content"
        )
    return validated


class ChatProvider(Protocol):
    def complete(self, messages: Sequence[Message], model: str | None = None) -> str:
        ...


class GravionCoreProvider:
    """The built-in Gravion Core engine.

    Gravion Core runs fully offline. It first tries the compact local
    trained model (bundled weights, or deployment-trained weights when
    a deployment provides them). When weights are missing/invalid or
    the trained model has no confident match, it falls back to the
    deterministic offline responder that proves the full module path
    (config → context → registry → provider → gate) with zero
    infrastructure.
    """

    name = "gravion-core"

    def __init__(
        self,
        model: GravionCoreLocalModel | None = None,
        *,
        weights_path=None,
        home=None,
        specialist_models: dict[str, GravionCoreLocalModel] | None = None,
    ):
        self._home = home
        self._specialist_models = dict(specialist_models or {})
        self._specialist_load_attempted = set(self._specialist_models)
        self._model = model
        if self._model is None:
            if weights_path is not None:
                try:
                    self._model = load_weights(weights_path, model_id=CORE_MODEL_ID)
                except Exception:
                    self._model = None
            else:
                try:
                    self._model = load_bundled_model()
                except Exception:
                    self._model = None

    @property
    def parameter_count(self) -> int:
        """Parameters in the loaded Gravion Core weights, or 0 on fallback."""

        return self._model.parameter_count if self._model is not None else 0

    def _specialist_model(self, model_id: str) -> GravionCoreLocalModel | None:
        if model_id not in SPECIALIST_MODEL_IDS:
            return None
        if model_id in self._specialist_load_attempted:
            return self._specialist_models.get(model_id)
        self._specialist_load_attempted.add(model_id)
        loaded = None
        if self._home is not None:
            deployment_path = deployment_weights_path(self._home, model_id)
            if deployment_path.exists():
                try:
                    loaded = load_weights(deployment_path, model_id=model_id)
                except Exception:
                    loaded = None
        if loaded is None:
            try:
                loaded = load_bundled_model(model_id)
            except Exception:
                loaded = None
        if loaded is not None:
            self._specialist_models[model_id] = loaded
        return loaded

    def parameter_count_for(self, model_id: str | None = None) -> int:
        """Parameters in the local weights for ``model_id``, or 0."""

        if model_id in (None, CORE_MODEL_ID):
            return self.parameter_count
        specialist = self._specialist_model(model_id)
        return specialist.parameter_count if specialist is not None else 0

    def complete(self, messages: Sequence[Message], model: str | None = None) -> str:
        validated = _validate_messages(messages)
        user = next(
            (m["content"] for m in reversed(validated) if m["role"] == "user"),
            "",
        )
        selected = CORE_MODEL_ID if model is None else model
        specialist = self._specialist_model(selected)
        if specialist is not None:
            prediction = specialist.predict(user)
            if prediction is not None:
                return prediction
        # Routing: a specialist with no confident match falls through to
        # Gravion Core (whose own gate only answers Gravion doc-style
        # questions) before the echo fallback, instead of giving up.
        if self._model is not None:
            prediction = self._model.predict(user)
            if prediction is not None:
                return prediction
        return f"[gravion-core] You said: {user}"


class EndpointProvider:
    """Chat completions against a configurable inference endpoint.

    POSTs {"model", "messages"} to {base_url}/chat/completions with a
    bearer API key when one is configured, and reads the reply from
    choices[0].message.content — a deliberately small, generic wire
    contract that any inference server can implement.
    """

    name = "endpoint"

    def __init__(
        self,
        config: ProviderConfig,
        opener=None,
        *,
        sleeper=None,
        clock=None,
        request_id_factory=None,
        log=None,
    ):
        self._config = config
        # `opener` is injectable so tests can exercise the wire format
        # without a network. Sleeper/clock/request-id/log are also
        # injectable so retry and timing behaviour stays testable.
        self._opener = opener or urllib.request.urlopen
        self._sleeper = sleeper or time.sleep
        self._clock = clock or time.monotonic
        self._request_id_factory = request_id_factory or (lambda: uuid.uuid4().hex)
        self._log = log or logger

    def _backoff_seconds(self, attempt: int) -> float:
        base = float(self._config.retry_backoff_seconds)
        return min(30.0, base * (2 ** max(0, attempt - 1)))

    def complete(self, messages: Sequence[Message], model: str | None = None) -> str:
        cfg = self._config
        if not cfg.base_url:
            raise ProviderError(
                "provider.type is 'endpoint' but no base_url is configured"
            )
        validated_messages = _validate_messages(messages)
        selected_model = model or cfg.model
        if not isinstance(selected_model, str) or not selected_model.strip():
            raise ProviderError("endpoint model must be a non-empty string")
        url = cfg.base_url.rstrip("/") + "/chat/completions"
        payload = {
            "model": selected_model,
            "messages": validated_messages,
        }
        if cfg.temperature is not None:
            payload["temperature"] = float(cfg.temperature)
        if cfg.max_tokens is not None:
            payload["max_tokens"] = int(cfg.max_tokens)
        encoded_payload = json.dumps(payload).encode("utf-8")
        request_id = str(self._request_id_factory())
        max_attempts = int(cfg.max_retries) + 1
        started = self._clock()

        for attempt in range(1, max_attempts + 1):
            headers = {
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Gravion-AI",
                "X-Gravion-Request-ID": request_id,
            }
            key = cfg.resolved_api_key()
            if key:
                headers["Authorization"] = f"Bearer {key}"
            request = urllib.request.Request(
                url,
                data=encoded_payload,
                headers=headers,
                method="POST",
            )
            attempt_started = self._clock()
            try:
                with self._opener(request, timeout=cfg.timeout_seconds) as resp:
                    raw = resp.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    raise ProviderError(
                        "endpoint response exceeded the 4 MiB safety limit"
                    )
                body = json.loads(raw.decode("utf-8"))
            except urllib.error.HTTPError as exc:
                duration_ms = round((self._clock() - attempt_started) * 1000)
                retryable = exc.code in _RETRYABLE_HTTP_STATUSES
                if retryable and attempt < max_attempts:
                    delay = self._backoff_seconds(attempt)
                    self._log.warning(
                        "provider retry request_id=%s model=%s attempt=%s/%s "
                        "status=%s timeout_seconds=%s duration_ms=%s "
                        "retry_delay_seconds=%s",
                        request_id,
                        selected_model,
                        attempt,
                        max_attempts,
                        exc.code,
                        cfg.timeout_seconds,
                        duration_ms,
                        delay,
                    )
                    self._sleeper(delay)
                    continue
                self._log.warning(
                    "provider failed request_id=%s model=%s attempt=%s/%s "
                    "status=%s timeout_seconds=%s duration_ms=%s",
                    request_id,
                    selected_model,
                    attempt,
                    max_attempts,
                    exc.code,
                    cfg.timeout_seconds,
                    duration_ms,
                )
                raise ProviderError(f"endpoint returned HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, OSError) as exc:
                duration_ms = round((self._clock() - attempt_started) * 1000)
                if attempt < max_attempts:
                    delay = self._backoff_seconds(attempt)
                    self._log.warning(
                        "provider retry request_id=%s model=%s attempt=%s/%s "
                        "error=%s timeout_seconds=%s duration_ms=%s "
                        "retry_delay_seconds=%s",
                        request_id,
                        selected_model,
                        attempt,
                        max_attempts,
                        exc.__class__.__name__,
                        cfg.timeout_seconds,
                        duration_ms,
                        delay,
                    )
                    self._sleeper(delay)
                    continue
                self._log.warning(
                    "provider failed request_id=%s model=%s attempt=%s/%s "
                    "error=%s timeout_seconds=%s duration_ms=%s",
                    request_id,
                    selected_model,
                    attempt,
                    max_attempts,
                    exc.__class__.__name__,
                    cfg.timeout_seconds,
                    duration_ms,
                )
                raise ProviderError(
                    f"endpoint call failed: {exc.__class__.__name__}"
                ) from exc
            except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                duration_ms = round((self._clock() - attempt_started) * 1000)
                self._log.warning(
                    "provider failed request_id=%s model=%s attempt=%s/%s "
                    "error=%s timeout_seconds=%s duration_ms=%s",
                    request_id,
                    selected_model,
                    attempt,
                    max_attempts,
                    exc.__class__.__name__,
                    cfg.timeout_seconds,
                    duration_ms,
                )
                raise ProviderError(
                    f"endpoint call failed: {exc.__class__.__name__}"
                ) from exc

            try:
                content = body["choices"][0]["message"]["content"]
            except (KeyError, IndexError, TypeError) as exc:
                raise ProviderError("unexpected endpoint response shape") from exc
            if not isinstance(content, str):
                raise ProviderError("endpoint response content must be a string")
            total_ms = round((self._clock() - started) * 1000)
            self._log.info(
                "provider completed request_id=%s model=%s attempts=%s "
                "timeout_seconds=%s duration_ms=%s response_bytes=%s",
                request_id,
                selected_model,
                attempt,
                cfg.timeout_seconds,
                total_ms,
                len(raw),
            )
            return content

        raise ProviderError("endpoint call failed: retry loop exhausted")


def build_provider(config: ProviderConfig, home=None) -> ChatProvider:
    if config.type == "gravion-core":
        if home is not None:
            weights_path = deployment_weights_path(home)
            if weights_path.exists():
                return GravionCoreProvider(weights_path=weights_path, home=home)
            return GravionCoreProvider(home=home)
        return GravionCoreProvider()
    if config.type == "endpoint":
        return EndpointProvider(config)
    raise ProviderError(f"unknown provider type {config.type!r}")

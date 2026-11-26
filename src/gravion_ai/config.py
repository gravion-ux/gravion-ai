"""Deployment configuration.

A Gravion deployment is a directory (the *home*) containing:

    gravion.json     this configuration
    policy.md        the company policy / work rules (injected as context)
    state/           identity, license, entitlement token, audit log
    data/            the deployment's own data (exported by `gravion export`)

Configuration is JSON so it loads with the standard library on every
supported Python (no TOML parser dependency on 3.10).
"""

from __future__ import annotations

import ipaddress
import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

from .errors import ConfigError
from .tiers import DEFAULT_TIER, get_tier

DEFAULT_NAME = "Gravion"
CONFIG_FILENAME = "gravion.json"
POLICY_FILENAME = "policy.md"

MODE_DEVELOPMENT = "development"
MODE_PRODUCTION = "production"
MODES = (MODE_DEVELOPMENT, MODE_PRODUCTION)

CONFIG_FIELDS = frozenset(
    {
        "name",
        "company",
        "mode",
        "tier",
        "provider",
        "authority_url",
        "self_upgrade_permission",
    }
)
PROVIDER_FIELDS = frozenset(
    {
        "type",
        "base_url",
        "model",
        "api_key_env",
        "api_key",
        "timeout_seconds",
        "max_retries",
        "retry_backoff_seconds",
        "temperature",
        "max_tokens",
    }
)
_ENV_VAR_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _is_loopback(hostname: str) -> bool:
    host = hostname.lower().rstrip(".")
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def _validate_url(value: object, field_name: str, *, production: bool = False) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{field_name} must be a non-empty URL string")
    parsed = urlsplit(value)
    try:
        _ = parsed.port
    except ValueError as exc:
        raise ConfigError(f"{field_name} contains an invalid port") from exc
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ConfigError(
            f"{field_name} must be an absolute HTTP(S) URL without "
            "embedded credentials"
        )
    if parsed.query or parsed.fragment:
        raise ConfigError(f"{field_name} must not contain a query string or fragment")
    if production and parsed.scheme == "http" and not _is_loopback(parsed.hostname):
        raise ConfigError(
            f"{field_name} must use HTTPS for remote endpoints in production"
        )
    return value.rstrip("/")


def default_home() -> Path:
    """Resolve the deployment home: $GRAVION_HOME or ~/.gravion-ai."""
    env = os.environ.get("GRAVION_HOME")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".gravion-ai"


@dataclass
class ProviderConfig:
    """How this deployment reaches a model.

    type "gravion-core" uses the built-in Gravion Core engine — fully
    offline, no network, no API key. type "endpoint" talks to any
    configurable inference endpoint the deployment operator runs or
    rents (base URL + API key), speaking a simple chat-completions
    JSON protocol; the model is selected from the Gravion model
    registry (see catalog.py).
    """

    type: str = "gravion-core"
    base_url: str = ""
    model: str = "gravion-core"
    api_key_env: str = "GRAVION_API_KEY"
    api_key: str | None = None
    timeout_seconds: float = 60.0
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    temperature: float | None = None
    max_tokens: int | None = None

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)

    def validate(self, *, production: bool = False) -> None:
        if self.type not in ("gravion-core", "endpoint"):
            raise ConfigError(
                f"provider.type must be 'gravion-core' or 'endpoint', "
                f"got {self.type!r}"
            )
        if not isinstance(self.model, str) or not self.model.strip():
            raise ConfigError("provider.model must be a non-empty string")
        if not isinstance(self.api_key_env, str) or not _ENV_VAR_RE.match(
            self.api_key_env
        ):
            raise ConfigError(
                "provider.api_key_env must be a valid environment variable name"
            )
        if self.api_key is not None and not isinstance(self.api_key, str):
            raise ConfigError("provider.api_key must be a string or null")
        if production and self.api_key:
            raise ConfigError(
                "provider.api_key must not be stored in gravion.json in "
                "production; set it through provider.api_key_env instead"
            )
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not 1 <= float(self.timeout_seconds) <= 300
        ):
            raise ConfigError("provider.timeout_seconds must be between 1 and 300")
        if (
            isinstance(self.max_retries, bool)
            or not isinstance(self.max_retries, int)
            or not 0 <= self.max_retries <= 5
        ):
            raise ConfigError("provider.max_retries must be an integer between 0 and 5")
        if (
            isinstance(self.retry_backoff_seconds, bool)
            or not isinstance(self.retry_backoff_seconds, (int, float))
            or not 0 <= float(self.retry_backoff_seconds) <= 30
        ):
            raise ConfigError(
                "provider.retry_backoff_seconds must be between 0 and 30"
            )
        if self.temperature is not None and (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not 0 <= float(self.temperature) <= 2
        ):
            raise ConfigError("provider.temperature must be between 0 and 2")
        if self.max_tokens is not None and (
            isinstance(self.max_tokens, bool)
            or not isinstance(self.max_tokens, int)
            or not 1 <= self.max_tokens <= 1_000_000
        ):
            raise ConfigError(
                "provider.max_tokens must be an integer between 1 and 1000000"
            )
        if self.type == "endpoint":
            self.base_url = _validate_url(
                self.base_url, "provider.base_url", production=production
            )
        elif self.base_url:
            self.base_url = _validate_url(
                self.base_url, "provider.base_url", production=production
            )


@dataclass
class GravionConfig:
    name: str = DEFAULT_NAME
    company: str = ""
    mode: str = MODE_DEVELOPMENT
    tier: str = DEFAULT_TIER
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    authority_url: str | None = None
    # §9 governed self-upgrade: a standing permission the operator
    # grants (and may revoke) once. Default off — no grant, no
    # self-changes, ever.
    self_upgrade_permission: bool = False
    home: Path = field(default_factory=default_home)

    # -- derived paths -------------------------------------------------
    @property
    def config_path(self) -> Path:
        return self.home / CONFIG_FILENAME

    @property
    def policy_path(self) -> Path:
        return self.home / POLICY_FILENAME

    @property
    def state_dir(self) -> Path:
        return self.home / "state"

    @property
    def data_dir(self) -> Path:
        return self.home / "data"

    # -- validation ----------------------------------------------------
    def validate(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigError("name must be a non-empty string")
        if not isinstance(self.company, str):
            raise ConfigError("company must be a string")
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}, got {self.mode!r}")
        try:
            get_tier(self.tier)
        except ValueError as exc:
            raise ConfigError(str(exc)) from exc
        if not isinstance(self.provider, ProviderConfig):
            raise ConfigError("provider must be a ProviderConfig")
        self.provider.validate(production=self.mode == MODE_PRODUCTION)
        if self.authority_url is not None:
            self.authority_url = _validate_url(
                self.authority_url,
                "authority_url",
                production=self.mode == MODE_PRODUCTION,
            )
        if not isinstance(self.self_upgrade_permission, bool):
            raise ConfigError("self_upgrade_permission must be true or false")

    # -- (de)serialisation ---------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("home", None)
        return data

    @classmethod
    def from_dict(cls, data: dict, home: Path) -> "GravionConfig":
        if not isinstance(data, dict):
            raise ConfigError("gravion.json must contain a JSON object")
        unknown = sorted(set(data) - CONFIG_FIELDS)
        if unknown:
            raise ConfigError(f"unknown config fields: {', '.join(unknown)}")
        provider_data = data.get("provider", {})
        if not isinstance(provider_data, dict):
            raise ConfigError("provider must be a JSON object")
        unknown_provider = sorted(set(provider_data) - PROVIDER_FIELDS)
        if unknown_provider:
            raise ConfigError(
                f"unknown provider fields: {', '.join(unknown_provider)}"
            )
        try:
            provider = ProviderConfig(**provider_data)
            cfg = cls(
                name=data.get("name", DEFAULT_NAME),
                company=data.get("company", ""),
                mode=data.get("mode", MODE_DEVELOPMENT),
                tier=data.get("tier", DEFAULT_TIER),
                provider=provider,
                authority_url=data.get("authority_url"),
                self_upgrade_permission=data.get("self_upgrade_permission", False),
                home=home,
            )
        except TypeError as exc:
            raise ConfigError(f"invalid config shape: {exc}") from exc
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, home: Path | None = None) -> "GravionConfig":
        home = Path(home) if home else default_home()
        path = home / CONFIG_FILENAME
        if not path.exists():
            raise ConfigError(
                f"no Gravion deployment at {home} (missing {CONFIG_FILENAME}); "
                f"run `gravion init` first"
            )
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"cannot read {path}: {exc}") from exc
        return cls.from_dict(data, home)

    def save(self) -> Path:
        self.validate()
        self.home.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.config_path.write_text(
            json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return self.config_path

    # -- company policy --------------------------------------------------
    def load_policy(self) -> str:
        """Return the company policy text (empty string if none set)."""
        if self.policy_path.exists():
            return self.policy_path.read_text(encoding="utf-8").strip()
        return ""

    def system_context(self) -> str:
        """The system prompt prefix every module runs under.

        This is where the instance's name and the company's policy are
        injected: the same modules behave as *this* company's assistant
        because the policy travels with the deployment, not the code.
        """
        lines = [f"You are {self.name}, an AI assistant"]
        if self.company:
            lines[0] += f" for {self.company}"
        lines[0] += "."
        lines.append(
            "You are built on the Gravion AI platform. Be accurate, "
            "concise, and follow the company policy below at all times."
        )
        policy = self.load_policy()
        if policy:
            lines.append("\n# Company policy\n")
            lines.append(policy)
        return "\n".join(lines)

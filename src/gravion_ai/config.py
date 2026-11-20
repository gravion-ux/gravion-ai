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

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .errors import ConfigError
from .tiers import DEFAULT_TIER, get_tier

DEFAULT_NAME = "Gravion"
CONFIG_FILENAME = "gravion.json"
POLICY_FILENAME = "policy.md"

MODE_DEVELOPMENT = "development"
MODE_PRODUCTION = "production"
MODES = (MODE_DEVELOPMENT, MODE_PRODUCTION)


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

    def resolved_api_key(self) -> str | None:
        if self.api_key:
            return self.api_key
        return os.environ.get(self.api_key_env)


@dataclass
class GravionConfig:
    name: str = DEFAULT_NAME
    company: str = ""
    mode: str = MODE_DEVELOPMENT
    tier: str = DEFAULT_TIER
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    authority_url: str | None = None
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
        if self.mode not in MODES:
            raise ConfigError(f"mode must be one of {MODES}, got {self.mode!r}")
        get_tier(self.tier)  # raises ValueError for unknown tiers
        if self.provider.type not in ("gravion-core", "endpoint"):
            raise ConfigError(
                f"provider.type must be 'gravion-core' or 'endpoint', "
                f"got {self.provider.type!r}"
            )

    # -- (de)serialisation ---------------------------------------------
    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("home", None)
        return data

    @classmethod
    def from_dict(cls, data: dict, home: Path) -> "GravionConfig":
        provider = ProviderConfig(**data.get("provider", {}))
        cfg = cls(
            name=data.get("name", DEFAULT_NAME),
            company=data.get("company", ""),
            mode=data.get("mode", MODE_DEVELOPMENT),
            tier=data.get("tier", DEFAULT_TIER),
            provider=provider,
            authority_url=data.get("authority_url"),
            home=home,
        )
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

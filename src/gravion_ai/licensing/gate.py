"""The entitlement gate — where enforcement actually happens.

Every module run in production mode passes through the gate:

    development mode → always allowed (no license needed, ever)
    production mode  → allowed only with a valid entitlement:
        1. a license document is present for this deployment,
        2. the license verifies (issuer signature, expiry, module),
        3. a signed entitlement token from the authority covers the
           module — freshly fetched, or a cached one still inside its
           TTL when the authority cannot be reached.

Any failure puts the deployment into the SUSPENDED state for that
check: the module does not run. Suspension never touches data — see
runtime.export_bundle, which works in every state.

Every gate decision is written to the tamper-evident audit log.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

from ..config import MODE_DEVELOPMENT, GravionConfig
from ..errors import LicensingError, SuspendedError
from .audit import AuditLog
from .authority import (
    EntitlementClient,
    build_entitlement_request,
    verify_entitlement_token,
)
from .identity import DeploymentIdentity
from .keys import KeyPair
from .license_doc import load_license, verify_license
from .timeutil import utcnow

LICENSE_FILENAME = "license.json"
TOKEN_FILENAME = "entitlement-token.json"


class GateState(str, Enum):
    DEVELOPMENT = "development"
    ACTIVE = "active"
    SUSPENDED = "suspended"


@dataclass(frozen=True)
class GateDecision:
    state: GateState
    allowed: bool
    module: str
    reason: str = ""


class EntitlementGate:
    def __init__(
        self,
        config: GravionConfig,
        *,
        issuer_public: KeyPair | None = None,
        client: EntitlementClient | None = None,
        audit: AuditLog | None = None,
        now_fn=None,
    ):
        self._config = config
        self._issuer_public = issuer_public
        self._client = client
        self._audit = audit or AuditLog(config.state_dir / "audit.jsonl")
        self._now_fn = now_fn or utcnow
        self._identity: DeploymentIdentity | None = None

    # -- helpers ---------------------------------------------------------
    def _now(self) -> datetime:
        return self._now_fn()

    def _deployment_id(self) -> str:
        if self._identity is None:
            self._identity = DeploymentIdentity.create_or_load(
                self._config.state_dir
            )
        return self._identity.deployment_id

    def _suspend(self, module: str, reason: str) -> GateDecision:
        self._audit.append(
            "gate.suspend", {"module": module, "reason": reason}
        )
        return GateDecision(GateState.SUSPENDED, False, module, reason)

    def _cached_token(self) -> dict | None:
        path = self._config.state_dir / TOKEN_FILENAME
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def _cache_token(self, token: dict) -> None:
        path = self._config.state_dir / TOKEN_FILENAME
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(token, indent=2) + "\n", encoding="utf-8")

    # -- the check ---------------------------------------------------------
    def check(self, module: str) -> GateDecision:
        cfg = self._config
        if cfg.mode == MODE_DEVELOPMENT:
            self._audit.append("gate.allow", {"module": module, "mode": "development"})
            return GateDecision(GateState.DEVELOPMENT, True, module)

        # Production mode from here on.
        if self._issuer_public is None:
            return self._suspend(module, "no issuer/authority key configured")

        license_path = cfg.state_dir / LICENSE_FILENAME
        if not license_path.exists():
            return self._suspend(
                module,
                "no license installed; production use requires a signed "
                "commercial agreement and license (see docs/LICENSING.md)",
            )
        try:
            document = load_license(license_path)
            deployment_id = self._deployment_id()
            claims = verify_license(
                document,
                self._issuer_public,
                now=self._now(),
                module=module,
                deployment_id=deployment_id,
            )
        except LicensingError as exc:
            return self._suspend(module, f"license invalid: {exc}")

        # Prefer a still-valid cached token; refresh when stale/absent.
        cached = self._cached_token()
        if cached is not None:
            try:
                verify_entitlement_token(
                    cached,
                    self._issuer_public,
                    now=self._now(),
                    module=module,
                    deployment_id=deployment_id,
                )
                self._audit.append(
                    "gate.allow",
                    {"module": module, "mode": "production", "via": "cached-token",
                     "holder": claims.holder},
                )
                return GateDecision(GateState.ACTIVE, True, module)
            except LicensingError:
                pass  # stale or not covering — fall through to a fresh fetch

        if self._client is None:
            return self._suspend(
                module,
                "no entitlement authority configured and no valid cached token",
            )
        try:
            request = build_entitlement_request(
                deployment_id, document, module, now=self._now()
            )
            token = self._client.fetch_token(request)
            verify_entitlement_token(
                token,
                self._issuer_public,
                now=self._now(),
                module=module,
                deployment_id=deployment_id,
            )
        except LicensingError as exc:
            return self._suspend(module, f"entitlement check failed: {exc}")
        self._cache_token(token)
        self._audit.append(
            "gate.allow",
            {"module": module, "mode": "production", "via": "authority",
             "holder": claims.holder},
        )
        return GateDecision(GateState.ACTIVE, True, module)

    def require(self, module: str) -> GateDecision:
        """Like check(), but raises SuspendedError when not allowed."""
        decision = self.check(module)
        if not decision.allowed:
            raise SuspendedError(
                f"production use suspended for module {module!r}: {decision.reason}"
            )
        return decision

"""Signed license documents.

A license is the *technical mirror* of a signed commercial agreement
(the legal layer — see docs/LICENSING.md). It is a JSON document:

    {
      "payload": {
        "license_id": "...", "deployment_id": "gravion-dep-...",
        "holder": "Acme Robotics Pvt Ltd",
        "tier": "tier-3-intermediate",
        "modules": ["chat", "support", ...]        // or ["*"]
        "issued_at": "...", "expires_at": "...",
        "agreement": {"reference": "GRAVION-CA-2026-001",
                      "equity_percent": 15,
                      "signed_at": "..."},
        "issuer": "<issuer key fingerprint>"
      },
      "signature": "<base64 Ed25519 signature over canonical payload>"
    }

Verification checks the issuer signature, expiry, and — per request —
that a module is actually entitled. The equity percentage is carried
in the document so the technical record can never drift from the
commercial agreement it mirrors; issuance refuses values outside the
approved 10–20% partnership band.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

from ..canonical import canonical_json, document_hash
from ..errors import (
    LicenseError,
    LicenseExpiredError,
    NotEntitledError,
)
from ..tiers import get_tier
from .keys import KeyPair, b64decode, b64encode
from .timeutil import from_iso, to_iso, utcnow

EQUITY_MIN_PERCENT = 10
EQUITY_MAX_PERCENT = 20


@dataclass(frozen=True)
class LicenseClaims:
    license_id: str
    deployment_id: str
    holder: str
    tier: str
    modules: tuple[str, ...]
    issued_at: datetime
    expires_at: datetime
    agreement_reference: str
    equity_percent: float
    issuer: str

    def entitles(self, module: str) -> bool:
        return "*" in self.modules or module in self.modules

    def require_module(self, module: str) -> None:
        if not self.entitles(module):
            raise NotEntitledError(
                f"license for {self.holder!r} does not entitle module "
                f"{module!r} (entitled: {sorted(self.modules)})"
            )


def issue_license(
    issuer: KeyPair,
    *,
    deployment_id: str,
    holder: str,
    tier: str,
    modules: list[str],
    agreement_reference: str,
    equity_percent: float,
    days: int = 365,
    now: datetime | None = None,
) -> dict:
    """Create and sign a license. Issuer-side tooling (project owner)."""
    get_tier(tier)  # validate tier id
    if not (EQUITY_MIN_PERCENT <= equity_percent <= EQUITY_MAX_PERCENT):
        raise LicenseError(
            f"equity_percent must be within the approved partnership band "
            f"{EQUITY_MIN_PERCENT}–{EQUITY_MAX_PERCENT}%, got {equity_percent}"
        )
    if not modules:
        raise LicenseError("a license must entitle at least one module")
    now = now or utcnow()
    payload = {
        "license_id": uuid.uuid4().hex,
        "deployment_id": deployment_id,
        "holder": holder,
        "tier": tier,
        "modules": sorted(modules),
        "issued_at": to_iso(now),
        "expires_at": to_iso(now + timedelta(days=days)),
        "agreement": {
            "reference": agreement_reference,
            "equity_percent": equity_percent,
            "signed_at": to_iso(now),
        },
        "issuer": issuer.key_id,
    }
    signature = issuer.sign(canonical_json(payload))
    return {"payload": payload, "signature": b64encode(signature)}


def license_hash(document: dict) -> str:
    """Hash of the license payload — the only license-derived value a
    verification request ever carries (see authority.py)."""
    return document_hash(document["payload"])


def verify_license(
    document: dict,
    issuer_public: KeyPair,
    *,
    now: datetime | None = None,
    module: str | None = None,
    deployment_id: str | None = None,
) -> LicenseClaims:
    """Verify signature, issuer, expiry and (optionally) entitlement.

    Raises BadSignatureError, LicenseExpiredError, NotEntitledError or
    LicenseError on any failure; returns the claims on success.
    """
    try:
        payload = document["payload"]
        signature = b64decode(document["signature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LicenseError(f"malformed license document: {exc}") from exc
    if payload.get("issuer") != issuer_public.key_id:
        raise LicenseError("license was issued by an unknown issuer key")
    issuer_public.verify(canonical_json(payload), signature)  # BadSignatureError
    now = now or utcnow()
    expires_at = from_iso(payload["expires_at"])
    if now >= expires_at:
        raise LicenseExpiredError(
            f"license {payload.get('license_id')} expired at {payload['expires_at']}"
        )
    if deployment_id is not None and payload.get("deployment_id") != deployment_id:
        raise LicenseError(
            "license belongs to a different deployment "
            f"({payload.get('deployment_id')})"
        )
    claims = LicenseClaims(
        license_id=payload["license_id"],
        deployment_id=payload["deployment_id"],
        holder=payload["holder"],
        tier=payload["tier"],
        modules=tuple(payload["modules"]),
        issued_at=from_iso(payload["issued_at"]),
        expires_at=expires_at,
        agreement_reference=payload["agreement"]["reference"],
        equity_percent=payload["agreement"]["equity_percent"],
        issuer=payload["issuer"],
    )
    if module is not None:
        claims.require_module(module)
    return claims


def save_license(document: dict, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def load_license(path: Path) -> dict:
    path = Path(path)
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LicenseError(f"cannot read license at {path}: {exc}") from exc

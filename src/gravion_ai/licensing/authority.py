"""Server-side entitlement enforcement — the verification protocol.

Local checks alone can be patched out, so production entitlement is
anchored server-side: the deployment asks the licensing authority
"is this license, on this deployment, entitled to run this module?"
and receives a short-lived **signed entitlement token** in return.
The full protocol spec lives in docs/LICENSING.md; this module is the
reference implementation of both ends.

Privacy by design — a verification request contains exactly four
fields and nothing else:

    deployment_id   who is asking (a key fingerprint, not a person)
    license_hash    SHA-256 of the license payload (not the license)
    module          which module is being gated
    requested_at    timestamp

No prompts, no completions, no user data, no telemetry, no IP-derived
identifiers beyond what the transport itself reveals. The tests pin
this field set so it cannot quietly grow.

Availability: the production gate honours a still-valid cached token
when the authority is unreachable, and only suspends when neither a
fresh nor a cached token can authorise the module.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Protocol

from ..canonical import canonical_json
from ..errors import (
    LicenseError,
    LicenseExpiredError,
    NotEntitledError,
)
from .keys import KeyPair, b64decode, b64encode
from .license_doc import license_hash, verify_license
from .timeutil import from_iso, to_iso, utcnow

DEFAULT_TOKEN_TTL_SECONDS = 900

# The complete, pinned field set of an entitlement request. Anything
# beyond this is a protocol violation (and a privacy bug).
REQUEST_FIELDS = frozenset(
    {"deployment_id", "license_hash", "module", "requested_at"}
)


@dataclass(frozen=True)
class EntitlementRequest:
    deployment_id: str
    license_hash: str
    module: str
    requested_at: str

    def to_dict(self) -> dict:
        return {
            "deployment_id": self.deployment_id,
            "license_hash": self.license_hash,
            "module": self.module,
            "requested_at": self.requested_at,
        }


def build_entitlement_request(
    deployment_id: str,
    license_document: dict,
    module: str,
    *,
    now: datetime | None = None,
) -> EntitlementRequest:
    return EntitlementRequest(
        deployment_id=deployment_id,
        license_hash=license_hash(license_document),
        module=module,
        requested_at=to_iso(now or utcnow()),
    )


def validate_request_payload(payload: dict) -> EntitlementRequest:
    """Server-side: refuse any request carrying unexpected fields."""
    keys = frozenset(payload)
    if keys != REQUEST_FIELDS:
        raise LicenseError(
            f"entitlement request must contain exactly {sorted(REQUEST_FIELDS)}, "
            f"got {sorted(keys)}"
        )
    return EntitlementRequest(**payload)


class MockAuthority:
    """In-process reference authority, for tests and offline development.

    Holds the issuer key server-side and a registry of issued licenses
    (keyed by license hash), exactly like the real licensing server —
    minus the network. It re-verifies the license on every request:
    signature, expiry, deployment binding and module entitlement.
    """

    def __init__(self, issuer: KeyPair, *, ttl_seconds: int = DEFAULT_TOKEN_TTL_SECONDS):
        self._issuer = issuer
        self._ttl = ttl_seconds
        self._licenses: dict[str, dict] = {}

    def register_license(self, license_document: dict) -> str:
        digest = license_hash(license_document)
        self._licenses[digest] = license_document
        return digest

    def handle_request(self, payload: dict, *, now: datetime | None = None) -> dict:
        request = validate_request_payload(payload)
        document = self._licenses.get(request.license_hash)
        if document is None:
            raise LicenseError("unknown license (hash not registered)")
        claims = verify_license(
            document,
            self._issuer,
            now=now,
            module=request.module,
            deployment_id=request.deployment_id,
        )
        now = now or utcnow()
        token_payload = {
            "deployment_id": request.deployment_id,
            "license_hash": request.license_hash,
            "holder": claims.holder,
            "tier": claims.tier,
            "modules": sorted(claims.modules),
            "issued_at": to_iso(now),
            "expires_at": to_iso(now + timedelta(seconds=self._ttl)),
            "authority": self._issuer.key_id,
        }
        signature = self._issuer.sign(canonical_json(token_payload))
        return {"payload": token_payload, "signature": b64encode(signature)}


def verify_entitlement_token(
    token: dict,
    authority_public: KeyPair,
    *,
    now: datetime | None = None,
    module: str | None = None,
    deployment_id: str | None = None,
) -> dict:
    """Verify a token's signature, expiry and coverage. Returns payload."""
    try:
        payload = token["payload"]
        signature = b64decode(token["signature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LicenseError(f"malformed entitlement token: {exc}") from exc
    if payload.get("authority") != authority_public.key_id:
        raise LicenseError("token was issued by an unknown authority key")
    authority_public.verify(canonical_json(payload), signature)  # BadSignatureError
    now = now or utcnow()
    if now >= from_iso(payload["expires_at"]):
        raise LicenseExpiredError("entitlement token expired")
    if deployment_id is not None and payload.get("deployment_id") != deployment_id:
        raise LicenseError("entitlement token belongs to a different deployment")
    if module is not None:
        modules = payload.get("modules", [])
        if "*" not in modules and module not in modules:
            raise NotEntitledError(
                f"entitlement token does not cover module {module!r}"
            )
    return payload


class EntitlementClient(Protocol):
    def fetch_token(self, request: EntitlementRequest) -> dict:
        ...


class MockAuthorityClient:
    """Client that talks to an in-process MockAuthority."""

    def __init__(self, authority: MockAuthority):
        self._authority = authority

    def fetch_token(self, request: EntitlementRequest) -> dict:
        return self._authority.handle_request(request.to_dict())


class HttpAuthorityClient:
    """Client for the real licensing server (see docs/LICENSING.md).

    POSTs the four-field request as JSON to {base_url}/v1/entitlement
    and expects the signed token document back. Kept deliberately thin:
    all verification intelligence lives in verify_entitlement_token.
    """

    MAX_RESPONSE_BYTES = 1024 * 1024

    def __init__(self, base_url: str, *, timeout: float = 10.0, opener=None):
        self._url = base_url.rstrip("/") + "/v1/entitlement"
        self._timeout = timeout
        self._opener = opener or urllib.request.urlopen

    def fetch_token(self, request: EntitlementRequest) -> dict:
        req = urllib.request.Request(
            self._url,
            data=json.dumps(request.to_dict()).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with self._opener(req, timeout=self._timeout) as resp:
                raw = resp.read(self.MAX_RESPONSE_BYTES + 1)
            if len(raw) > self.MAX_RESPONSE_BYTES:
                raise LicenseError(
                    "entitlement authority response exceeded the 1 MiB safety limit"
                )
            return json.loads(raw.decode("utf-8"))
        except LicenseError:
            raise
        except (
            urllib.error.URLError,
            OSError,
            UnicodeDecodeError,
            json.JSONDecodeError,
        ) as exc:
            raise LicenseError(
                f"entitlement authority unreachable at {self._url}: {exc}"
            ) from exc

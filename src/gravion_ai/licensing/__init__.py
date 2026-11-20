"""The Gravion AI licensing subsystem.

Two layers, deliberately separate (see docs/LICENSING.md):

* **Legal entitlement** — the commercial agreement a customer signs
  before production use (docs/LICENSING.md, LICENSE-COMMERCIAL.md).
  Lawyers own that layer; this package only *references* it (every
  license carries the agreement reference and equity percentage).

* **Technical enforcement** — this package: cryptographic deployment
  identity, issuer-signed licenses, issuer-signed module manifests, a
  tamper-evident audit log, and a server-side entitlement protocol.
  Enforcement is *safe*: the worst state it can put a deployment into
  is SUSPENDED — inference stops, data is never deleted, export always
  works. There is no kill switch, by design and by contract.
"""

from .audit import AuditLog
from .authority import (
    EntitlementClient,
    EntitlementRequest,
    HttpAuthorityClient,
    MockAuthority,
    MockAuthorityClient,
    build_entitlement_request,
    verify_entitlement_token,
)
from .gate import EntitlementGate, GateDecision, GateState
from .identity import DeploymentIdentity
from .keys import KeyPair, generate_keypair
from .license_doc import (
    LicenseClaims,
    issue_license,
    license_hash,
    load_license,
    verify_license,
)
from .modules_sig import build_manifest, sign_manifest, verify_module

__all__ = [
    "AuditLog",
    "DeploymentIdentity",
    "EntitlementClient",
    "EntitlementGate",
    "EntitlementRequest",
    "GateDecision",
    "GateState",
    "HttpAuthorityClient",
    "KeyPair",
    "LicenseClaims",
    "MockAuthority",
    "MockAuthorityClient",
    "build_entitlement_request",
    "build_manifest",
    "generate_keypair",
    "issue_license",
    "license_hash",
    "load_license",
    "sign_manifest",
    "verify_entitlement_token",
    "verify_license",
    "verify_module",
]

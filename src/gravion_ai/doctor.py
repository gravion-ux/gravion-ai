"""Production-readiness checks for a Gravion deployment.

`gravion doctor` is deliberately read-only: it does not run a module,
does not fetch an entitlement token, and does not create an identity
or audit entries. It answers the operator's pre-flight question —
"is this deployment configured well enough to serve?" — with
PASS / WARN / FAIL checks and actionable hints.
"""

from __future__ import annotations

from dataclasses import dataclass

from .catalog import ModelCatalog
from .config import MODE_PRODUCTION, GravionConfig
from .errors import CatalogError, GravionError
from .licensing import AuditLog, DeploymentIdentity, load_license, verify_license
from .licensing.gate import LICENSE_FILENAME
from .runtime import ISSUER_PUBLIC_FILENAME, load_issuer_public


@dataclass(frozen=True)
class DoctorCheck:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _production(config: GravionConfig) -> bool:
    return config.mode == MODE_PRODUCTION


def run_doctor(config: GravionConfig) -> list[DoctorCheck]:
    checks: list[DoctorCheck] = []

    def add(name: str, status: str, detail: str) -> None:
        checks.append(DoctorCheck(name=name, status=status, detail=detail))

    try:
        config.validate()
        add("config", "PASS", f"{config.mode} mode, tier {config.tier}")
    except GravionError as exc:
        add("config", "FAIL", str(exc))
        return checks

    policy = config.load_policy()
    if policy:
        add("policy", "PASS", f"policy.md present ({len(policy)} chars)")
    elif _production(config):
        add("policy", "FAIL", "policy.md is missing or empty; production deployments must set company policy")
    else:
        add("policy", "WARN", "policy.md is missing or empty; set company policy before production")

    if DeploymentIdentity.exists(config.state_dir):
        try:
            identity = DeploymentIdentity.load(config.state_dir)
            add("identity", "PASS", f"deployment id {identity.deployment_id}")
        except GravionError as exc:
            add("identity", "FAIL", str(exc))
    else:
        add("identity", "FAIL", "no deployment identity; run `gravion license identity`")

    audit_path = config.state_dir / "audit.jsonl"
    if not audit_path.exists():
        add("audit", "FAIL", "audit log is missing; run `gravion init` or restore state/")
    else:
        try:
            count = AuditLog(audit_path).verify()
            add("audit", "PASS", f"audit chain valid ({count} entries)")
        except GravionError as exc:
            add("audit", "FAIL", str(exc))

    try:
        catalog = ModelCatalog.load_for_home(config.home)
        add("catalog", "PASS", f"{len(catalog.list())} Gravion models registered")
    except CatalogError as exc:
        add("catalog", "FAIL", str(exc))

    provider = config.provider
    if provider.type == "gravion-core":
        add("provider", "PASS", "built-in Gravion Core engine (offline)")
    else:
        add("provider", "PASS", f"endpoint {provider.base_url} model {provider.model}")
        if provider.api_key:
            status = "FAIL" if _production(config) else "WARN"
            add(
                "provider-secret",
                status,
                "API key is stored in gravion.json; move it to the environment variable named by provider.api_key_env",
            )
        elif provider.resolved_api_key():
            add("provider-secret", "PASS", f"API key available via {provider.api_key_env}")
        else:
            add(
                "provider-secret",
                "WARN",
                f"no API key found in {provider.api_key_env}; local endpoints may not need one",
            )

    if _production(config):
        issuer_public = load_issuer_public(config)
        if issuer_public is None:
            add(
                "licensing",
                "FAIL",
                f"issuer public key missing at state/{ISSUER_PUBLIC_FILENAME}",
            )
        else:
            license_path = config.state_dir / LICENSE_FILENAME
            if not license_path.exists():
                add("licensing", "FAIL", "no license installed; production use requires a signed commercial agreement and license")
            else:
                try:
                    document = load_license(license_path)
                    deployment_id = None
                    if DeploymentIdentity.exists(config.state_dir):
                        deployment_id = DeploymentIdentity.load(
                            config.state_dir
                        ).deployment_id
                    claims = verify_license(
                        document,
                        issuer_public,
                        deployment_id=deployment_id,
                    )
                    add(
                        "licensing",
                        "PASS",
                        f"license valid for {claims.holder} until {claims.expires_at.isoformat()}",
                    )
                except GravionError as exc:
                    add("licensing", "FAIL", str(exc))
        if config.authority_url:
            add("authority", "PASS", f"entitlement authority configured: {config.authority_url}")
        else:
            add(
                "authority",
                "WARN",
                "no authority_url configured; production runs need a valid cached entitlement token",
            )
    else:
        add("licensing", "PASS", "development mode needs no license")

    return checks


def doctor_ok(checks: list[DoctorCheck]) -> bool:
    return all(check.status != "FAIL" for check in checks)

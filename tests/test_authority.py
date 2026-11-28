import pytest

from gravion_ai.errors import BadSignatureError, LicenseError, LicenseExpiredError
from gravion_ai.licensing import (
    DeploymentIdentity,
    MockAuthority,
    MockAuthorityClient,
    build_entitlement_request,
    issue_license,
    verify_entitlement_token,
)
from gravion_ai.licensing.authority import REQUEST_FIELDS


def _setup(issuer, tmp_path, ttl=900):
    identity = DeploymentIdentity.create_or_load(tmp_path)
    document = issue_license(
        issuer,
        deployment_id=identity.deployment_id,
        holder="Acme Robotics",
        tier="tier-2-basic",
        modules=["chat"],
        agreement_reference="GRAVION-CA-2026-001",
        equity_percent=10,
        days=30,
    )
    authority = MockAuthority(issuer, ttl_seconds=ttl)
    authority.register_license(document)
    return identity, document, authority


def test_mock_authority_issues_verifiable_token(issuer, tmp_path):
    identity, document, authority = _setup(issuer, tmp_path)
    client = MockAuthorityClient(authority)
    request = build_entitlement_request(identity.deployment_id, document, "chat")
    token = client.fetch_token(request)
    payload = verify_entitlement_token(
        token, issuer, module="chat", deployment_id=identity.deployment_id
    )
    assert payload["holder"] == "Acme Robotics"
    assert payload["modules"] == ["chat"]


def test_request_payload_contains_only_the_specified_fields(issuer, tmp_path):
    """Privacy: the verification request must never grow extra fields —
    no user data, no content, no telemetry."""
    identity, document, _ = _setup(issuer, tmp_path)
    payload = build_entitlement_request(
        identity.deployment_id, document, "chat"
    ).to_dict()
    assert set(payload) == set(REQUEST_FIELDS)
    assert set(payload) == {
        "deployment_id", "license_hash", "module", "requested_at",
    }
    blob = str(payload).lower()
    for forbidden in ("prompt", "content", "user", "email", "telemetry", "ip"):
        assert forbidden not in blob


def test_authority_rejects_request_with_extra_fields(issuer, tmp_path):
    identity, document, authority = _setup(issuer, tmp_path)
    payload = build_entitlement_request(
        identity.deployment_id, document, "chat"
    ).to_dict()
    payload["user_email"] = "someone@example.com"
    with pytest.raises(LicenseError, match="exactly"):
        authority.handle_request(payload)


def test_authority_rejects_unknown_license(issuer, tmp_path):
    identity, document, authority = _setup(issuer, tmp_path)
    other = issue_license(
        issuer,
        deployment_id=identity.deployment_id,
        holder="Someone Else",
        tier="tier-2-basic",
        modules=["chat"],
        agreement_reference="GRAVION-CA-2026-999",
        equity_percent=10,
        days=30,
    )
    request = build_entitlement_request(identity.deployment_id, other, "chat")
    with pytest.raises(LicenseError, match="unknown license"):
        authority.handle_request(request.to_dict())


def test_authority_rejects_unentitled_module(issuer, tmp_path):
    identity, document, authority = _setup(issuer, tmp_path)
    request = build_entitlement_request(identity.deployment_id, document, "video")
    with pytest.raises(LicenseError):
        authority.handle_request(request.to_dict())


def test_tampered_token_rejected(issuer, tmp_path):
    identity, document, authority = _setup(issuer, tmp_path)
    token = MockAuthorityClient(authority).fetch_token(
        build_entitlement_request(identity.deployment_id, document, "chat")
    )
    token["payload"]["modules"] = ["*"]
    with pytest.raises(BadSignatureError):
        verify_entitlement_token(token, issuer, module="chat")


def test_expired_token_rejected(issuer, tmp_path):
    identity, document, authority = _setup(issuer, tmp_path, ttl=-1)
    token = MockAuthorityClient(authority).fetch_token(
        build_entitlement_request(identity.deployment_id, document, "chat")
    )
    with pytest.raises(LicenseExpiredError):
        verify_entitlement_token(token, issuer, module="chat")

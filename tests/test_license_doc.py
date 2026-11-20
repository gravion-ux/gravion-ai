from datetime import timedelta

import pytest

from gravion_ai.errors import (
    BadSignatureError,
    LicenseError,
    LicenseExpiredError,
    NotEntitledError,
)
from gravion_ai.licensing import issue_license, verify_license
from gravion_ai.licensing.timeutil import utcnow

DEP = "gravion-dep-" + "a" * 40


def _issue(issuer, **overrides):
    kwargs = dict(
        deployment_id=DEP,
        holder="Acme Robotics",
        tier="tier-3-intermediate",
        modules=["chat", "support"],
        agreement_reference="GRAVION-CA-2026-001",
        equity_percent=15,
        days=365,
    )
    kwargs.update(overrides)
    return issue_license(issuer, **kwargs)


def test_issue_and_verify_happy_path(issuer):
    document = _issue(issuer)
    claims = verify_license(document, issuer, module="chat", deployment_id=DEP)
    assert claims.holder == "Acme Robotics"
    assert claims.tier == "tier-3-intermediate"
    assert claims.equity_percent == 15
    assert claims.agreement_reference == "GRAVION-CA-2026-001"


def test_tampered_license_rejected(issuer):
    document = _issue(issuer)
    document["payload"]["modules"] = ["*"]  # privilege escalation attempt
    with pytest.raises(BadSignatureError):
        verify_license(document, issuer)


def test_expired_license_rejected(issuer):
    past = utcnow() - timedelta(days=10)
    document = _issue(issuer, days=5, now=past)
    with pytest.raises(LicenseExpiredError):
        verify_license(document, issuer)


def test_wrong_module_rejected(issuer):
    document = _issue(issuer)
    with pytest.raises(NotEntitledError):
        verify_license(document, issuer, module="video")


def test_wrong_deployment_rejected(issuer):
    document = _issue(issuer)
    with pytest.raises(LicenseError):
        verify_license(document, issuer, deployment_id="gravion-dep-" + "b" * 40)


def test_wrong_issuer_rejected(issuer):
    from gravion_ai.licensing import KeyPair

    document = _issue(issuer)
    with pytest.raises(LicenseError):
        verify_license(document, KeyPair.generate())


def test_equity_outside_approved_band_refused(issuer):
    with pytest.raises(LicenseError, match="equity"):
        _issue(issuer, equity_percent=25)
    with pytest.raises(LicenseError, match="equity"):
        _issue(issuer, equity_percent=5)


def test_wildcard_module_entitles_everything(issuer):
    document = _issue(issuer, modules=["*"])
    claims = verify_license(document, issuer, module="video")
    assert claims.entitles("anything")

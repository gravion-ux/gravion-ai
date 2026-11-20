from datetime import timedelta

import pytest

from gravion_ai.errors import SuspendedError
from gravion_ai.licensing import EntitlementGate, MockAuthorityClient
from gravion_ai.licensing.gate import GateState
from gravion_ai.licensing.timeutil import utcnow


def test_development_mode_runs_without_any_license(make_config):
    config = make_config(mode="development")
    gate = EntitlementGate(config)
    decision = gate.check("chat")
    assert decision.allowed and decision.state is GateState.DEVELOPMENT
    gate.require("chat")  # does not raise


def test_production_blocks_without_license(make_config):
    config = make_config(mode="production")
    gate = EntitlementGate(config, issuer_public=None)
    decision = gate.check("chat")
    assert not decision.allowed and decision.state is GateState.SUSPENDED
    with pytest.raises(SuspendedError):
        gate.require("chat")


def test_production_runs_with_valid_entitlement(issuer, licensed_home):
    config, authority, _ = licensed_home()
    gate = EntitlementGate(
        config, issuer_public=issuer, client=MockAuthorityClient(authority)
    )
    decision = gate.require("chat")
    assert decision.allowed and decision.state is GateState.ACTIVE


def test_production_uses_cached_token_when_authority_unreachable(issuer, licensed_home):
    config, authority, _ = licensed_home()
    gate = EntitlementGate(
        config, issuer_public=issuer, client=MockAuthorityClient(authority)
    )
    gate.require("chat")  # fetches + caches

    class DeadClient:
        def fetch_token(self, request):
            raise AssertionError("authority must not be called — cache should serve")

    gate2 = EntitlementGate(config, issuer_public=issuer, client=DeadClient())
    decision = gate2.check("chat")
    assert decision.allowed and decision.state is GateState.ACTIVE


def test_production_suspends_when_license_expires(issuer, licensed_home):
    config, authority, _ = licensed_home(days=1)
    future = utcnow() + timedelta(days=2)
    gate = EntitlementGate(
        config,
        issuer_public=issuer,
        client=MockAuthorityClient(authority),
        now_fn=lambda: future,
    )
    decision = gate.check("chat")
    assert not decision.allowed and decision.state is GateState.SUSPENDED
    assert "expired" in decision.reason


def test_production_suspends_for_unentitled_module(issuer, licensed_home):
    config, authority, _ = licensed_home(modules=("chat",))
    gate = EntitlementGate(
        config, issuer_public=issuer, client=MockAuthorityClient(authority)
    )
    decision = gate.check("video")
    assert not decision.allowed and decision.state is GateState.SUSPENDED


def test_gate_decisions_are_audited(issuer, licensed_home):
    from gravion_ai.licensing import AuditLog

    config, authority, _ = licensed_home()
    gate = EntitlementGate(
        config, issuer_public=issuer, client=MockAuthorityClient(authority)
    )
    gate.require("chat")
    gate.check("video")
    log = AuditLog(config.state_dir / "audit.jsonl")
    events = [e["event"] for e in log.entries()]
    assert "gate.allow" in events and "gate.suspend" in events
    assert log.verify() == len(events)

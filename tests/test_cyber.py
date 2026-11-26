from pathlib import Path

import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.cyber.module import (
    TASK_ANALYSE,
    TASK_REPORT,
    TASK_TRIAGE,
    build_brief,
    detect_task,
    extract_indicators,
    parse_severity,
)
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry
from gravion_ai.runtime import GravionRuntime

SHA256 = "a" * 64


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None):
        self.calls.append((list(messages), model))
        return "recorded reply"


def _ctx(provider, model="gravion-shield"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def test_cyber_instantiates_and_answers():
    module = default_registry().instantiate("cyber")
    out = module.run(_ctx(GravionCoreProvider()), "analyse this firewall log excerpt")
    assert "analyse this firewall log excerpt" in out


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("triage: alert from the IDS about 10.0.0.8", TASK_TRIAGE),
        ("Triage this alert: repeated logins from one IP", TASK_TRIAGE),
        ("please triage the phishing report", TASK_TRIAGE),
        ("report: CVE-2024-1234 in our VPN gateway", TASK_REPORT),
        ("write up this finding for the client", TASK_REPORT),
        ("write a report on the incident", TASK_REPORT),
        ("analyse this firewall log excerpt", TASK_ANALYSE),
        ("what does this config do?", TASK_ANALYSE),
        # Natural phrasings that used to fall through to analyse:
        ("can you triage this alert?", TASK_TRIAGE),
        ("could you triage the phishing report?", TASK_TRIAGE),
        ("could you write up this finding?", TASK_REPORT),
        ("can you report this incident?", TASK_REPORT),
        ("please write a report on the incident", TASK_REPORT),
        # Analysis openers win over later report/triage nouns:
        ("analyse this triage report", TASK_ANALYSE),
        ("explain the triage process", TASK_ANALYSE),
    ],
)
def test_detect_task(prompt, expected):
    assert detect_task(prompt) == expected


def test_extract_indicators_by_kind():
    prompt = (
        "host 10.0.0.8 fetched https://evil.example/payload and "
        "dropped hash " + SHA256 + " — relates to cve-2024-1234"
    )
    indicators = extract_indicators(prompt)
    assert indicators["ipv4"] == ["10.0.0.8"]
    assert indicators["url"] == ["https://evil.example/payload"]
    assert indicators["sha256"] == [SHA256]
    assert indicators["cve"] == ["CVE-2024-1234"]


def test_extract_indicators_dedupes_and_validates():
    prompt = "10.0.0.8 10.0.0.8 999.1.1.1 and 10.0.0.9"
    indicators = extract_indicators(prompt)
    # The duplicate collapses; the invalid octet (999) is refused.
    assert indicators["ipv4"] == ["10.0.0.8", "10.0.0.9"]
    assert extract_indicators("a plain question about passwords") == {}


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("severity: high — IDS alert", "high"),
        ("Severity: Critical, host isolated", "critical"),
        ("an alert of medium severity: noted", "medium"),
        ("a routine alert", None),
    ],
)
def test_parse_severity(prompt, expected):
    assert parse_severity(prompt) == expected


def test_brief_records_task_indicators_severity_and_request():
    brief = build_brief("triage: alert from 10.0.0.8, severity: high")
    assert "Task: triage" in brief
    assert "ipv4: 10.0.0.8" in brief
    assert "Stated severity: high" in brief
    brief = build_brief("analyse this config")
    assert "Task: analyse" in brief
    assert "Indicators: none extracted" in brief
    assert "Stated severity: not stated" in brief


@pytest.mark.parametrize(
    "prompt,marker",
    [
        ("analyse this log", "Cyber task: analyse"),
        ("triage: IDS alert", "Cyber task: triage"),
        ("report: finding in the VPN", "Cyber task: report"),
    ],
)
def test_cyber_guidance_and_brief_reach_the_provider(prompt, marker):
    provider = RecordingProvider()
    module = default_registry().instantiate("cyber")
    out = module.run(_ctx(provider), prompt)
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-shield"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert marker in messages[0]["content"]
    assert "# Cyber brief" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": prompt}


def test_cyber_runtime_dev_end_to_end_with_gravion_shield(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-shield"))
    assert runtime.resolve_model("cyber") == "gravion-shield"
    out = runtime.cyber("triage: alert from 10.0.0.8")
    assert "triage: alert from 10.0.0.8" in out


def test_cyber_runtime_rejects_model_without_cyber_capability(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.cyber("analyse this log")


def test_cyber_manifest_signs_and_verifies(issuer, cyber_module_copy):
    payload = build_manifest(
        cyber_module_copy,
        name="cyber",
        module_type="cyber",
        version="0.1.0",
        entrypoint="gravion_ai.modules.cyber.module:CyberModule",
    )
    write_signed_manifest(cyber_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(cyber_module_copy, issuer)
    assert verified["name"] == "cyber"
    assert verified["type"] == "cyber"
    assert "module.py" in verified["files"]


def test_bundled_cyber_manifest_is_unsigned(issuer):
    import gravion_ai.modules.cyber as cyber_pkg

    with pytest.raises(ModuleTamperError):
        verify_module(Path(cyber_pkg.__file__).parent, issuer)

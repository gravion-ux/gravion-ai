import json
from pathlib import Path

import pytest

from gravion_ai.cli import main


@pytest.fixture()
def home(tmp_path, monkeypatch):
    path = tmp_path / "deploy"
    monkeypatch.setenv("GRAVION_HOME", str(path))
    return path


def test_init_status_chat_flow(home, capsys):
    assert main(["init", "--name", "Gravion", "--company", "Acme Robotics"]) == 0
    assert (home / "gravion.json").exists()
    assert (home / "policy.md").exists()
    assert (home / "state" / "identity.key").exists()

    capsys.readouterr()
    assert main(["status"]) == 0
    out = capsys.readouterr().out
    assert "Gravion" in out and "development" in out

    assert main(["chat", "hello", "there"]) == 0
    out = capsys.readouterr().out
    assert "hello there" in out


def test_support_command_flow(home, capsys):
    assert main(["init", "--name", "Gravion", "--company", "Acme Robotics"]) == 0
    capsys.readouterr()
    assert main(["support", "where", "is", "my", "invoice?"]) == 0
    out = capsys.readouterr().out
    assert "where is my invoice?" in out
    # An explicit request for a human escalates without a model answer.
    assert main(["support", "can", "I", "talk", "to", "a", "human"]) == 0
    out = capsys.readouterr().out
    assert "human on our support team" in out


def test_coder_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-code"]) == 0
    capsys.readouterr()
    assert main(["coder", "write", "a", "sort", "function"]) == 0
    out = capsys.readouterr().out
    assert "write a sort function" in out


def test_image_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-vision"]) == 0
    capsys.readouterr()
    assert main(["image", "a", "square", "logo", "for", "our", "app"]) == 0
    out = capsys.readouterr().out
    assert "a square logo for our app" in out


def test_music_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-audio"]) == 0
    capsys.readouterr()
    assert main(["music", "a", "30", "second", "jingle"]) == 0
    out = capsys.readouterr().out
    assert "a 30 second jingle" in out


def test_video_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-video"]) == 0
    capsys.readouterr()
    assert main(["video", "a", "30", "second", "teaser"]) == 0
    out = capsys.readouterr().out
    assert "a 30 second teaser" in out


def test_website_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-builder"]) == 0
    capsys.readouterr()
    assert main(["website", "a", "website", "for", "our", "bakery"]) == 0
    out = capsys.readouterr().out
    assert "a website for our bakery" in out


def test_cyber_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-shield"]) == 0
    capsys.readouterr()
    assert main(["cyber", "triage:", "alert", "from", "10.0.0.8"]) == 0
    out = capsys.readouterr().out
    assert "triage: alert from 10.0.0.8" in out


def test_agent_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-agent"]) == 0
    capsys.readouterr()
    assert main(["agent", "plan", "an", "agent", "for", "reports"]) == 0
    out = capsys.readouterr().out
    assert "plan an agent for reports" in out


def test_agent_plan_out_writes_a_plan_file(home, capsys, tmp_path):
    assert main(["init", "--model", "gravion-agent"]) == 0
    capsys.readouterr()
    plan_path = tmp_path / "plan.json"
    assert main(["agent", "plan", "an", "agent", "for", "reports",
                 "--plan-out", str(plan_path)]) == 0
    out = capsys.readouterr().out
    assert "agent plan written to" in out
    assert "planned — not executed" in out
    assert plan_path.exists()


def test_omni_command_flow(home, capsys):
    assert main(["init", "--model", "gravion-omni"]) == 0
    capsys.readouterr()
    assert main(["omni", "coder:", "write", "a", "haiku", "function"]) == 0
    out = capsys.readouterr().out
    assert "write a haiku function" in out


def test_modules_list(home, capsys):
    assert main(["modules", "list"]) == 0
    out = capsys.readouterr().out
    assert "chat" in out and "video" in out and "implemented" in out


def test_license_identity_and_audit(home, capsys):
    assert main(["init"]) == 0
    capsys.readouterr()
    assert main(["license", "identity"]) == 0
    out = capsys.readouterr().out
    assert "gravion-dep-" in out
    assert main(["license", "audit", "verify"]) == 0
    assert "VALID" in capsys.readouterr().out


def test_full_licensing_roundtrip_via_cli(home, tmp_path, capsys):
    issuer_dir = tmp_path / "issuer"
    assert main(["license", "issuer-keygen", "--out", str(issuer_dir)]) == 0
    assert main(["init", "--mode", "production"]) == 0
    # discover the deployment id from the identity record
    record = json.loads((home / "state" / "identity.json").read_text())
    capsys.readouterr()
    license_path = tmp_path / "license.json"
    assert main([
        "license", "issue",
        "--issuer-key", str(issuer_dir / "issuer.key"),
        "--deployment-id", record["deployment_id"],
        "--holder", "Acme Robotics",
        "--tier", "tier-3-intermediate",
        "--modules", "chat,support",
        "--agreement-ref", "GRAVION-CA-2026-001",
        "--equity", "15",
        "--out", str(license_path),
    ]) == 0
    assert main([
        "license", "verify",
        "--license", str(license_path),
        "--issuer-pub", str(issuer_dir / "issuer.pub"),
    ]) == 0
    out = capsys.readouterr().out
    assert "VALID" in out and "Acme Robotics" in out
    assert main(["license", "install", "--license", str(license_path)]) == 0
    assert (home / "state" / "license.json").exists()


def test_export_command(home, tmp_path, capsys):
    assert main(["init"]) == 0
    out_zip = tmp_path / "bundle.zip"
    assert main(["export", "--out", str(out_zip)]) == 0
    assert out_zip.exists()
    assert "gravion.json" in capsys.readouterr().out


def test_chat_in_production_without_license_fails_cleanly(home, capsys):
    assert main(["init", "--mode", "production"]) == 0
    capsys.readouterr()
    assert main(["chat", "hello"]) == 1
    assert "suspended" in capsys.readouterr().err.lower()

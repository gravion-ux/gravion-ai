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


def test_modules_list(home, capsys):
    assert main(["modules", "list"]) == 0
    out = capsys.readouterr().out
    assert "chat" in out and "video" in out and "planned" in out


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

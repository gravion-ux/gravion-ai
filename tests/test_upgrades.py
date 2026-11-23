import json
from pathlib import Path

import pytest

from gravion_ai.canonical import canonical_json
from gravion_ai.cli import main
from gravion_ai.errors import UpgradeError
from gravion_ai.licensing.audit import AuditLog
from gravion_ai.licensing.keys import b64encode
from gravion_ai.registry import (
    ModuleManifest,
    ModuleRegistry,
    ModuleStatus,
    ModuleType,
)
from gravion_ai.upgrades import (
    UpgradeService,
    verify_update_package,
    write_update_package,
)


def _registry_for(module_dir: Path) -> ModuleRegistry:
    registry = ModuleRegistry()
    registry.register(
        ModuleManifest(
            name="chat",
            type=ModuleType.CHAT,
            version="0.1.0",
            entrypoint="gravion_ai.modules.chat.module:ChatModule",
            status=ModuleStatus.IMPLEMENTED,
            path=module_dir,
        )
    )
    return registry


def _service(make_config, issuer, module_dir, tmp_path, grant=True):
    config = make_config(mode="development")
    config.self_upgrade_permission = grant
    audit = AuditLog(tmp_path / "audit.jsonl")
    service = UpgradeService(
        config,
        registry=_registry_for(module_dir),
        audit=audit,
        issuer_public=issuer,
    )
    return service, audit


def _package(tmp_path, issuer, source_files: dict, **overrides):
    source = tmp_path / "src"
    for rel, content in source_files.items():
        path = source / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    kwargs = dict(
        module="chat",
        from_version="0.1.0",
        to_version="0.1.1",
        kind="bugfix",
        changelog="Fixed the greeting.",
        source_dir=source,
    )
    kwargs.update(overrides)
    package_dir = tmp_path / "pkg"
    write_update_package(package_dir, issuer, **kwargs)
    return package_dir


def _events(audit: AuditLog):
    return [
        json.loads(line)["event"]
        for line in audit.path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def test_apply_refused_without_permission(make_config, issuer, chat_module_copy, tmp_path):
    service, audit = _service(
        make_config, issuer, chat_module_copy, tmp_path, grant=False
    )
    package = _package(tmp_path, issuer, {"module.py": "# v2\n"})
    before = (chat_module_copy / "module.py").read_bytes()
    with pytest.raises(UpgradeError, match="permission is not granted"):
        service.apply(package)
    assert (chat_module_copy / "module.py").read_bytes() == before
    assert "upgrade.refused" in _events(audit)


def test_apply_refuses_a_tampered_package(make_config, issuer, chat_module_copy, tmp_path):
    service, audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    package = _package(tmp_path, issuer, {"module.py": "# v2\n"})
    (package / "files" / "module.py").write_text("# evil\n", encoding="utf-8")
    with pytest.raises(UpgradeError, match="does not match its signed hash"):
        service.apply(package)
    assert "upgrade.refused" in _events(audit)


def test_apply_refuses_an_unsigned_package(make_config, issuer, chat_module_copy, tmp_path):
    service, _audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    package = _package(tmp_path, issuer, {"module.py": "# v2\n"})
    document = json.loads((package / "update.json").read_text(encoding="utf-8"))
    document["signature"] = ""
    (package / "update.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(UpgradeError, match="unsigned"):
        service.apply(package)


def test_apply_refuses_out_of_scope_kind(make_config, issuer, chat_module_copy, tmp_path):
    """A validly signed package whose kind is not bugfix /
    enhancement is out of the governor's scope — refused."""
    service, _audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    package = _package(tmp_path, issuer, {"module.py": "# v2\n"})
    document = json.loads((package / "update.json").read_text(encoding="utf-8"))
    document["payload"]["kind"] = "licensing-change"
    document["signature"] = b64encode(
        issuer.sign(canonical_json(document["payload"]))
    )
    (package / "update.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(UpgradeError, match="bug fixes and enhancements only"):
        service.apply(package)


def test_apply_refuses_version_mismatch(make_config, issuer, chat_module_copy, tmp_path):
    service, _audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    package = _package(
        tmp_path, issuer, {"module.py": "# v2\n"}, from_version="9.9.9"
    )
    with pytest.raises(UpgradeError, match="installed version is 0.1.0"):
        service.apply(package)


def test_successful_apply_changelog_backup_and_audit(
    make_config, issuer, chat_module_copy, tmp_path
):
    service, audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    original = (chat_module_copy / "module.py").read_bytes()
    package = _package(
        tmp_path, issuer, {"module.py": "# chat module, patched\n"}
    )
    result = service.apply(package)
    assert result["from_version"] == "0.1.0"
    assert result["to_version"] == "0.1.1"
    # New content is live.
    assert (chat_module_copy / "module.py").read_text(
        encoding="utf-8"
    ) == "# chat module, patched\n"
    # The changelog is in the deployment, with the package's text.
    changelog = Path(result["changelog_path"])
    assert changelog.exists()
    assert "Fixed the greeting." in changelog.read_text(encoding="utf-8")
    # The previous version is retained, byte for byte.
    backup = Path(result["backup_path"])
    assert (backup / "module.py").read_bytes() == original
    # The audit event carries before/after hashes.
    lines = audit.path.read_text(encoding="utf-8").splitlines()
    applied = [json.loads(l) for l in lines if '"upgrade.applied"' in l]
    assert len(applied) == 1
    details = applied[0]["details"]
    assert details["before_hashes"]["module.py"] != details["after_hashes"]["module.py"]


def test_failed_self_check_rolls_back_automatically(
    make_config, issuer, chat_module_copy, tmp_path
):
    service, audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    original = (chat_module_copy / "module.py").read_bytes()
    package = _package(
        tmp_path, issuer, {"module.py": "def broken(:\n"}, to_version="0.1.1"
    )
    with pytest.raises(UpgradeError, match="rolled back|restored automatically"):
        service.apply(package)
    assert (chat_module_copy / "module.py").read_bytes() == original
    events = _events(audit)
    assert "upgrade.rollback" in events


def test_manual_rollback_restores_the_previous_version(
    make_config, issuer, chat_module_copy, tmp_path
):
    service, audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    original = (chat_module_copy / "module.py").read_bytes()
    package = _package(tmp_path, issuer, {"module.py": "# patched\n"})
    service.apply(package)
    assert (chat_module_copy / "module.py").read_bytes() != original
    result = service.rollback("chat")
    assert result["restored_version"] == "0.1.0"
    assert (chat_module_copy / "module.py").read_bytes() == original
    assert "upgrade.rollback" in _events(audit)


def test_rollback_without_a_backup_refuses(
    make_config, issuer, chat_module_copy, tmp_path
):
    service, _audit = _service(make_config, issuer, chat_module_copy, tmp_path)
    with pytest.raises(UpgradeError, match="no retained versions"):
        service.rollback("chat")


def test_grant_and_revoke_are_recorded(make_config, issuer, chat_module_copy, tmp_path):
    service, audit = _service(
        make_config, issuer, chat_module_copy, tmp_path, grant=False
    )
    assert service.status()["permission"] is False
    service.grant()
    assert service.status()["permission"] is True
    # Persisted to the config file, not just in memory.
    from gravion_ai.config import GravionConfig

    reloaded = GravionConfig.load(service._config.home)
    assert reloaded.self_upgrade_permission is True
    service.revoke()
    assert service.status()["permission"] is False
    events = _events(audit)
    assert "upgrade.permission_granted" in events
    assert "upgrade.permission_revoked" in events


def test_verify_update_package_round_trip(issuer, tmp_path):
    package = _package(tmp_path, issuer, {"module.py": "# v2\n"})
    payload = verify_update_package(package, issuer)
    assert payload["module"] == "chat"
    assert payload["kind"] == "bugfix"


@pytest.fixture()
def home(tmp_path, monkeypatch):
    path = tmp_path / "deploy"
    monkeypatch.setenv("GRAVION_HOME", str(path))
    return path


def test_upgrade_cli_grant_status_revoke(home, capsys):
    assert main(["init"]) == 0
    capsys.readouterr()
    assert main(["upgrade", "status"]) == 0
    assert "not granted" in capsys.readouterr().out
    assert main(["upgrade", "grant"]) == 0
    assert "GRANTED" in capsys.readouterr().out
    assert main(["upgrade", "status"]) == 0
    assert "granted" in capsys.readouterr().out
    assert main(["upgrade", "revoke"]) == 0
    capsys.readouterr()
    # Apply without a grant refuses cleanly (exit 1, nothing touched).
    assert main(["upgrade", "apply", "--package", str(home / "nope")]) == 1

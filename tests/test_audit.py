import json

import pytest

from gravion_ai.errors import AuditChainError
from gravion_ai.licensing import AuditLog


def _log(tmp_path) -> AuditLog:
    log = AuditLog(tmp_path / "audit.jsonl")
    log.append("deployment.init", {"name": "Gravion"})
    log.append("gate.allow", {"module": "chat"})
    log.append("gate.suspend", {"module": "chat", "reason": "expired"})
    return log


def test_chain_verifies(tmp_path):
    log = _log(tmp_path)
    assert log.verify() == 3
    assert log.head_hash() == log.entries()[-1]["hash"]


def test_edited_entry_detected(tmp_path):
    log = _log(tmp_path)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    entry = json.loads(lines[1])
    entry["details"]["module"] = "video"  # rewrite history
    lines[1] = json.dumps(entry, sort_keys=True)
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError):
        log.verify()


def test_deleted_entry_detected(tmp_path):
    log = _log(tmp_path)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    del lines[1]  # remove the middle entry
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError):
        log.verify()


def test_reordered_entries_detected(tmp_path):
    log = _log(tmp_path)
    lines = log.path.read_text(encoding="utf-8").splitlines()
    lines[0], lines[1] = lines[1], lines[0]
    log.path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError):
        log.verify()


def test_empty_log_verifies_as_zero(tmp_path):
    log = AuditLog(tmp_path / "missing.jsonl")
    assert log.verify() == 0

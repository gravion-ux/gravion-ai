"""Tamper-evident licensing audit log.

An append-only JSONL file where every entry is hash-chained to the
previous one:

    entry_hash = SHA-256( prev_hash || canonical(entry without hash) )

Editing, reordering or deleting any historical entry breaks the chain
at that point, and `verify()` reports exactly where. Both the
deployment and (on request) the licensor can verify the chain — it is
the shared, tamper-evident record of licensing events: identity
creation, license checks, gate decisions, module verification,
suspension, export.

The log contains licensing metadata only — never prompts, completions,
or user content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..canonical import canonical_json, sha256_hex
from ..errors import AuditChainError
from .timeutil import to_iso, utcnow

GENESIS_HASH = "0" * 64


def _entry_hash(prev_hash: str, entry_without_hash: dict) -> str:
    return sha256_hex(prev_hash.encode("ascii") + canonical_json(entry_without_hash))


class AuditLog:
    def __init__(self, path: Path):
        self.path = Path(path)

    # -- reading ---------------------------------------------------------
    def entries(self) -> list[dict]:
        if not self.path.exists():
            return []
        out = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                out.append(json.loads(line))
        return out

    def head_hash(self) -> str:
        entries = self.entries()
        return entries[-1]["hash"] if entries else GENESIS_HASH

    # -- appending ---------------------------------------------------------
    def append(self, event: str, details: dict[str, Any] | None = None) -> dict:
        entries = self.entries()
        prev_hash = entries[-1]["hash"] if entries else GENESIS_HASH
        entry = {
            "seq": len(entries) + 1,
            "ts": to_iso(utcnow()),
            "event": event,
            "details": details or {},
            "prev_hash": prev_hash,
        }
        entry["hash"] = _entry_hash(prev_hash, entry)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return entry

    # -- verification ------------------------------------------------------
    def verify(self) -> int:
        """Verify the whole chain. Returns the number of entries.

        Raises AuditChainError at the first sign of tampering.
        """
        prev_hash = GENESIS_HASH
        count = 0
        for expected_seq, entry in enumerate(self.entries(), start=1):
            count = expected_seq
            if entry.get("seq") != expected_seq:
                raise AuditChainError(
                    f"audit entry out of order or missing at position "
                    f"{expected_seq} (found seq={entry.get('seq')})"
                )
            if entry.get("prev_hash") != prev_hash:
                raise AuditChainError(
                    f"audit chain broken at seq {expected_seq}: prev_hash "
                    f"does not match the previous entry's hash"
                )
            body = {k: v for k, v in entry.items() if k != "hash"}
            if _entry_hash(prev_hash, body) != entry.get("hash"):
                raise AuditChainError(
                    f"audit entry seq {expected_seq} was modified after writing"
                )
            prev_hash = entry["hash"]
        return count

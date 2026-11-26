"""Persistent run/session event log foundation.

The licensing audit log records entitlement and security events. This
log is the operational companion: it records module runs so a
deployment can inspect what ran, when, with which module/model, and
whether it completed or failed. It is deliberately content-free by
default: prompts, completions, API keys and other secret-shaped
detail values are redacted before writing.
"""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .licensing.timeutil import to_iso, utcnow

EVENT_TYPES = frozenset(
    {
        "session.started",
        "session.completed",
        "run.started",
        "run.completed",
        "run.failed",
        "provider.retry",
        "provider.failed",
        "provider.completed",
    }
)
_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "content",
    "password",
    "prompt",
    "secret",
    "token",
)
_MAX_DETAIL_STRING = 500


def _sanitize(value: Any, *, key: str = "") -> Any:
    lowered = key.lower()
    if any(part in lowered for part in _SENSITIVE_KEY_PARTS):
        return "[redacted]"
    if isinstance(value, dict):
        return {str(k): _sanitize(v, key=str(k)) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    if isinstance(value, str):
        return value[:_MAX_DETAIL_STRING]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:_MAX_DETAIL_STRING]


@dataclass(frozen=True)
class RunEvent:
    seq: int
    ts: str
    event: str
    run_id: str
    session_id: str | None = None
    module: str | None = None
    model: str | None = None
    details: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "event": self.event,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "module": self.module,
            "model": self.model,
            "details": self.details or {},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RunEvent":
        return cls(
            seq=int(data.get("seq", 0)),
            ts=str(data.get("ts", "")),
            event=str(data.get("event", "")),
            run_id=str(data.get("run_id", "")),
            session_id=data.get("session_id"),
            module=data.get("module"),
            model=data.get("model"),
            details=dict(data.get("details") or {}),
        )


class RunEventLog:
    """Append-only JSONL log for session/run events."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()

    @staticmethod
    def new_run_id() -> str:
        return uuid.uuid4().hex

    @staticmethod
    def new_session_id() -> str:
        return uuid.uuid4().hex

    def entries(
        self,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        event: str | None = None,
    ) -> list[RunEvent]:
        if not self.path.exists():
            return []
        events: list[RunEvent] = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            item = RunEvent.from_dict(json.loads(line))
            if run_id is not None and item.run_id != run_id:
                continue
            if session_id is not None and item.session_id != session_id:
                continue
            if event is not None and item.event != event:
                continue
            events.append(item)
        return events

    def append(
        self,
        event: str,
        *,
        run_id: str | None = None,
        session_id: str | None = None,
        module: str | None = None,
        model: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> RunEvent:
        if event not in EVENT_TYPES:
            raise ValueError(
                f"unknown run event {event!r}; valid: {sorted(EVENT_TYPES)}"
            )
        with self._lock:
            existing = self.entries()
            item = RunEvent(
                seq=len(existing) + 1,
                ts=to_iso(utcnow()),
                event=event,
                run_id=run_id or self.new_run_id(),
                session_id=session_id,
                module=module,
                model=model,
                details=_sanitize(details or {}),
            )
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(item.to_dict(), sort_keys=True) + "\n")
            return item

    def count(self) -> int:
        return len(self.entries())

    def latest_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """Summarise the most recent runs by run_id, newest first."""
        grouped: dict[str, list[RunEvent]] = {}
        for item in self.entries():
            grouped.setdefault(item.run_id, []).append(item)
        summaries: list[dict[str, Any]] = []
        for run_id, items in grouped.items():
            first = items[0]
            last = items[-1]
            summaries.append(
                {
                    "run_id": run_id,
                    "session_id": first.session_id,
                    "module": first.module,
                    "model": first.model,
                    "started_at": first.ts,
                    "last_event": last.event,
                    "last_at": last.ts,
                    "event_count": len(items),
                }
            )
        summaries.sort(key=lambda row: row["last_at"], reverse=True)
        return summaries[: max(0, int(limit))]


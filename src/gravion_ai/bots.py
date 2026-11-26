"""Drop-in bot integration slot.

The operator may already have a bot of their own (for example one
living in their self chats). Gravion does not recreate, rewrite or
execute that bot. Instead, a deployment gets a clean slot:

    <home>/bots/
      bot.json          the slot manifest (written by the operator)
      ...               the operator's own bot files, unchanged

Only one adapter kind is wired today: **http**. The operator runs
their bot however it already runs and exposes a small HTTP endpoint;
Gravion posts {"messages": [...], "files": [...]} and expects
{"reply": "..."} (a chat-completions-shaped response is also
accepted). Dropped code is never executed by Gravion — `bot check`
validates the manifest only.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .config import GravionConfig, _validate_url
from .errors import ConfigError, GravionError

BOT_DIRNAME = "bots"
BOT_MANIFEST = "bot.json"
BOT_FIELDS = frozenset(
    {"name", "kind", "endpoint_url", "health_path", "timeout_seconds", "description"}
)
NEEDED_INPUTS = [
    "the bot code, link, or export (as-is; Gravion will not recreate it)",
    "how it runs today (language/runtime and the exact start command or endpoint)",
    "its input/output format (what a message looks like in, what a reply looks like out)",
    "names of any environment variables it needs (never paste secret values into chat)",
    "whether it can expose a local HTTP endpoint, or only runs inside a chat app",
]


class BotSlotError(GravionError):
    """The bot slot manifest is invalid or the bot call failed."""


@dataclass(frozen=True)
class BotSlot:
    name: str
    kind: str
    endpoint_url: str
    health_path: str = "/health"
    timeout_seconds: float = 60.0
    description: str = ""


def bot_dir(config: GravionConfig) -> Path:
    return config.home / BOT_DIRNAME


def validate_bot_manifest(data: object) -> BotSlot:
    if not isinstance(data, dict):
        raise BotSlotError("bot.json must contain a JSON object")
    unknown = sorted(set(data) - BOT_FIELDS)
    if unknown:
        raise BotSlotError(f"unknown bot.json fields: {', '.join(unknown)}")
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        raise BotSlotError("bot.json name must be a non-empty string")
    kind = data.get("kind", "http")
    if kind != "http":
        raise BotSlotError(
            "bot.json kind must be 'http' today; dropped code is never executed by Gravion"
        )
    try:
        endpoint_url = _validate_url(data.get("endpoint_url"), "bot endpoint_url")
    except ConfigError as exc:
        raise BotSlotError(str(exc)) from exc
    health_path = data.get("health_path", "/health")
    if not isinstance(health_path, str) or not health_path.startswith("/"):
        raise BotSlotError("bot.json health_path must start with '/'")
    timeout = data.get("timeout_seconds", 60.0)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= float(timeout) <= 300:
        raise BotSlotError("bot.json timeout_seconds must be between 1 and 300")
    description = data.get("description", "")
    if not isinstance(description, str):
        raise BotSlotError("bot.json description must be a string")
    return BotSlot(
        name=name.strip(),
        kind=kind,
        endpoint_url=endpoint_url,
        health_path=health_path,
        timeout_seconds=float(timeout),
        description=description,
    )


def load_bot_slot(config: GravionConfig) -> BotSlot | None:
    path = bot_dir(config) / BOT_MANIFEST
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise BotSlotError(f"cannot read bot manifest at {path}: {exc}") from exc
    return validate_bot_manifest(data)


def _slot_files(config: GravionConfig) -> list[str]:
    root = bot_dir(config)
    if not root.exists():
        return []
    return sorted(
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file()
    )


def bot_status(config: GravionConfig) -> dict:
    files = _slot_files(config)
    try:
        slot = load_bot_slot(config)
    except BotSlotError as exc:
        return {
            "configured": False,
            "files": files,
            "problems": [str(exc)],
            "needed": list(NEEDED_INPUTS),
        }
    if slot is None:
        return {
            "configured": False,
            "files": files,
            "problems": [],
            "needed": list(NEEDED_INPUTS),
        }
    return {
        "configured": True,
        "name": slot.name,
        "kind": slot.kind,
        "endpoint_url": slot.endpoint_url,
        "health_path": slot.health_path,
        "description": slot.description,
        "files": files,
        "problems": [],
        "needed": [],
    }


def check_bot_slot(config: GravionConfig) -> dict:
    """Validate the slot manifest without running any bot code."""
    return bot_status(config)


class HttpBotAdapter:
    """Call the operator's bot over HTTP; never executes bot code."""

    def __init__(self, slot: BotSlot, opener=None):
        self._slot = slot
        self._opener = opener or urllib.request.urlopen

    def reply(self, messages: Sequence[dict], files: Sequence[dict] | None = None) -> str:
        payload = {
            "messages": list(messages),
            "files": list(files or []),
        }
        request = urllib.request.Request(
            self._slot.endpoint_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Gravion-AI-BotSlot",
            },
            method="POST",
        )
        try:
            with self._opener(request, timeout=self._slot.timeout_seconds) as resp:
                raw = resp.read(4 * 1024 * 1024 + 1)
            if len(raw) > 4 * 1024 * 1024:
                raise BotSlotError("bot endpoint response exceeded the 4 MiB safety limit")
            body = json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise BotSlotError(f"bot endpoint returned HTTP {exc.code}") from exc
        except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise BotSlotError(f"bot endpoint call failed: {exc.__class__.__name__}") from exc
        if isinstance(body, dict) and isinstance(body.get("reply"), str):
            return body["reply"]
        try:
            content = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise BotSlotError("bot endpoint returned an unexpected response shape") from exc
        if not isinstance(content, str):
            raise BotSlotError("bot endpoint reply content must be a string")
        return content

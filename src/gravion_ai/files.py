"""File attachments for Gravion deployments.

The web UI accepts files as JSON payloads (the browser reads them
locally and sends text or base64). This module validates, stores and
extracts text from those attachments under the deployment's data
directory. It never executes uploaded content, never follows paths
outside the upload directory, and caps sizes before anything is
written.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from .config import GravionConfig
from .errors import GravionError

MAX_FILES = 8
MAX_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_BYTES = 8 * 1024 * 1024
MAX_CONTEXT_CHARS = 24_000
TEXT_SUFFIXES = frozenset(
    {
        ".txt",
        ".md",
        ".markdown",
        ".json",
        ".jsonl",
        ".csv",
        ".tsv",
        ".py",
        ".js",
        ".ts",
        ".html",
        ".css",
        ".xml",
        ".yaml",
        ".yml",
        ".toml",
        ".ini",
        ".log",
    }
)
_SAFE_NAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


class FileAttachmentError(GravionError):
    """An uploaded file payload is invalid or unsafe."""


@dataclass(frozen=True)
class StoredFile:
    name: str
    stored_name: str
    mime: str
    size: int
    sha256: str
    text_preview: str
    text_chars: int
    path: str

    def public_dict(self) -> dict:
        return {
            "name": self.name,
            "stored_name": self.stored_name,
            "mime": self.mime,
            "size": self.size,
            "sha256": self.sha256,
            "text_chars": self.text_chars,
            "path": self.path,
        }


def _safe_filename(name: object) -> str:
    if not isinstance(name, str) or not name.strip():
        raise FileAttachmentError("each file needs a non-empty name")
    base = Path(name.replace("\\", "/")).name.strip()
    safe = _SAFE_NAME_RE.sub("-", base).strip(".-")
    if not safe:
        safe = "attachment"
    return safe[:120]


def _decode_file(record: dict) -> tuple[str, str, bytes]:
    if not isinstance(record, dict):
        raise FileAttachmentError("each file must be an object")
    name = _safe_filename(record.get("name"))
    mime = record.get("mime", "application/octet-stream")
    if not isinstance(mime, str) or not mime.strip():
        raise FileAttachmentError(f"file {name!r}: mime must be a non-empty string")
    if "text" in record and record.get("text") is not None:
        text = record["text"]
        if not isinstance(text, str):
            raise FileAttachmentError(f"file {name!r}: text must be a string")
        data = text.encode("utf-8")
    elif "data_base64" in record and record.get("data_base64") is not None:
        encoded = record["data_base64"]
        if not isinstance(encoded, str):
            raise FileAttachmentError(f"file {name!r}: data_base64 must be a string")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise FileAttachmentError(f"file {name!r}: invalid base64 data") from exc
    else:
        raise FileAttachmentError(f"file {name!r}: provide text or data_base64")
    if len(data) > MAX_FILE_BYTES:
        raise FileAttachmentError(
            f"file {name!r} exceeds the {MAX_FILE_BYTES}-byte limit"
        )
    return name, mime.strip(), data


def _extract_text(name: str, mime: str, data: bytes) -> str:
    suffix = Path(name).suffix.lower()
    if suffix in TEXT_SUFFIXES or mime.startswith("text/") or mime in {
        "application/json",
        "application/xml",
    }:
        return data.decode("utf-8", errors="replace")
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return ""
    printable = sum(1 for ch in text if ch.isprintable() or ch in "\n\r\t")
    return text if text and printable / len(text) > 0.95 else ""


def store_uploads(config: GravionConfig, files: object) -> list[StoredFile]:
    """Validate and store uploaded files; return their metadata.

    Files land in data/uploads/<batch-id>/ with sanitized names and a
    manifest.json. Nothing is executed; text extraction is decoding
    only, truncated later when context is built.
    """
    if files is None:
        return []
    if not isinstance(files, list):
        raise FileAttachmentError("files must be a list")
    if len(files) > MAX_FILES:
        raise FileAttachmentError(f"at most {MAX_FILES} files per request")

    decoded: list[tuple[str, str, bytes]] = []
    total = 0
    for record in files:
        name, mime, data = _decode_file(record)
        total += len(data)
        if total > MAX_TOTAL_BYTES:
            raise FileAttachmentError(
                f"attachments exceed the {MAX_TOTAL_BYTES}-byte total limit"
            )
        decoded.append((name, mime, data))
    if not decoded:
        return []

    batch_id = uuid.uuid4().hex[:12]
    batch_dir = config.data_dir / "uploads" / batch_id
    batch_dir.mkdir(parents=True, exist_ok=True)
    stored: list[StoredFile] = []
    used_names: set[str] = set()
    for name, mime, data in decoded:
        stored_name = name
        counter = 2
        while stored_name in used_names:
            stem = Path(name).stem
            suffix = Path(name).suffix
            stored_name = f"{stem}-{counter}{suffix}"
            counter += 1
        used_names.add(stored_name)
        path = batch_dir / stored_name
        path.write_bytes(data)
        text = _extract_text(name, mime, data)
        stored.append(
            StoredFile(
                name=name,
                stored_name=stored_name,
                mime=mime,
                size=len(data),
                sha256=hashlib.sha256(data).hexdigest(),
                text_preview=text[:MAX_CONTEXT_CHARS],
                text_chars=len(text),
                path=str(Path("data") / "uploads" / batch_id / stored_name),
            )
        )
    manifest = {
        "batch_id": batch_id,
        "files": [item.public_dict() for item in stored],
    }
    (batch_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return stored


def build_file_context(files: list[StoredFile]) -> str:
    """Build the prompt context block for stored attachments."""
    if not files:
        return ""
    parts = ["# Attached files"]
    remaining = MAX_CONTEXT_CHARS
    for item in files:
        header = f"\n## {item.name} ({item.mime}, {item.size} bytes, sha256 {item.sha256[:12]})"
        parts.append(header)
        if item.text_preview and remaining > 0:
            excerpt = item.text_preview[:remaining]
            parts.append(excerpt)
            remaining -= len(excerpt)
            if item.text_chars > len(excerpt):
                parts.append("[file text truncated for context]")
        else:
            parts.append("[binary or non-text file stored; no text extracted]")
    return "\n".join(parts)


def list_uploads(config: GravionConfig) -> list[dict]:
    root = config.data_dir / "uploads"
    if not root.exists():
        return []
    uploads: list[dict] = []
    for manifest_path in sorted(root.glob("*/manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for item in manifest.get("files", []):
            if isinstance(item, dict):
                uploads.append({"batch_id": manifest.get("batch_id"), **item})
    return uploads

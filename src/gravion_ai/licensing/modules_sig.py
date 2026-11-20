"""Signed module verification.

Every shippable module carries a manifest — name, type, version and a
SHA-256 of every file in the module — signed by the issuer. In
production mode the loader verifies, before any module code runs:

1. the manifest signature (it really came from the issuer), and
2. every file hash (nothing was modified after signing).

A mismatch means the module is refused and a `module.tamper` event is
written to the audit log by the caller. This is what stops a swapped or
patched module from silently running under a customer's license.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from ..canonical import canonical_json
from ..errors import ModuleTamperError
from .keys import KeyPair, b64decode, b64encode

MANIFEST_FILENAME = "manifest.json"
_EXCLUDED_NAMES = {MANIFEST_FILENAME, "__pycache__"}


def _iter_module_files(module_dir: Path):
    for path in sorted(module_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(module_dir)
        if any(part in _EXCLUDED_NAMES for part in rel.parts):
            continue
        if path.suffix in (".pyc", ".pyo"):
            continue
        yield rel


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_manifest(
    module_dir: Path,
    *,
    name: str,
    module_type: str,
    version: str,
    entrypoint: str,
) -> dict:
    """Compute the (unsigned) manifest payload for a module directory."""
    module_dir = Path(module_dir)
    files = {
        rel.as_posix(): hash_file(module_dir / rel)
        for rel in _iter_module_files(module_dir)
    }
    return {
        "name": name,
        "type": module_type,
        "version": version,
        "entrypoint": entrypoint,
        "files": files,
    }


def sign_manifest(manifest_payload: dict, issuer: KeyPair) -> dict:
    signature = issuer.sign(canonical_json(manifest_payload))
    return {"payload": manifest_payload, "signature": b64encode(signature)}


def write_signed_manifest(module_dir: Path, signed: dict) -> Path:
    path = Path(module_dir) / MANIFEST_FILENAME
    path.write_text(json.dumps(signed, indent=2) + "\n", encoding="utf-8")
    return path


def load_signed_manifest(module_dir: Path) -> dict:
    path = Path(module_dir) / MANIFEST_FILENAME
    if not path.exists():
        raise ModuleTamperError(
            f"module at {module_dir} has no signed manifest ({MANIFEST_FILENAME})"
        )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ModuleTamperError(f"manifest in {module_dir} is not valid JSON: {exc}") from exc


def verify_module(module_dir: Path, issuer_public: KeyPair) -> dict:
    """Verify a module directory against its signed manifest.

    Returns the verified manifest payload. Raises ModuleTamperError on
    a bad signature, a missing/extra file, or any hash mismatch.
    """
    module_dir = Path(module_dir)
    signed = load_signed_manifest(module_dir)
    try:
        payload = signed["payload"]
        signature = b64decode(signed["signature"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ModuleTamperError(f"malformed signed manifest: {exc}") from exc
    try:
        issuer_public.verify(canonical_json(payload), signature)
    except Exception as exc:
        raise ModuleTamperError(f"manifest signature invalid: {exc}") from exc

    expected: dict = payload.get("files", {})
    actual = {
        rel.as_posix(): hash_file(module_dir / rel)
        for rel in _iter_module_files(module_dir)
    }
    if set(actual) != set(expected):
        missing = sorted(set(expected) - set(actual))
        extra = sorted(set(actual) - set(expected))
        raise ModuleTamperError(
            f"module file set changed (missing: {missing}, unexpected: {extra})"
        )
    changed = [name for name in expected if expected[name] != actual[name]]
    if changed:
        raise ModuleTamperError(f"module files modified after signing: {changed}")
    return payload

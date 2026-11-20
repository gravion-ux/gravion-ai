"""Canonical JSON + hashing helpers.

Every signature and hash chain in gravion-ai is computed over a canonical
JSON encoding (sorted keys, tight separators, UTF-8) so that the same
document always produces the same bytes — and therefore the same
signature or hash — regardless of who serialised it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> bytes:
    """Serialise *obj* to canonical JSON bytes."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def document_hash(obj: Any) -> str:
    """SHA-256 of the canonical encoding of a JSON document."""
    return sha256_hex(canonical_json(obj))

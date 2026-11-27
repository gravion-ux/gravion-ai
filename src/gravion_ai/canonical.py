"""Canonical JSON + hashing helpers.

Every signature and hash chain in gravion-ai is computed over a canonical
JSON encoding (sorted keys, tight separators, UTF-8) so that the same
document always produces the same bytes — and therefore the same
signature or hash — regardless of who serialised it.

File hashes are computed over canonical *bytes* for the same reason: a
file's hash must not depend on the line endings the checkout happens
to use.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
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


def canonical_bytes(data: bytes) -> bytes:
    """The canonical form of a file's bytes, for hashing.

    Git stores text with LF endings, but a Windows checkout with
    `core.autocrlf=true` rewrites the working tree to CRLF. Hashing the
    canonical form — CRLF normalised to LF — means a manifest signed
    over the repository's bytes still verifies on such a checkout, and
    a module signed on one platform still verifies on another.

    Binary content is returned unchanged: anything holding a NUL byte,
    or anything that is not valid UTF-8, is hashed exactly as it is.
    """
    if b"\x00" in data:
        return data
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        return data
    return text.replace("\r\n", "\n").encode("utf-8")


def sha256_file(path) -> str:
    """SHA-256 of a file's canonical bytes.

    The whole file is read at once: the canonical form of a text file
    is its UTF-8 text, and a chunk boundary can fall inside a
    multi-byte character.
    """
    return sha256_hex(canonical_bytes(Path(path).read_bytes()))

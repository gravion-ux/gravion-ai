"""Canonical bytes for hashing.

A file's hash must not depend on the line endings the checkout happens
to use. Git stores LF, but a Windows checkout with `core.autocrlf=true`
rewrites the working tree to CRLF — and a manifest signed over the LF
form still has to verify there. Binary content is never rewritten.
"""

import hashlib

from gravion_ai.canonical import canonical_bytes, sha256_file


def test_crlf_text_is_canonicalised_to_lf():
    lf = b"def f():\n    return 1\n"

    assert canonical_bytes(lf.replace(b"\n", b"\r\n")) == lf


def test_lf_text_is_unchanged():
    lf = b"def f():\n    return 1\n"

    assert canonical_bytes(lf) == lf


def test_binary_content_is_never_canonicalised():
    blob = b"\x89PNG\r\n\x1a\n\x00\x00\r\npayload"

    assert canonical_bytes(blob) == blob


def test_non_utf8_content_is_never_canonicalised():
    blob = b"\xff\xfe\r\n\xfa"

    assert canonical_bytes(blob) == blob


def test_lone_carriage_return_is_left_alone():
    assert canonical_bytes(b"a\rb\n") == b"a\rb\n"


def test_sha256_file_hashes_a_crlf_file_as_its_lf_form(tmp_path):
    lf = b"line one\nline two\n"
    path = tmp_path / "module.py"
    path.write_bytes(lf.replace(b"\n", b"\r\n"))

    assert sha256_file(path) == hashlib.sha256(lf).hexdigest()


def test_sha256_file_hashes_binary_content_exactly(tmp_path):
    blob = b"\x89PNG\r\n\x1a\n\x00\x00\r\npayload"
    path = tmp_path / "image.png"
    path.write_bytes(blob)

    assert sha256_file(path) == hashlib.sha256(blob).hexdigest()


def test_sha256_file_matches_the_hash_of_the_lf_bytes(tmp_path):
    lf = b"alpha\nbeta\n"
    path = tmp_path / "module.py"
    path.write_bytes(lf)

    assert sha256_file(path) == hashlib.sha256(lf).hexdigest()

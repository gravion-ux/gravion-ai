import pytest

from gravion_ai.errors import ModuleTamperError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)


def _sign(issuer, module_dir):
    payload = build_manifest(
        module_dir,
        name="chat",
        module_type="chat",
        version="0.1.0",
        entrypoint="gravion_ai.modules.chat.module:ChatModule",
    )
    write_signed_manifest(module_dir, sign_manifest(payload, issuer))


def test_signed_module_verifies(issuer, chat_module_copy):
    _sign(issuer, chat_module_copy)
    payload = verify_module(chat_module_copy, issuer)
    assert payload["name"] == "chat"
    assert "module.py" in payload["files"]


def test_tampered_module_file_rejected(issuer, chat_module_copy):
    _sign(issuer, chat_module_copy)
    target = chat_module_copy / "module.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# backdoor\n", encoding="utf-8")
    with pytest.raises(ModuleTamperError, match="modified"):
        verify_module(chat_module_copy, issuer)


def test_added_file_rejected(issuer, chat_module_copy):
    _sign(issuer, chat_module_copy)
    (chat_module_copy / "extra.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ModuleTamperError):
        verify_module(chat_module_copy, issuer)


def _rewrite_line_endings(module_dir, *, to_crlf: bool):
    """Rewrite every module file with LF or CRLF endings."""
    for path in sorted(module_dir.rglob("*.py")):
        data = path.read_bytes().replace(b"\r\n", b"\n")
        if to_crlf:
            data = data.replace(b"\n", b"\r\n")
        path.write_bytes(data)


def test_signed_module_verifies_after_a_crlf_rewrite(issuer, chat_module_copy):
    # The manifest is signed over the LF form, the way it is in the
    # repository; a Windows checkout rewrites the files to CRLF.
    _rewrite_line_endings(chat_module_copy, to_crlf=False)
    _sign(issuer, chat_module_copy)
    _rewrite_line_endings(chat_module_copy, to_crlf=True)

    payload = verify_module(chat_module_copy, issuer)

    assert payload["name"] == "chat"


def test_crlf_rewrite_with_changed_content_is_rejected(issuer, chat_module_copy):
    _rewrite_line_endings(chat_module_copy, to_crlf=False)
    _sign(issuer, chat_module_copy)
    target = chat_module_copy / "module.py"
    target.write_bytes(
        target.read_bytes().replace(b"\n", b"\r\n") + b"# backdoor\r\n"
    )

    with pytest.raises(ModuleTamperError, match="modified"):
        verify_module(chat_module_copy, issuer)


def test_unsigned_module_rejected(issuer, chat_module_copy):
    # The bundled manifest ships unsigned (empty signature).
    with pytest.raises(ModuleTamperError):
        verify_module(chat_module_copy, issuer)


def test_missing_manifest_rejected(issuer, tmp_path):
    (tmp_path / "module.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ModuleTamperError, match="no signed manifest"):
        verify_module(tmp_path, issuer)


def test_wrong_issuer_signature_rejected(issuer, chat_module_copy):
    from gravion_ai.licensing import KeyPair

    _sign(issuer, chat_module_copy)
    with pytest.raises(ModuleTamperError):
        verify_module(chat_module_copy, KeyPair.generate())

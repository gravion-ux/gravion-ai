"""Ed25519 keys — the one cryptographic primitive licensing is built on.

Why Ed25519: small keys, fast verification, deterministic signatures,
and first-class support in the `cryptography` library. The same
:class:`KeyPair` type serves both roles in the system:

* the **issuer** key (held by the project owner / licensing server),
  which signs licenses, module manifests and entitlement tokens;
* the **deployment** key (generated per deployment, see identity.py),
  whose public half *is* the deployment's identity.

The Signer/Verifier surface is intentionally tiny (sign / verify /
public bytes) so a different algorithm could be dropped in behind it
without touching callers.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)

from ..canonical import sha256_hex
from ..errors import BadSignatureError, LicensingError


def b64encode(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def b64decode(text: str) -> bytes:
    return base64.b64decode(text.encode("ascii"))


class KeyPair:
    """An Ed25519 keypair (or, when built from public bytes only, a
    verify-only key)."""

    def __init__(self, private_key: Ed25519PrivateKey | None,
                 public_key: Ed25519PublicKey):
        self._private = private_key
        self._public = public_key

    # -- construction --------------------------------------------------
    @classmethod
    def generate(cls) -> "KeyPair":
        private = Ed25519PrivateKey.generate()
        return cls(private, private.public_key())

    @classmethod
    def from_private_pem(cls, pem: bytes) -> "KeyPair":
        private = serialization.load_pem_private_key(pem, password=None)
        if not isinstance(private, Ed25519PrivateKey):
            raise LicensingError("PEM does not contain an Ed25519 private key")
        return cls(private, private.public_key())

    @classmethod
    def from_public_pem(cls, pem: bytes) -> "KeyPair":
        public = serialization.load_pem_public_key(pem)
        if not isinstance(public, Ed25519PublicKey):
            raise LicensingError("PEM does not contain an Ed25519 public key")
        return cls(None, public)

    @classmethod
    def from_public_bytes(cls, raw: bytes) -> "KeyPair":
        return cls(None, Ed25519PublicKey.from_public_bytes(raw))

    # -- serialisation ---------------------------------------------------
    def private_pem(self) -> bytes:
        if self._private is None:
            raise LicensingError("this key is verify-only; no private half")
        return self._private.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )

    def public_pem(self) -> bytes:
        return self._public.public_bytes(
            serialization.Encoding.PEM,
            serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def public_bytes(self) -> bytes:
        return self._public.public_bytes(
            serialization.Encoding.Raw, serialization.PublicFormat.Raw
        )

    @property
    def key_id(self) -> str:
        """Stable fingerprint of the public key (SHA-256, hex)."""
        return sha256_hex(self.public_bytes())

    # -- sign / verify ---------------------------------------------------
    def sign(self, data: bytes) -> bytes:
        if self._private is None:
            raise LicensingError("this key is verify-only; cannot sign")
        return self._private.sign(data)

    def verify(self, data: bytes, signature: bytes) -> None:
        """Raise BadSignatureError unless *signature* covers *data*."""
        try:
            self._public.verify(signature, data)
        except InvalidSignature as exc:
            raise BadSignatureError("signature verification failed") from exc

    # -- persistence -----------------------------------------------------
    def save_private(self, path: Path) -> Path:
        """Write the private key with owner-only permissions (0600).

        Private keys are only ever written to local disk by this method;
        nothing in gravion-ai transmits them anywhere.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.write(fd, self.private_pem())
        finally:
            os.close(fd)
        os.chmod(path, 0o600)
        return path

    def save_public(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(self.public_pem())
        return path


def generate_keypair() -> KeyPair:
    return KeyPair.generate()

"""Cryptographic deployment identity.

Every production deployment generates its own Ed25519 keypair. The
public key's fingerprint is the **deployment ID** — the identity the
licensing server knows the deployment by. The private key:

* is generated on the deployment machine,
* is stored locally with 0600 permissions,
* is **never transmitted** — not to the licensing server, not anywhere.

Verification requests (see authority.py) carry only the deployment ID
and a hash of the license, so identity never drags user data with it.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import IdentityError
from .keys import KeyPair

PRIVATE_FILENAME = "identity.key"
PUBLIC_FILENAME = "identity.pub"
RECORD_FILENAME = "identity.json"


class DeploymentIdentity:
    def __init__(self, keypair: KeyPair, state_dir: Path):
        self._keypair = keypair
        self.state_dir = Path(state_dir)

    @property
    def deployment_id(self) -> str:
        return "gravion-dep-" + self._keypair.key_id[:40]

    @property
    def public_key_pem(self) -> bytes:
        return self._keypair.public_pem()

    def sign(self, data: bytes) -> bytes:
        """Sign with the deployment key (e.g. future challenge-response
        authentication to the licensing authority)."""
        return self._keypair.sign(data)

    # -- persistence -----------------------------------------------------
    @classmethod
    def create_or_load(cls, state_dir: Path) -> "DeploymentIdentity":
        state_dir = Path(state_dir)
        private_path = state_dir / PRIVATE_FILENAME
        if private_path.exists():
            return cls.load(state_dir)
        keypair = KeyPair.generate()
        keypair.save_private(private_path)
        keypair.save_public(state_dir / PUBLIC_FILENAME)
        identity = cls(keypair, state_dir)
        (state_dir / RECORD_FILENAME).write_text(
            json.dumps(
                {
                    "deployment_id": identity.deployment_id,
                    "public_key_fingerprint": keypair.key_id,
                    "algorithm": "ed25519",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return identity

    @classmethod
    def load(cls, state_dir: Path) -> "DeploymentIdentity":
        state_dir = Path(state_dir)
        private_path = state_dir / PRIVATE_FILENAME
        if not private_path.exists():
            raise IdentityError(
                f"no deployment identity in {state_dir}; "
                f"run `gravion license identity` first"
            )
        try:
            keypair = KeyPair.from_private_pem(private_path.read_bytes())
        except Exception as exc:  # corrupt key file
            raise IdentityError(f"cannot load identity key: {exc}") from exc
        return cls(keypair, state_dir)

    @classmethod
    def exists(cls, state_dir: Path) -> bool:
        return (Path(state_dir) / PRIVATE_FILENAME).exists()

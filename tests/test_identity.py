import os
import stat

import pytest
from cryptography.hazmat.primitives import serialization

from gravion_ai.licensing import DeploymentIdentity


def test_identity_created_with_deployment_id(tmp_path):
    identity = DeploymentIdentity.create_or_load(tmp_path)
    assert identity.deployment_id.startswith("gravion-dep-")
    assert (tmp_path / "identity.key").exists()
    assert (tmp_path / "identity.pub").exists()
    assert (tmp_path / "identity.json").exists()


def test_identity_persists_across_loads(tmp_path):
    first = DeploymentIdentity.create_or_load(tmp_path)
    second = DeploymentIdentity.create_or_load(tmp_path)
    assert first.deployment_id == second.deployment_id


# Windows has no POSIX permission bits: os.chmod() only toggles the read-only
# attribute there, so st_mode reports the Windows default rather than 0600.
# The owner-only guarantee is a POSIX property, so the bit-level assertion is
# made where the operating system can express it.  The file itself is still
# checked everywhere.
requires_posix_permissions = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission bits are not available on Windows"
)


@requires_posix_permissions
def test_private_key_permissions_are_owner_only(tmp_path):
    DeploymentIdentity.create_or_load(tmp_path)
    mode = stat.S_IMODE(os.stat(tmp_path / "identity.key").st_mode)
    assert mode == 0o600


def test_private_key_is_written_and_loadable(tmp_path):
    identity = DeploymentIdentity.create_or_load(tmp_path)
    key = tmp_path / "identity.key"
    assert key.exists()
    assert key.stat().st_size > 0

    pem = key.read_bytes()
    assert b"PRIVATE KEY" in pem

    # The file on disk is the private key the identity actually signs with.
    private = serialization.load_pem_private_key(pem, password=None)
    public = private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    assert public == identity.public_key_pem


def test_identity_signs_and_public_key_verifies(tmp_path):
    from gravion_ai.licensing import KeyPair

    identity = DeploymentIdentity.create_or_load(tmp_path)
    signature = identity.sign(b"challenge")
    public = KeyPair.from_public_pem(identity.public_key_pem)
    public.verify(b"challenge", signature)  # does not raise

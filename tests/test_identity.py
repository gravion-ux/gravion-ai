import os
import stat
import sys

import pytest

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


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Windows does not apply POSIX permission bits, so os.chmod "
    "cannot produce mode 0o600",
)
def test_private_key_permissions_are_owner_only(tmp_path):
    DeploymentIdentity.create_or_load(tmp_path)
    mode = stat.S_IMODE(os.stat(tmp_path / "identity.key").st_mode)
    assert mode == 0o600


def test_identity_signs_and_public_key_verifies(tmp_path):
    from gravion_ai.licensing import KeyPair

    identity = DeploymentIdentity.create_or_load(tmp_path)
    signature = identity.sign(b"challenge")
    public = KeyPair.from_public_pem(identity.public_key_pem)
    public.verify(b"challenge", signature)  # does not raise

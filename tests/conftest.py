"""Shared fixtures: an issuer keypair and deployment factories."""

from __future__ import annotations

from pathlib import Path

import pytest

from gravion_ai.config import GravionConfig, ProviderConfig
from gravion_ai.licensing import DeploymentIdentity, KeyPair, issue_license
from gravion_ai.licensing.gate import LICENSE_FILENAME
from gravion_ai.licensing.license_doc import save_license


@pytest.fixture()
def issuer() -> KeyPair:
    return KeyPair.generate()


@pytest.fixture()
def make_config(tmp_path):
    """Factory: a saved GravionConfig in a fresh home under tmp_path."""

    def _make(name="Gravion", mode="development", tier="tier-2-basic",
              provider_type="gravion-core", model="gravion-core",
              company="Acme Robotics") -> GravionConfig:
        home = tmp_path / f"home-{name}-{mode}-{provider_type}-{model}"
        config = GravionConfig(
            name=name,
            company=company,
            mode=mode,
            tier=tier,
            provider=ProviderConfig(type=provider_type, model=model),
            home=home,
        )
        config.save()
        config.policy_path.write_text(
            "# Policy\n- Be helpful.\n- Never share internal pricing.\n",
            encoding="utf-8",
        )
        return config

    return _make


@pytest.fixture()
def licensed_home(make_config, issuer):
    """Factory: a production deployment with identity + installed,
    authority-registered license. Returns (config, authority, document).
    """
    from gravion_ai.licensing import MockAuthority

    def _make(modules=("chat",), days=365, tier="tier-3-intermediate",
              model="gravion-core"):
        config = make_config(mode="production", tier=tier, model=model)
        identity = DeploymentIdentity.create_or_load(config.state_dir)
        document = issue_license(
            issuer,
            deployment_id=identity.deployment_id,
            holder="Acme Robotics",
            tier=tier,
            modules=list(modules),
            agreement_reference="GRAVION-CA-2026-001",
            equity_percent=15,
            days=days,
        )
        save_license(document, config.state_dir / LICENSE_FILENAME)
        # The deployment trusts the issuer key as its authority key.
        issuer.save_public(config.state_dir / "issuer.pub")
        authority = MockAuthority(issuer)
        authority.register_license(document)
        return config, authority, document

    return _make


def _copy_module(tmp_path: Path, package, dirname: str) -> Path:
    """A writable copy of a bundled module directory."""
    import shutil

    src = Path(package.__file__).parent
    dst = tmp_path / dirname
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("__pycache__"))
    return dst


@pytest.fixture()
def chat_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled chat module directory."""
    import gravion_ai.modules.chat as chat_pkg

    return _copy_module(tmp_path, chat_pkg, "chat-module")


@pytest.fixture()
def support_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled support module directory."""
    import gravion_ai.modules.support as support_pkg

    return _copy_module(tmp_path, support_pkg, "support-module")


@pytest.fixture()
def coder_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled coder module directory."""
    import gravion_ai.modules.coder as coder_pkg

    return _copy_module(tmp_path, coder_pkg, "coder-module")

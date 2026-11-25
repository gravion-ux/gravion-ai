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


@pytest.fixture()
def image_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled image module directory."""
    import gravion_ai.modules.image as image_pkg

    return _copy_module(tmp_path, image_pkg, "image-module")


@pytest.fixture()
def music_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled music module directory."""
    import gravion_ai.modules.music as music_pkg

    return _copy_module(tmp_path, music_pkg, "music-module")


@pytest.fixture()
def video_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled video module directory."""
    import gravion_ai.modules.video as video_pkg

    return _copy_module(tmp_path, video_pkg, "video-module")


@pytest.fixture()
def website_builder_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled website-builder module directory."""
    import gravion_ai.modules.website_builder as website_builder_pkg

    return _copy_module(tmp_path, website_builder_pkg, "website-builder-module")


@pytest.fixture()
def cyber_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled cyber module directory."""
    import gravion_ai.modules.cyber as cyber_pkg

    return _copy_module(tmp_path, cyber_pkg, "cyber-module")


@pytest.fixture()
def agentic_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled agentic module directory."""
    import gravion_ai.modules.agentic as agentic_pkg

    return _copy_module(tmp_path, agentic_pkg, "agentic-module")


@pytest.fixture()
def all_in_one_module_copy(tmp_path) -> Path:
    """A writable copy of the bundled all-in-one module directory."""
    import gravion_ai.modules.all_in_one as all_in_one_pkg

    return _copy_module(tmp_path, all_in_one_pkg, "all-in-one-module")

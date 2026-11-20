import pytest

from gravion_ai.config import GravionConfig
from gravion_ai.errors import ConfigError


def test_defaults_are_gravion_development(make_config):
    config = make_config()
    assert config.name == "Gravion"
    assert config.mode == "development"
    assert config.provider.type == "gravion-core"
    assert config.provider.model == "gravion-core"


def test_save_load_roundtrip(make_config, tmp_path):
    config = make_config(name="Helper", tier="tier-1-complete")
    loaded = GravionConfig.load(config.home)
    assert loaded.name == "Helper"
    assert loaded.tier == "tier-1-complete"
    assert loaded.company == "Acme Robotics"


def test_system_context_injects_name_company_and_policy(make_config):
    config = make_config(name="Helper")
    context = config.system_context()
    assert "Helper" in context
    assert "Acme Robotics" in context
    assert "Never share internal pricing" in context


def test_invalid_mode_rejected(tmp_path):
    config = GravionConfig(mode="staging", home=tmp_path / "h")
    with pytest.raises(ConfigError):
        config.validate()


def test_invalid_provider_type_rejected(tmp_path):
    from gravion_ai.config import ProviderConfig

    config = GravionConfig(
        provider=ProviderConfig(type="carrier-pigeon"), home=tmp_path / "h"
    )
    with pytest.raises(ConfigError):
        config.validate()


def test_load_missing_deployment_raises(tmp_path):
    with pytest.raises(ConfigError):
        GravionConfig.load(tmp_path / "nope")

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


def test_unknown_tier_is_config_error(tmp_path):
    config = GravionConfig(tier="tier-99", home=tmp_path / "h")
    with pytest.raises(ConfigError):
        config.validate()


def test_endpoint_provider_requires_valid_base_url(tmp_path):
    from gravion_ai.config import ProviderConfig

    config = GravionConfig(
        provider=ProviderConfig(type="endpoint", base_url=""), home=tmp_path / "h"
    )
    with pytest.raises(ConfigError):
        config.validate()


def test_production_rejects_stored_api_key_and_remote_http(tmp_path):
    from gravion_ai.config import ProviderConfig

    stored_key = GravionConfig(
        mode="production",
        provider=ProviderConfig(
            type="endpoint",
            base_url="https://endpoint.example/v1",
            api_key="secret",
        ),
        home=tmp_path / "h",
    )
    with pytest.raises(ConfigError):
        stored_key.validate()

    remote_http = GravionConfig(
        mode="production",
        provider=ProviderConfig(
            type="endpoint", base_url="http://endpoint.example/v1"
        ),
        home=tmp_path / "h",
    )
    with pytest.raises(ConfigError):
        remote_http.validate()


def test_from_dict_rejects_unknown_fields(tmp_path):
    with pytest.raises(ConfigError):
        GravionConfig.from_dict({"surprise": True}, tmp_path / "h")
    with pytest.raises(ConfigError):
        GravionConfig.from_dict({"provider": {"surprise": True}}, tmp_path / "h")


def test_provider_retry_and_generation_fields_validate(tmp_path):
    from gravion_ai.config import ProviderConfig

    config = GravionConfig.from_dict(
        {
            "provider": {
                "type": "endpoint",
                "base_url": "http://localhost:9999/v1",
                "max_retries": 3,
                "retry_backoff_seconds": 1.5,
                "temperature": 0.2,
                "max_tokens": 256,
            }
        },
        tmp_path / "h",
    )
    assert config.provider.max_retries == 3
    assert config.provider.retry_backoff_seconds == 1.5
    assert config.provider.temperature == 0.2
    assert config.provider.max_tokens == 256

    for bad in (
        ProviderConfig(max_retries=6),
        ProviderConfig(retry_backoff_seconds=31),
        ProviderConfig(temperature=2.1),
        ProviderConfig(max_tokens=0),
    ):
        with pytest.raises(ConfigError):
            GravionConfig(provider=bad, home=tmp_path / "h").validate()

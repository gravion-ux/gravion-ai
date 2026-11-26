import zipfile

import pytest

from gravion_ai.errors import ModuleError, SuspendedError
from gravion_ai.licensing import MockAuthorityClient
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    write_signed_manifest,
)
from gravion_ai.registry import ModuleManifest, ModuleRegistry, ModuleStatus, ModuleType
from gravion_ai.runtime import GravionRuntime


def test_dev_chat_end_to_end(make_config):
    runtime = GravionRuntime(make_config(mode="development"))
    out = runtime.chat("what can you do?")
    assert "what can you do?" in out


def test_runtime_records_persistent_run_events_without_prompt(make_config):
    config = make_config(mode="development")
    runtime = GravionRuntime(config)
    runtime.chat("private prompt text")
    events = runtime.events.entries()
    assert [e.event for e in events] == ["run.started", "run.completed"]
    assert events[0].run_id == events[1].run_id
    assert events[0].details["input_chars"] == len("private prompt text")
    raw = (config.state_dir / "run-events.jsonl").read_text(encoding="utf-8")
    assert "private prompt text" not in raw
    assert runtime.status()["run_events"] == 2


def test_runtime_records_failed_run_event(make_config):
    runtime = GravionRuntime(make_config(mode="production"))
    with pytest.raises(SuspendedError):
        runtime.chat("hello")
    events = runtime.events.entries()
    assert [e.event for e in events] == ["run.started", "run.failed"]
    assert events[-1].details["error_type"] == "SuspendedError"


def test_chat_resolves_model_through_registry(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    assert runtime.resolve_model("chat") == "gravion-core"


def test_chat_rejects_model_without_chat_capability(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-vision"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.chat("hello")


def test_production_chat_suspended_without_license(make_config):
    runtime = GravionRuntime(make_config(mode="production"))
    with pytest.raises(SuspendedError):
        runtime.chat("hello")


def test_production_chat_runs_with_license_and_signed_module(
    issuer, licensed_home, chat_module_copy
):
    config, authority, _ = licensed_home()
    payload = build_manifest(
        chat_module_copy,
        name="chat",
        module_type="chat",
        version="0.1.0",
        entrypoint="gravion_ai.modules.chat.module:ChatModule",
    )
    write_signed_manifest(chat_module_copy, sign_manifest(payload, issuer))
    registry = ModuleRegistry()
    registry.register(
        ModuleManifest(
            name="chat",
            type=ModuleType.CHAT,
            version="0.1.0",
            entrypoint="gravion_ai.modules.chat.module:ChatModule",
            status=ModuleStatus.IMPLEMENTED,
            path=chat_module_copy,
        )
    )
    runtime = GravionRuntime(
        config,
        issuer_public=issuer,
        client=MockAuthorityClient(authority),
        registry=registry,
    )
    out = runtime.chat("production hello")
    assert "production hello" in out


def test_production_chat_refuses_tampered_module(
    issuer, licensed_home, chat_module_copy
):
    from gravion_ai.errors import ModuleTamperError

    config, authority, _ = licensed_home()
    payload = build_manifest(
        chat_module_copy,
        name="chat",
        module_type="chat",
        version="0.1.0",
        entrypoint="gravion_ai.modules.chat.module:ChatModule",
    )
    write_signed_manifest(chat_module_copy, sign_manifest(payload, issuer))
    target = chat_module_copy / "module.py"
    target.write_text(target.read_text(encoding="utf-8") + "\n# patched\n", encoding="utf-8")
    registry = ModuleRegistry()
    registry.register(
        ModuleManifest(
            name="chat",
            type=ModuleType.CHAT,
            version="0.1.0",
            entrypoint="gravion_ai.modules.chat.module:ChatModule",
            status=ModuleStatus.IMPLEMENTED,
            path=chat_module_copy,
        )
    )
    runtime = GravionRuntime(
        config,
        issuer_public=issuer,
        client=MockAuthorityClient(authority),
        registry=registry,
    )
    with pytest.raises(ModuleTamperError):
        runtime.chat("hello")
    events = [e["event"] for e in runtime.audit.entries()]
    assert "module.tamper" in events


def test_export_works_while_suspended_and_contains_artefacts(make_config, tmp_path):
    config = make_config(mode="production")  # no license -> suspended
    (config.data_dir / "notes.txt").write_text("customer data", encoding="utf-8")
    runtime = GravionRuntime(config)
    out = tmp_path / "export.zip"
    included = runtime.export_bundle(out)
    assert "gravion.json" in included
    assert "modules.json" in included
    assert "models.json" in included
    assert "state/audit.jsonl" in included
    assert "data/notes.txt" in included
    with zipfile.ZipFile(out) as archive:
        names = archive.namelist()
    assert not any("identity.key" in n for n in names), "private key must never export"
    assert any(n == "data/notes.txt" for n in names)


def test_export_includes_license_and_public_identity(issuer, licensed_home, tmp_path):
    config, _, _ = licensed_home()
    runtime = GravionRuntime(config)
    out = tmp_path / "export2.zip"
    included = runtime.export_bundle(out)
    assert "state/license.json" in included
    assert "state/identity.json" in included
    assert not any("identity.key" in n for n in included)


def test_runtime_rejects_empty_prompt_before_running(make_config):
    # Regression: runtime.chat("") used to return the echo fallback
    # instead of a clear error (the web layer already refused these).
    runtime = GravionRuntime(make_config(mode="development"))
    for prompt in ("", "   ", "\n\n"):
        with pytest.raises(ModuleError, match="non-empty string"):
            runtime.chat(prompt)
    with pytest.raises(ModuleError, match="non-empty string"):
        runtime.run_module("chat", None)  # type: ignore[arg-type]
    # No run was recorded for the refused prompts.
    assert runtime.events.entries() == []

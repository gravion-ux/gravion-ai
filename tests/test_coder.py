import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError
from gravion_ai.licensing import MockAuthorityClient
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.coder.module import (
    TASK_EXPLAIN,
    TASK_GENERATE,
    TASK_REVIEW,
    detect_task,
)
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import (
    ModuleContext,
    ModuleManifest,
    ModuleRegistry,
    ModuleStatus,
    ModuleType,
    default_registry,
)
from gravion_ai.runtime import GravionRuntime


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None):
        self.calls.append((list(messages), model))
        return "recorded reply"


def _ctx(provider, model="gravion-code"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def test_coder_instantiates_and_answers():
    module = default_registry().instantiate("coder")
    out = module.run(_ctx(GravionCoreProvider()), "write a sort function")
    assert "sorting method" in out


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("review: def f(x): return x/0", TASK_REVIEW),
        ("Review this code for bugs", TASK_REVIEW),
        ("please review the attached module", TASK_REVIEW),
        ("do a code review of this function", TASK_REVIEW),
        ("explain: what does this regex do?", TASK_EXPLAIN),
        ("Explain this code to me", TASK_EXPLAIN),
        ("please explain the caching layer", TASK_EXPLAIN),
        ("write a function that retries with backoff", TASK_GENERATE),
        ("add a login form to the settings page", TASK_GENERATE),
        ("fix the off-by-one in the parser", TASK_GENERATE),
        # Natural phrasings that used to fall through to generate:
        ("can you explain this code?", TASK_EXPLAIN),
        ("what does this code do?", TASK_EXPLAIN),
        ("how does this function work?", TASK_EXPLAIN),
        ("could you review my code?", TASK_REVIEW),
        ("check this code for bugs", TASK_REVIEW),
        # Generation openers win over later explain/review words:
        ("write code to explain how sorting works", TASK_GENERATE),
        ("write a function that reviews log files", TASK_GENERATE),
        ("add a check to this function", TASK_GENERATE),
        # Regression: the "code review" noun phrase used to win
        # over the opening verb, so building or explaining a
        # review tool was misread as asking for a review.
        ("write a code review tool", TASK_GENERATE),
        ("explain code review process", TASK_EXPLAIN),
        ("code review: def f(x): return x", TASK_REVIEW),
        # Regression: the bare "code review" phrase used to win
        # over an earlier explanation verb, so asking what the
        # process is routed to a review. The signal stated
        # first governs.
        ("can you explain the code review process to me", TASK_EXPLAIN),
        ("please do a code review and explain the issues", TASK_REVIEW),
        ("reviewing my code would help", TASK_REVIEW),
    ],
)
def test_detect_task(prompt, expected):
    assert detect_task(prompt) == expected


@pytest.mark.parametrize(
    "prompt,marker",
    [
        ("write a retry helper", "Coder mode: generate"),
        ("review: x = 1", "Coder mode: review"),
        ("explain this code", "Coder mode: explain"),
    ],
)
def test_coder_guidance_follows_the_task(prompt, marker):
    provider = RecordingProvider()
    module = default_registry().instantiate("coder")
    out = module.run(_ctx(provider), prompt)
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-code"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert marker in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": prompt}


def test_coder_runtime_dev_end_to_end_with_gravion_code(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-code"))
    assert runtime.resolve_model("coder") == "gravion-code"
    out = runtime.coder("write a debounce helper")
    assert "quiet period" in out


def test_coder_runtime_rejects_model_without_coder_capability(make_config):
    # Gravion Core serves chat + support, not coder: the registry
    # check must refuse rather than answer with the wrong model.
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.coder("write a debounce helper")


def test_coder_manifest_signs_and_verifies(issuer, coder_module_copy):
    payload = build_manifest(
        coder_module_copy,
        name="coder",
        module_type="coder",
        version="0.1.0",
        entrypoint="gravion_ai.modules.coder.module:CoderModule",
    )
    write_signed_manifest(coder_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(coder_module_copy, issuer)
    assert verified["name"] == "coder"
    assert verified["type"] == "coder"
    assert "module.py" in verified["files"]


def test_bundled_coder_manifest_is_unsigned(issuer):
    import gravion_ai.modules.coder as coder_pkg
    from pathlib import Path

    with pytest.raises(ModuleTamperError):
        verify_module(Path(coder_pkg.__file__).parent, issuer)


def test_coder_production_runs_with_license_and_signed_module(
    issuer, licensed_home, coder_module_copy
):
    config, authority, _ = licensed_home(modules=("coder",), model="gravion-code")
    payload = build_manifest(
        coder_module_copy,
        name="coder",
        module_type="coder",
        version="0.1.0",
        entrypoint="gravion_ai.modules.coder.module:CoderModule",
    )
    write_signed_manifest(coder_module_copy, sign_manifest(payload, issuer))
    registry = ModuleRegistry()
    registry.register(
        ModuleManifest(
            name="coder",
            type=ModuleType.CODER,
            version="0.1.0",
            entrypoint="gravion_ai.modules.coder.module:CoderModule",
            status=ModuleStatus.IMPLEMENTED,
            path=coder_module_copy,
        )
    )
    runtime = GravionRuntime(
        config,
        issuer_public=issuer,
        client=MockAuthorityClient(authority),
        registry=registry,
    )
    out = runtime.coder("write a retry helper")
    assert "Retry only temporary failures" in out

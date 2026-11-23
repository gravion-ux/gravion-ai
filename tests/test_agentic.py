import json
from pathlib import Path

import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError, ProviderError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.agentic.module import (
    TASK_BUILD,
    TASK_PLAN,
    TASK_RUN,
    build_brief,
    detect_task,
    parse_steps,
    parse_tools,
)
from gravion_ai.modules.agentic.plan import (
    PLAN_STATUS,
    LocalPlanner,
    plan_from_prompt,
    write_plan,
)
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry
from gravion_ai.runtime import GravionRuntime


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None):
        self.calls.append((list(messages), model))
        return "recorded reply"


def _ctx(provider, model="gravion-agent"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def test_agentic_instantiates_and_answers():
    module = default_registry().instantiate("agentic")
    out = module.run(_ctx(GravionCoreProvider()), "plan an agent that files our reports")
    assert "plan an agent that files our reports" in out


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("build: an agent that watches the inbox", TASK_BUILD),
        ("build an agent to file reports", TASK_BUILD),
        ("create an agent for onboarding", TASK_BUILD),
        ("run: the inbox agent on today's mail", TASK_RUN),
        ("run the agent on this week's tickets", TASK_RUN),
        ("plan an agent that files our reports", TASK_PLAN),
        ("an agent for expense approvals", TASK_PLAN),
    ],
)
def test_detect_task(prompt, expected):
    assert detect_task(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("an agent using email and calendar", ["email", "calendar"]),
        ("an agent with web search, a browser and the api",
         ["web search", "browser", "api"]),
        ("an agent that reads files and runs code", ["files", "code"]),
        ("an agent for expense approvals", []),
    ],
)
def test_parse_tools(prompt, expected):
    assert parse_tools(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("steps: check inbox, draft replies, send summary",
         ["check inbox", "draft replies", "send summary"]),
        ("step: review the queue", ["review the queue"]),
        ("an agent for expense approvals", []),
    ],
)
def test_parse_steps(prompt, expected):
    assert parse_steps(prompt) == expected


def test_brief_records_task_tools_steps_and_request():
    brief = build_brief(
        "build an agent using email, steps: check inbox, draft replies"
    )
    assert "Task: build" in brief
    assert "Tools: email" in brief
    assert "Steps: check inbox; draft replies" in brief
    brief = build_brief("an agent for expense approvals")
    assert "Task: plan" in brief
    assert "Tools: none stated" in brief
    assert "Steps: model to plan" in brief


@pytest.mark.parametrize(
    "prompt,marker",
    [
        ("plan an inbox agent", "Agent task: plan"),
        ("build an agent for reports", "Agent task: build"),
        ("run: the inbox agent", "Agent task: run"),
    ],
)
def test_agentic_guidance_and_brief_reach_the_provider(prompt, marker):
    provider = RecordingProvider()
    module = default_registry().instantiate("agentic")
    out = module.run(_ctx(provider), prompt)
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-agent"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert marker in messages[0]["content"]
    assert "# Agent brief" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": prompt}


def test_agentic_runtime_dev_end_to_end_with_gravion_agent(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-agent"))
    assert runtime.resolve_model("agentic") == "gravion-agent"
    out = runtime.agentic("plan an agent that files our reports")
    assert "plan an agent that files our reports" in out


def test_agentic_runtime_rejects_model_without_agentic_capability(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.agentic("plan an agent")


def test_agentic_manifest_signs_and_verifies(issuer, agentic_module_copy):
    payload = build_manifest(
        agentic_module_copy,
        name="agentic",
        module_type="agentic",
        version="0.1.0",
        entrypoint="gravion_ai.modules.agentic.module:AgenticModule",
    )
    write_signed_manifest(agentic_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(agentic_module_copy, issuer)
    assert verified["name"] == "agentic"
    assert verified["type"] == "agentic"
    assert "module.py" in verified["files"]
    assert "plan.py" in verified["files"]


def test_bundled_agentic_manifest_is_unsigned(issuer):
    import gravion_ai.modules.agentic as agentic_pkg

    with pytest.raises(ModuleTamperError):
        verify_module(Path(agentic_pkg.__file__).parent, issuer)


# -- the local planner --------------------------------------------------------


def test_plan_from_prompt_uses_request_steps():
    plan = plan_from_prompt(
        "build an agent using email, steps: check inbox, draft replies"
    )
    assert plan.task == "build"
    assert plan.tools == ("email",)
    assert plan.steps == ("check inbox", "draft replies")
    assert plan.steps_source == "request"
    assert plan.status == PLAN_STATUS


def test_plan_from_prompt_falls_back_to_marked_scaffold():
    plan = plan_from_prompt("an agent for expense approvals")
    assert plan.steps_source == "scaffold"
    assert plan.steps  # the scaffold is never empty
    assert plan.status == "planned — not executed"


def test_local_planner_writes_an_honest_plan_file(tmp_path):
    result = write_plan(
        "build an agent using email, steps: check inbox, draft replies",
        tmp_path / "plan.json",
    )
    assert result.provider == "gravion-local-planner"
    document = json.loads((tmp_path / "plan.json").read_text(encoding="utf-8"))
    assert document["provider"] == "gravion-local-planner"
    assert document["status"] == "planned — not executed"
    assert document["steps"] == ["check inbox", "draft replies"]
    assert document["steps_source"] == "request"
    assert document["tools"] == ["email"]


def test_local_planner_refuses_an_empty_request(tmp_path):
    with pytest.raises(ProviderError, match="non-empty request"):
        LocalPlanner().write(plan_from_prompt("   "), tmp_path / "plan.json")


def test_runtime_plan_agent_end_to_end(make_config, tmp_path):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-agent"))
    result = runtime.plan_agent("plan an agent that files reports", tmp_path / "p.json")
    assert result.provider == "gravion-local-planner"
    assert (tmp_path / "p.json").exists()
    assert result.plan.status == PLAN_STATUS

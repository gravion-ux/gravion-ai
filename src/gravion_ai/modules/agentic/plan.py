"""The local planner for the agentic module (Piece 6).

`AgenticModule.run` briefs **Gravion Agent** — the trained model —
exactly like the other modules brief theirs. This planner is the
module's other path: turning a request into a structured plan file
on the machine the deployment runs on, with no network and no
trained weights. It exists so a deployment can develop, demo and
test its agent workflow end-to-end before the trained bundle is
installed.

What the local planner is, honestly:

- It is a **development planner**, not the trained Gravion Agent
  model, and it does not pretend to be: the plan file records the
  provider name (`gravion-local-planner`) so nothing downstream
  can mistake the plan for the model's work.
- A plan is **not an execution**. The plan file's status says so
  in plain words: `planned — not executed`. Nothing in the plan
  is run, no tool is invoked, no side effect happens. Executing
  tool-using agents is the trained bundle's job.
- Steps come from the request when it states them ("steps: …").
  Otherwise the plan carries a fixed, honestly-labelled scaffold
  (clarify → select tools → execute with checks → report) for the
  model or a human to flesh out — never invented specifics.
- An empty request is an error, not an empty plan: the planner
  never reports success for a plan it did not actually write.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...errors import ProviderError
from .module import detect_task, parse_steps, parse_tools

PLAN_STATUS = "planned — not executed"

# The fixed scaffold used when the request states no steps. It is
# a shape, not a plan for the specific goal — the plan file marks
# it as scaffold so nobody mistakes it for tailored steps.
_SCAFFOLD_STEPS = (
    "clarify the goal, constraints and success criteria",
    "select the tools each stage needs",
    "carry out the work step by step, checking each result",
    "report the outcome and anything left undone",
)


@dataclass(frozen=True)
class AgentPlan:
    """A fully-specified plan document: the goal, the task kind,
    the tools, the steps, and the honest status."""

    goal: str
    task: str
    tools: tuple[str, ...]
    steps: tuple[str, ...]
    steps_source: str  # "request" or "scaffold"
    status: str = PLAN_STATUS


@dataclass(frozen=True)
class PlanResult:
    """What writing a plan produced: the file and who planned it."""

    path: Path
    provider: str
    plan: AgentPlan


def plan_from_prompt(prompt: str) -> AgentPlan:
    """Derive a plan document from a plain-language request, using
    the module's own parsers. Steps come from the request when it
    states them; otherwise the fixed scaffold is recorded — and
    marked as scaffold."""
    steps = parse_steps(prompt)
    if steps:
        chosen, source = tuple(steps), "request"
    else:
        chosen, source = _SCAFFOLD_STEPS, "scaffold"
    return AgentPlan(
        goal=prompt.strip(),
        task=detect_task(prompt),
        tools=tuple(parse_tools(prompt)),
        steps=chosen,
        steps_source=source,
    )


@runtime_checkable
class Planner(Protocol):
    """A planner: takes a plan document, writes it as JSON,
    reports what it produced. The trained Gravion Agent bundle and
    the local development planner both sit behind this interface."""

    name: str

    def write(self, plan: AgentPlan, out_path: Path) -> PlanResult:
        ...


class LocalPlanner:
    """The offline development planner.

    See the module docstring for what it is and is not: a small,
    deterministic, honestly-labelled plan file — never an
    execution, and never a stand-in claiming to be the trained
    model.
    """

    name = "gravion-local-planner"

    def write(self, plan: AgentPlan, out_path: Path) -> PlanResult:
        if not plan.goal:
            raise ProviderError(
                "the local planner needs a non-empty request; "
                "describe the goal (and tools or steps if you have "
                "them) or use the trained Gravion Agent bundle"
            )
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "goal": plan.goal,
            "task": plan.task,
            "tools": list(plan.tools),
            "steps": list(plan.steps),
            "steps_source": plan.steps_source,
            "status": plan.status,
            "provider": self.name,
        }
        out_path.write_text(
            json.dumps(document, indent=2) + "\n", encoding="utf-8"
        )
        return PlanResult(path=out_path, provider=self.name, plan=plan)


def write_plan(
    prompt: str,
    out_path: Path,
    planner: Planner | None = None,
) -> PlanResult:
    """Plan a request to a JSON file: derive the plan, then write
    it with the given planner (the local development planner by
    default)."""
    plan = plan_from_prompt(prompt)
    return (planner or LocalPlanner()).write(plan, Path(out_path))

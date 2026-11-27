"""AgenticModule — planning and building tool-using agents
(Piece 6).

The agentic module turns a plain-language request into a
structured **agent brief** for Gravion Agent, following the same
shape as the other modules (manifest in, `run(ctx, prompt)` out)
and adding deterministic logic of its own:

1. **Task detection.** A request is one of three tasks: *plan*
   (design an agent for a goal), *build* (create the agent), or
   *run* (execute an agent on a task). Build and run are detected
   from explicit phrasing — a "build:" / "run:" style prefix, or a
   request to build / run *an agent* — mirroring the coder module's
   task detection. Anything else is planning.

2. **Brief parsing.** Two pieces of intent are extracted: the
   **tools** the request names (web search, browser, email,
   calendar, files, code, shell, database, API — matched from a
   fixed vocabulary, in vocabulary order); and the **steps**, when
   the request states them explicitly ("steps: check inbox, draft
   replies, send summary"). Unstated steps are recorded as "model
   to plan" rather than invented here.

The brief is appended to the deployment's system context together
with the task guidance. The module is served by **Gravion
Agent**, whose first trained bundle ships at Tier 2 in this piece
(see data/models.json); the runtime's registry check refuses a
model that does not serve the agentic capability instead of
silently answering with the wrong model.

`run` is the model path. `plan` is the development path: it hands
the request to the local planner (see plan.py), which writes a
structured plan file offline — honestly labelled as a plan from
the local planner. The plan is a plan: nothing in it is executed,
because executing tool-using agents is the trained bundle's job.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...registry import ModuleContext, ModuleManifest

TASK_PLAN = "plan"
TASK_BUILD = "build"
TASK_RUN = "run"

TASK_GUIDANCE = {
    TASK_PLAN: """

# Agent task: plan
Design a tool-using agent for the goal in the brief below. Lay
out the steps in order, the tool each step needs (from the tools
named in the brief where given), the checks between steps, and
where a human must approve before the agent proceeds. Follow the
deployment's policy context above — it bounds what any agent in
this deployment may do.""",
    TASK_BUILD: """

# Agent task: build
Build the tool-using agent described in the brief below: its
goal, its tools (from the brief), its step sequence, its guard
checks, and its stop conditions. Follow the deployment's policy
context above — it bounds what any agent in this deployment may
do. Prefer the smallest agent that does the job reliably.""",
    TASK_RUN: """

# Agent task: run
Run the agent on the task in the brief below, following the steps
and tools recorded there. Check each step's result before moving
to the next, stop and report instead of improvising when a step
fails, and follow the deployment's policy context above — it
bounds what any agent in this deployment may do.""",
}

_BUILD_PREFIXES = (
    "build:",
    "build an agent",
    "build a agent",
    "create:",
    "create an agent",
    "make an agent",
)

_RUN_PREFIXES = (
    "run:",
    "run the agent",
    "run an agent",
    "execute:",
)

# The tool vocabulary the brief recognises, in the order tools are
# recorded. Matching is on word boundaries so "apishield" does not
# claim the "api" tool.
_TOOLS = (
    "web search",
    "browser",
    "email",
    "calendar",
    "files",
    "code",
    "shell",
    "database",
    "api",
)

_STEPS_RE = re.compile(r"\bsteps?\s*:\s*(.+)", re.IGNORECASE)

# A build/run request need not open with the verb: "please build
# an agent…", "build me an agent…" and "execute the agent…" ask
# for the same tasks as the prefix forms. The agent must be the
# verb's object, so "plan an agent that builds reports" stays a
# plan.
_BUILD_AGENT_RE = re.compile(
    r"\b(?:build|create|make)\s+(?:me\s+)?(?:an?\s+)?agent\b"
)
_RUN_AGENT_RE = re.compile(
    r"\b(?:run|execute)\s+(?:the\s+|an?\s+)?agent\b"
)

# A negated tool ("no email", "without a browser") is a tool the
# request rules out, not one it names. A negator cancels a tool
# only when it governs it: same clause (commas break clauses)
# and within the few words immediately before the tool name.
_TOOL_NEGATOR_RE = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|n't\b)")
_CLAUSE_BREAKS = ".!?;,"


def _negated_before(text: str, match_start: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS)
        + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-3:])
    return bool(tail) and _TOOL_NEGATOR_RE.search(tail) is not None


def _tool_negated(text: str, match_start: int) -> bool:
    return _negated_before(text, match_start)


def detect_task(prompt: str) -> str:
    """Classify an agentic prompt as plan / build / run."""
    text = prompt.strip().lower()
    # A leading "please"/"kindly" does not change the task:
    # "please build: ..." asks to build exactly like "build:".
    prefixed = re.sub(r"^(?:please|kindly)\s+", "", text)
    if prefixed.startswith(_BUILD_PREFIXES):
        return TASK_BUILD
    if prefixed.startswith(_RUN_PREFIXES):
        return TASK_RUN
    # A negated verb rules its task out, the same way a negated
    # tool is not a named tool: "don't build an agent yet, plan
    # one" asks for a plan, and recording it as a build brief
    # handed the model the opposite task.
    if any(
        not _negated_before(text, match.start())
        for match in _BUILD_AGENT_RE.finditer(text)
    ):
        return TASK_BUILD
    if any(
        not _negated_before(text, match.start())
        for match in _RUN_AGENT_RE.finditer(text)
    ):
        return TASK_RUN
    return TASK_PLAN


def parse_tools(prompt: str) -> list[str]:
    """Collect the tools the request names, in vocabulary order."""
    text = prompt.lower()
    tools: list[str] = []
    for tool in _TOOLS:
        for match in re.finditer(rf"\b{re.escape(tool)}\b", text):
            if not _tool_negated(text, match.start()):
                tools.append(tool)
                break
    return tools


def parse_steps(prompt: str) -> list[str]:
    """Collect explicitly stated steps ("steps: a, b, c").

    Only an explicit steps list is recorded; anything else returns
    an empty list, meaning the brief leaves the steps to the model
    rather than inventing them here.
    """
    match = _STEPS_RE.search(prompt)
    if not match:
        return []
    # Steps separate on semicolons as well as commas, and a new
    # sentence after the list ("steps: a, b. Also be quick") is
    # not part of the last step.
    listing = match.group(1)
    sentence = re.search(r"\.\s+(?=[A-Z])", listing)
    if sentence:
        listing = listing[: sentence.start()]
    # A numbered list ("steps: 1) read mail 2) draft 3) send")
    # separates on its markers, numbers stripped. A marker is
    # a short number with a dot or paren and a following space,
    # so a version ("2.0 notes") inside a step never splits it.
    numbered = re.split(r"(?:^|(?<=[\s;,]))\d{1,2}[.)]\s+", listing)
    if len(numbered) > 1:
        steps = [
            part.strip().strip(",;").strip().rstrip(".")
            for part in numbered
        ]
        return [step for step in steps if step]
    steps = [
        part.strip().rstrip(".")
        for part in re.split(r"[;,]", listing)
    ]
    return [step for step in steps if step]


def build_brief(prompt: str) -> str:
    """Render the structured agent brief for a request."""
    tools = parse_tools(prompt)
    steps = parse_steps(prompt)
    return (
        "\n\n# Agent brief\n"
        f"Task: {detect_task(prompt)}\n"
        f"Tools: {', '.join(tools) if tools else 'none stated'}\n"
        f"Steps: {'; '.join(steps) if steps else 'model to plan'}\n"
        f"Request: {prompt.strip()}"
    )


class AgenticModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        guidance = TASK_GUIDANCE[detect_task(prompt)]
        messages = [
            {
                "role": "system",
                "content": ctx.system_context + guidance + build_brief(prompt),
            },
            {"role": "user", "content": prompt},
        ]
        return ctx.provider.complete(messages, model=ctx.model)

    def plan(self, prompt: str, out_path, planner=None):
        """Write a structured plan for the request via the local
        planner.

        This is the development path (see plan.py): the output is
        a plan file from the local planner, honestly labelled —
        and a plan is all it is. Nothing in the plan is executed;
        running tool-using agents is the trained Gravion Agent
        bundle's job, served through `run`.
        """
        from .plan import LocalPlanner, plan_from_prompt

        plan = plan_from_prompt(prompt)
        planner = planner or LocalPlanner()
        return planner.write(plan, Path(out_path))

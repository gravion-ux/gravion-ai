# Gravion AI — Public AI/Hacker Agent Patterns

Date: 2026-11-26  
Scope: public, lawful sources only. No leaked or proprietary source code was used. This note extracts architecture patterns and shortcomings to avoid; it does not copy code.

## Executive takeaway

The strongest public AI coding/security agents are not just prompts around a model. They are harnesses: durable state, typed tools, scoped execution, approvals, traces, evals, memory governance, and provider routing. Gravion AI should borrow those harness patterns while keeping its own original implementation and honest parameter story.

## Patterns by project

### OpenHands

Public pattern: event-sourced conversation state, typed actions/observations, decoupled SDK/tools/workspace/server packages, optional sandbox isolation, stuck detection, budgets, and replayable event logs.

Gravion AI upgrade:
- Add `RunEvent` records for user messages, model calls, tool requests, tool results, approvals, checkpoints, and completion.
- Keep one source of truth for run state; project UI/CLI views from events.
- Separate core runtime, tools, workspace execution, and server surfaces.
- Add stuck detection: repeated tool calls, no-progress loops, budget exhaustion, and max-step limits.

Shortcoming to avoid:
- Monolithic config sprawl and tight coupling between research code, UI, tools, and sandbox assumptions.

### Aider

Public pattern: repository map built from symbol definitions/references, graph ranking, token-budgeted context, git-native edits, lint/test feedback, and rollback-friendly patches.

Gravion AI upgrade:
- Build a `RepoMap` for coder/agentic modules: files, symbols, imports, tests, recent diffs, ranked context, token budget.
- Show patch previews before applying edits.
- After edits, run syntax/lint/tests and feed failures back into the run.

Shortcoming to avoid:
- Dumping whole files into prompts. Context should be compiled and ranked, not pasted blindly.

### PentestGPT-style public projects

Public pattern: staged security workflow, reasoning/generation/parsing separation, pentesting task tree, human-in-the-loop guidance, durable state, evidence provenance, and report generation.

Gravion AI upgrade:
- For cyber/security modules, use staged runs: scope confirmation → asset discovery → hypothesis → validation → evidence → report.
- Maintain a task tree with status, evidence links, confidence, and next actions.
- Keep deterministic scope checks outside the model: allowed targets, allowed methods, rate limits, and approval gates.
- Require evidence provenance for every finding: command/output/log reference, timestamp, target, and reproduction notes.

Shortcoming to avoid:
- Letting the model invent scope, evidence, or severity. Security conclusions must be tied to observed evidence and authorized scope.

### LangGraph

Public pattern: graph/state-machine execution, checkpointers, thread IDs, interrupts, human-in-the-loop resume, and durable state snapshots.

Gravion AI upgrade:
- Add checkpointed runs with `run_id`/`thread_id`, step snapshots, pause/resume, and approval interrupts.
- Make side effects idempotent or place them after approval points.
- Store interrupt payloads with enough context for a human decision: proposed action, inputs, risk, and rollback.

Shortcoming to avoid:
- Re-running non-idempotent work after resume. Expensive or destructive steps must be guarded by idempotency keys and approval state.

### AutoGen

Public pattern: event-driven actor model, typed asynchronous messages, layered Core/AgentChat/Extensions APIs, explicit state save/load, termination conditions, cancellation, and observability.

Gravion AI upgrade:
- Define typed internal events: `TaskCreated`, `ModelRequested`, `ToolRequested`, `ToolResult`, `ApprovalNeeded`, `RunCompleted`, `RunFailed`.
- Add cancellation tokens and termination rules for every run.
- Emit traces for model calls, tool calls, memory reads/writes, approvals, and errors.

Shortcoming to avoid:
- Loose multi-agent chat loops without typed events, cancellation, or termination rules.

### CrewAI

Public pattern: role-based crews for prototyping, deterministic flows for production control, unified memory with hierarchical scopes, composite recall scoring, consolidation, and forget/reset controls.

Gravion AI upgrade:
- Add memory scopes: deployment, company, user, project, session, module.
- Score recall by semantic similarity, recency, and importance; show match reasons in debug views.
- Add provenance, expiry, contradiction handling, and forget controls.

Shortcoming to avoid:
- One unscoped memory pile. Unscoped memory creates contradictions, stale facts, and privacy leaks.

## Cross-cutting shortcomings to fix in Gravion AI

1. **State loss:** chat history is not durable state. Fix with checkpoints, event logs, and resume.
2. **Tool compounding failure:** one bad tool call can poison a run. Fix with schemas, timeouts, retries, idempotency, and result verification.
3. **Unsafe execution:** code/shell/browser/file tools need sandboxing, allowlists, dry-run diffs, and approvals.
4. **Memory pollution:** memory needs scopes, provenance, expiry, importance, and forget controls.
5. **Provider drift:** model names, limits, and errors change. Fix with aliases, capability checks, fallbacks, and contract tests.
6. **Cost blindness:** retries and loops hide spend. Fix with per-run budgets, token/cost caps, and loop limits.
7. **Weak evals:** demos pass while production fails. Fix with golden tasks, regression evals, RAG metrics, and release gates.
8. **Observability gaps:** without traces, debugging is guesswork. Fix with structured spans for model, tool, memory, retrieval, approval, and error steps.
9. **UI opacity:** users need run timelines, tool cards, citations, diffs, and exportable reports.
10. **Security overclaiming:** hacker/security modules must separate hypotheses from verified findings and keep scope/evidence explicit.

## Recommended Gravion AI implementation order

1. `runtime/events.py` — typed run events and append-only event log.
2. `runtime/checkpoints.py` — SQLite checkpoint store with run/thread IDs.
3. `tools/contracts.py` — tool schemas, risk levels, timeout/retry/idempotency policy.
4. `providers/gateway.py` — model aliases, fallback chains, budgets, redacted logs, cost ledger.
5. `memory/scopes.py` — scoped memory with provenance, expiry, importance, and forget.
6. `coder/repomap.py` — ranked repository context for coder/agentic modules.
7. `security/task_tree.py` — authorized-scope task tree with evidence provenance.
8. `evals/` — golden tasks per module and release gates.
9. Web UI — run timeline, tool cards, approvals, diff preview, exportable run report.

## Parameter notes

### Gravion AI

Gravion AI currently has **no trained neural weights** and therefore **no Gravion-owned parameter count**. Its offline core is a deterministic rule-based engine. When Gravion AI connects to an endpoint model, that model's parameter count belongs to the connected model, not to Gravion AI.

Correct wording:
- “Gravion AI is a modular AI platform/runtime. It does not ship trained neural weights in this build.”
- “Connected endpoint models may have their own parameter counts; those counts are not Gravion-owned parameters.”
- Avoid: “Gravion AI has X billion parameters” unless Gravion-owned trained weights actually exist and are documented.

### Public open-model parameter examples

These are examples of publicly stated open-model sizes. They are endpoint/model facts, not Gravion AI parameters.

| Model family | Publicly stated sizes | Notes |
|---|---:|---|
| Llama 3.1 | 8B, 70B, 405B | Dense models; 128K context in public summaries. |
| Llama 3.2 | 1B, 3B, plus vision sizes in some summaries | Small/edge track. |
| Qwen2.5 / Qwen3 | 0.5B–72B dense ranges plus MoE variants | Strong multilingual/code positioning. |
| DeepSeek-V3 / R1 | 671B total, about 37B active | MoE; total vs active parameters must be distinguished. |
| Mistral 7B | 7B | Dense small model. |
| Mixtral 8x7B | about 47B total, about 13B active | MoE; do not describe active params as total params. |
| Mixtral 8x22B | about 141B total, about 39B active | MoE. |
| Gemma 3 | 1B, 4B, 12B, 27B in public summaries | Lightweight/open track. |

Parameter-count rules for Gravion AI docs:
- Always distinguish **total parameters** from **active parameters** for MoE models.
- Do not transfer a connected model's parameter count to Gravion AI.
- Do not claim trained Gravion bundles/weights until weights, training provenance, evals, and release artifacts exist.
- Prefer capability claims tied to evals over parameter-count marketing.

## Sources

- OpenHands survey: https://github.com/signalnine/thunderdome/blob/HEAD/docs/survey/openhands.md
- OpenHands SDK summary: https://ai-infrastructure.net/openhands-agent-sdk/
- Aider survey: https://github.com/signalnine/thunderdome/blob/HEAD/docs/survey/aider.md
- Aider repository mapping: https://deepwiki.com/Aider-AI/aider/4.1-repository-mapping
- PentestGPT public repo: https://github.com/GreyDGL/PentestGPT
- PentestGPT USENIX appendix: http://usenix.org/system/files/usenixsecurity24-appendix-deng.pdf
- LangGraph interrupts: https://docs.langchain.com/oss/javascript/langgraph/interrupts
- AutoGen research note: https://github.com/stephendenisedwards/micro-x-agent-loop-python/blob/HEAD/documentation/docs/research/deep-research-autogen-0.4.md
- CrewAI memory docs: https://docs.crewai.com/v1.10.0/en/concepts/memory
- Open-weight model sizes summary: https://www.theorempath.com/topics/llama-and-open-weight-models
- Ollama model parameter cheatsheet: https://www.quickcfg.com/en/cheatsheet/ollama/

# Gravion AI Upgrade Plan — 26 Nov 2026

## Purpose

Upgrade Gravion AI from a modular platform scaffold with strong licensing/runtime plumbing into a more complete local AI product surface: routed providers, explicit tools, memory, evaluations, traces, safety gates, a polished web workspace, and a Trainer path that can prepare, validate, evaluate, and register trained variants without claiming weights that do not exist.

This plan is based on the current local project at version `0.11.0`: Python package under `src/gravion_ai/`, module packages under `src/gravion_ai/modules/`, provider plumbing in `providers.py`, runtime orchestration in `runtime.py`, deployment config in `config.py`, licensing/entitlement in `src/gravion_ai/licensing/`, training dataset/job code in `src/gravion_ai/training/`, and a standard-library web UI in `web.py`.

## Current baseline

- Providers: `GravionCoreProvider` and `EndpointProvider` behind `ChatProvider.complete(messages, model=None) -> str`.
- Runtime path: `GravionRuntime.run_module()` gates entitlement, verifies production module signatures, resolves a model through `ModelCatalog`, builds `ModuleContext`, then calls `module.run(ctx, prompt)`.
- Modules: chat, support, coder, image, music, video, website-builder, cyber, agentic, all-in-one.
- Agentic module currently parses named tools/steps and writes plans; it does not execute tools.
- Web UI: one embedded HTML page plus `/api/status`, `/api/bot/status`, `/api/uploads`, `/api/files`, `/api/chat`.
- Training: dataset preparation and job validation exist; `run_training_job()` honestly refuses because this build has no training backend.
- Tests/CI: pytest suite in `tests/`, GitHub Actions matrix for Python 3.10–3.12, coverage upload, and CLI smoke test.

## Design rules for the upgrade

1. Keep the existing module contract working while adding richer context objects.
2. Keep Gravion Core offline and deterministic for tests and development.
3. Treat endpoint models as operator-connected plumbing; do not claim connected model parameters as Gravion-owned weights.
4. Make every side-effecting capability explicit, allowlisted, traced, and testable.
5. Keep licensing entitlement separate from product safety gates.
6. Prefer standard-library implementations unless a dependency clearly pays for itself.
7. Every phase must land with tests first or tests in the same change.

---

## Phase 0 — Baseline freeze and compatibility contract

### Goal
Lock the current behavior before refactoring so provider/tool/memory changes do not break modules, licensing, or the web API.

### Files to change
- `README.md` — add an “Upgrade track” note pointing to this plan after Phase 1 starts.
- `ARCHITECTURE.md` — add sections for provider routing, tools, memory, evals, tracing, and safety once each phase lands.
- `CHANGELOG.md` — add an Unreleased section for the upgrade track.
- `tests/test_runtime.py` — add characterization tests for current `run_module()` ordering: gate → manifest verification → model resolution → module execution.
- `tests/test_providers.py` — pin current `complete()` behavior for Gravion Core and endpoint payload shape.
- `tests/test_web_files_bots.py` — pin current `/api/chat` response keys before adding trace/memory fields.

### Acceptance tests
- `python -m pytest tests/test_runtime.py tests/test_providers.py tests/test_web_files_bots.py -q`
- Existing full suite still passes: `python -m pytest -q --cov=gravion_ai --cov-report=xml`

---

## Phase 1 — Provider router

### Goal
Replace the single-provider assumption with a router that can choose Gravion Core or one/more configured endpoints by capability, tier, latency budget, cost class, and fallback policy.

### Files to change
- `src/gravion_ai/providers.py`
  - Add `ProviderResult` with `text`, `provider`, `model`, `latency_ms`, `usage`, and `raw` metadata.
  - Keep `ChatProvider.complete()` backward compatible.
  - Add `ProviderRouter.complete_result(messages, capability, model=None, route_hint=None)`.
  - Add bounded retry/fallback for endpoint HTTP 429/5xx and timeouts; never retry non-idempotent tool execution at this layer.
- `src/gravion_ai/config.py`
  - Extend `ProviderConfig` or add `ProviderRouteConfig` with route name, provider type, model, capabilities, priority, timeout, and fallback order.
  - Preserve current `provider` config compatibility for existing deployments.
  - Validate production rules: no stored API keys in production, HTTPS for remote endpoints.
- `src/gravion_ai/catalog.py`
  - Add capability/tier lookup helpers used by the router.
  - Validate optional model metadata: `context_window`, `supports_tools`, `supports_json`, `cost_class`, `latency_class`.
- `src/gravion_ai/data/models.json`
  - Add routing metadata to Gravion model entries without changing model IDs.
- `src/gravion_ai/runtime.py`
  - Build `ProviderRouter` in `GravionRuntime.__init__`.
  - Pass routed provider/model metadata into `ModuleContext`.
  - Keep `resolve_model()` behavior for registry validation and all-in-one checks.
- `src/gravion_ai/cli.py`
  - Add `gravion providers list` and `gravion providers test --capability chat` using a mock/offline-safe path for Gravion Core.
- `tests/test_provider_router.py` — new.
- `tests/test_providers.py`, `tests/test_config.py`, `tests/test_runtime.py`, `tests/test_catalog.py` — update.

### Acceptance tests
- Router chooses Gravion Core offline when no endpoint is configured.
- Router chooses endpoint route for `coder` when configured for that capability.
- Router falls back to the next route on simulated endpoint 500/timeout and records the fallback reason.
- Router refuses a model that does not serve the requested capability.
- Existing modules still work through `runtime.chat("hello")` with Gravion Core.
- Commands:
  - `python -m pytest tests/test_provider_router.py tests/test_providers.py tests/test_config.py tests/test_runtime.py tests/test_catalog.py -q`

---

## Phase 2 — Tool registry

### Goal
Turn the agentic module’s parsed tool names into a real, explicit tool registry with schemas, permissions, side-effect classes, and dry-run support.

### Files to change
- `src/gravion_ai/tool_registry.py` — new.
  - `ToolSpec`: name, description, input schema, output schema, side-effect class (`read`, `write_local`, `network`, `execute`), required entitlement module, timeout.
  - `ToolRegistry.register()`, `list()`, `describe()`, `validate_call()`, `run()`.
  - Built-in tools: `model_catalog.lookup`, `files.context`, `doctor.check`, `agent.plan`, `website.build`, `video.render`, `training.prepare_dataset`, `training.validate_job`.
  - No arbitrary shell tool by default.
- `src/gravion_ai/registry.py`
  - Extend `ModuleContext` with `tools`, `memory`, `trace_id`, and `safety` fields while keeping old constructor usage compatible.
- `src/gravion_ai/runtime.py`
  - Construct the default tool registry.
  - Route local actions (`plan_agent`, `build_website`, `render_video`) through tool specs where practical.
- `src/gravion_ai/modules/agentic/module.py`
  - Replace free-form tool vocabulary with registry-backed tool names.
  - Keep deterministic parsing tests stable.
- `src/gravion_ai/modules/agentic/plan.py`
  - Include tool side-effect class and required approval flag in generated plans.
- `src/gravion_ai/cli.py`
  - Add `gravion tools list`, `gravion tools describe <name>`, `gravion tools run <name> --dry-run`.
- `src/gravion_ai/web.py`
  - Add `GET /api/tools` and include available tools in `/api/status`.
- `tests/test_tool_registry.py` — new.
- `tests/test_agentic.py`, `tests/test_runtime.py`, `tests/test_web_files_bots.py` — update.

### Acceptance tests
- Unknown tool calls fail validation with a precise error.
- `write_local` tools require explicit allowlist/approval in production mode.
- Dry-run returns a plan and performs no file writes.
- Agentic plan output includes tool names, side-effect classes, and approval requirements.
- Web `/api/tools` returns only enabled tools.
- Commands:
  - `python -m pytest tests/test_tool_registry.py tests/test_agentic.py tests/test_runtime.py tests/test_web_files_bots.py -q`

---

## Phase 3 — Memory

### Goal
Add local, exportable, operator-controlled memory for conversations and deployment facts, with retention and redaction controls.

### Files to change
- `src/gravion_ai/memory.py` — new.
  - `MemoryStore` backed by SQLite via `sqlite3` or JSONL under deployment `data/memory/`.
  - Records: `id`, `scope`, `kind`, `text`, `source`, `created_at`, `expires_at`, `tags`, `redacted`.
  - APIs: `add()`, `search()`, `recent()`, `forget()`, `export()`, `compact()`.
  - Default retrieval: recent conversation turns plus explicit deployment facts; no silent long-term learning from prompts unless enabled.
- `src/gravion_ai/config.py`
  - Add `MemoryConfig`: enabled, max_context_items, retention_days, store_prompts, store_outputs, redact_secrets.
  - Validate unknown fields strictly like existing config.
- `src/gravion_ai/runtime.py`
  - Inject memory context into `ModuleContext`.
  - Add `run_module(..., session_id=None, remember=False)` without breaking existing calls.
- `src/gravion_ai/web.py`
  - Accept `session_id` and `remember` in `/api/chat`.
  - Add `GET /api/memory`, `DELETE /api/memory/<id>`, `POST /api/memory/forget-session`.
- `src/gravion_ai/cli.py`
  - Add `gravion memory list`, `gravion memory search`, `gravion memory forget`, `gravion memory export`.
- `src/gravion_ai/files.py`
  - Ensure uploaded file context can be referenced by memory records without copying file contents into memory by default.
- `tests/test_memory.py` — new.
- `tests/test_config.py`, `tests/test_runtime.py`, `tests/test_web_files_bots.py` — update.

### Acceptance tests
- Memory is disabled by default for prompt storage unless config enables it.
- A remembered fact is retrieved in a later run with the same session/deployment scope.
- `forget` removes the record and later retrieval does not return it.
- Secret-looking strings are redacted before storage when redaction is enabled.
- Export includes memory records and remains available even when entitlement is suspended.
- Commands:
  - `python -m pytest tests/test_memory.py tests/test_config.py tests/test_runtime.py tests/test_web_files_bots.py -q`

---

## Phase 4 — Eval harness

### Goal
Add repeatable evaluations for modules, provider routes, tools, safety behavior, and Trainer outputs.

### Files to change
- `src/gravion_ai/evals.py` — new.
  - `EvalCase`: id, module, prompt, expected checks, tags, provider route, tool expectations.
  - Check types: contains, not_contains, regex, JSON schema-ish key checks, tool-call expectation, refusal expectation, latency budget.
  - `EvalRunner.run()` returns per-case pass/fail, score, latency, trace ID, and failure reasons.
  - Deterministic seed/order for local runs.
- `tests/fixtures/evals/core.jsonl` — new small fixture set for chat/support/coder/cyber/agentic.
- `src/gravion_ai/cli.py`
  - Add `gravion eval run --suite core --provider gravion-core`.
  - Add `--json` output for CI artifacts.
- `src/gravion_ai/training/datasets.py`
  - Reuse validation/dedupe helpers for eval dataset loading where appropriate.
- `src/gravion_ai/runtime.py`
  - Allow eval runner to call modules with tracing enabled and memory disabled by default.
- `tests/test_evals.py` — new.
- `tests/test_cli.py` — add eval CLI smoke tests.

### Acceptance tests
- Core eval suite passes against Gravion Core for deterministic cases.
- A deliberately wrong expected string fails with a readable reason.
- JSON output includes suite name, total, passed, failed, score, and case IDs.
- Eval run does not write memory unless explicitly requested.
- Commands:
  - `python -m pytest tests/test_evals.py tests/test_cli.py -q`
  - `gravion eval run --suite core --provider gravion-core --json`

---

## Phase 5 — Tracing and logs

### Goal
Add structured traces for every module run, provider call, tool call, safety decision, and eval case, while keeping the licensing audit log separate and tamper-evident.

### Files to change
- `src/gravion_ai/tracing.py` — new.
  - `TraceEvent`, `TraceRecorder`, `new_trace_id()`, JSONL writer under `state/traces.jsonl` or `data/traces/`.
  - Fields: trace_id, span_id, parent_span_id, event, module, model, provider, route, latency_ms, status, error_type, tool_name, safety_decision.
  - Prompt/output content excluded by default; optional hashed fingerprints only.
- `src/gravion_ai/runtime.py`
  - Start a trace per `run_module()` call.
  - Record gate, manifest verification, model resolution, provider call, tool calls, and completion/failure.
- `src/gravion_ai/providers.py`
  - Return timing/usage metadata through `ProviderResult` for tracing.
- `src/gravion_ai/tool_registry.py`
  - Record tool validation, dry-run, execution, timeout, and failure events.
- `src/gravion_ai/licensing/audit.py`
  - Keep audit log unchanged for licensing/security events; do not mix product traces into the hash-chained audit log.
- `src/gravion_ai/config.py`
  - Add `ObservabilityConfig`: tracing_enabled, trace_content_mode (`none`, `hash`, `redacted`), max_trace_bytes.
- `src/gravion_ai/web.py`
  - Add `GET /api/traces/recent` with redacted summaries.
  - Include `trace_id` in `/api/chat` responses.
- `src/gravion_ai/cli.py`
  - Add `gravion trace recent` and `gravion trace show <trace_id>`.
- `tests/test_tracing.py` — new.
- `tests/test_runtime.py`, `tests/test_web_files_bots.py`, `tests/test_audit.py` — update.

### Acceptance tests
- A chat run creates one trace with provider and completion events.
- A provider failure records status/error without leaking API keys or prompt text.
- Tool dry-run and tool execution produce distinct trace events.
- Licensing audit verification still passes and remains separate from traces.
- Web chat response includes `trace_id` and recent traces endpoint returns redacted summaries.
- Commands:
  - `python -m pytest tests/test_tracing.py tests/test_runtime.py tests/test_web_files_bots.py tests/test_audit.py -q`

---

## Phase 6 — Safety gates

### Goal
Add product safety checks around prompts, files, tools, outputs, and cyber/agentic workflows without weakening the existing licensing gate.

### Files to change
- `src/gravion_ai/safety.py` — new.
  - `SafetyPolicy`, `SafetyDecision`, `check_input()`, `check_file_context()`, `check_tool_call()`, `check_output()`.
  - Checks: secret patterns, prompt-injection markers in uploaded files, disallowed tool side effects, production endpoint policy, output credential leakage, cyber authorized-use framing.
  - Decisions: allow, warn, block, require_approval.
- `src/gravion_ai/config.py`
  - Add `SafetyConfig`: enabled, block_secrets, require_tool_approval, allowed_tool_side_effects, cyber_authorized_use_required.
- `src/gravion_ai/runtime.py`
  - Run input safety before module execution and output safety before returning.
  - Record safety decisions in traces.
- `src/gravion_ai/tool_registry.py`
  - Enforce side-effect policy before execution.
- `src/gravion_ai/files.py`
  - Mark uploaded text as untrusted context and run file-context safety checks.
- `src/gravion_ai/modules/cyber/module.py`
  - Keep defensive framing explicit; require authorized scope language for active-testing style requests.
- `src/gravion_ai/modules/agentic/module.py` and `plan.py`
  - Mark network/execute tools as approval-required in plans.
- `src/gravion_ai/doctor.py`
  - Add checks for unsafe production config: stored API key, HTTP remote endpoint, tracing content mode, tool approval disabled.
- `tests/test_safety.py` — new.
- `tests/test_gate.py`, `tests/test_cyber.py`, `tests/test_agentic.py`, `tests/test_web_files_bots.py` — update.

### Acceptance tests
- Secret-looking output is blocked or redacted according to policy.
- Uploaded file prompt-injection text is treated as data, not instructions, and is flagged in trace/safety decision.
- Production mode blocks `execute` tools unless explicitly allowlisted.
- Cyber module refuses active unauthorized testing framing and offers defensive/reporting help.
- Doctor reports unsafe production settings.
- Commands:
  - `python -m pytest tests/test_safety.py tests/test_gate.py tests/test_cyber.py tests/test_agentic.py tests/test_web_files_bots.py -q`

---

## Phase 7 — Web UI polish

### Goal
Turn the single-page local UI into a usable workspace: conversation history, module/tool visibility, provider route status, traces, memory controls, eval results, and clearer honesty labels.

### Files to change
- `src/gravion_ai/web.py`
  - Keep standard-library server and loopback default.
  - Split inline assets only if needed; otherwise keep one-file deployment simplicity.
  - Add panels: status, provider route, tools, memory, traces, evals, uploads, bot slot.
  - Add conversation transcript with session ID, copy answer, retry, clear session.
  - Add loading/error states per request and accessible labels/focus states.
  - Add endpoints: `/api/tools`, `/api/memory`, `/api/traces/recent`, `/api/evals/run`, `/api/providers`.
- `docs/WEB.md`
  - Document local-only default, reverse-proxy/auth responsibility, new endpoints, and data storage locations.
- `src/gravion_ai/files.py`
  - Expose upload metadata needed by the UI without exposing raw stored paths unnecessarily.
- `tests/test_web_files_bots.py`
  - Add API tests for new endpoints and response shapes.
- `tests/test_web_api.py` — new if web tests become too large.

### Acceptance tests
- `/` loads with module selector, prompt box, file attach, status, tools, memory, and trace panels.
- `/api/chat` returns answer, module, files, trace_id, provider route, and safety decision summary.
- Empty prompt, unknown module, oversized body, and suspended deployment return stable JSON errors.
- UI works without external network assets.
- Keyboard-only flow can select module, enter prompt, send, and read output.
- Commands:
  - `python -m pytest tests/test_web_files_bots.py tests/test_web_api.py -q`

---

## Phase 8 — Trainer integration

### Goal
Connect dataset preparation, training job validation, evals, model registration, and bundle descriptors into one honest Trainer workflow.

### Files to change
- `src/gravion_ai/training/datasets.py`
  - Add dataset report fields for source counts, dedupe rate, split sizes, language/tag distribution, and safety-filter counts.
  - Add optional eval split generation separate from validation split.
- `src/gravion_ai/training/jobs.py`
  - Extend job schema with backend (`none`, `external-command`, `hosted`), base model, output model ID, eval suite, seed, and artifact paths.
  - Keep honest refusal when backend is `none`.
- `src/gravion_ai/training/backends.py` — new.
  - `TrainingBackend` protocol.
  - `NoBackend` for current behavior.
  - `ExternalCommandBackend` for operator-configured local training commands with explicit allowlist and timeout.
  - `HostedBackend` client stub that validates payloads and reports “not configured” until a real hosted endpoint exists.
- `src/gravion_ai/training/trainer.py` — new.
  - Orchestrate prepare → validate → train/dry-run → eval → model card → registry candidate.
  - Produce `training-report.json` and `model-card.md`.
- `src/gravion_ai/catalog.py`
  - Add helper to register a private trained variant into deployment `models.json` after eval passes.
- `src/gravion_ai/bundles.py`
  - Include training report hash, eval score, dataset hash, and model-card path in bundle descriptors.
- `src/gravion_ai/cli.py`
  - Extend `gravion train` with `prepare`, `validate`, `run`, `eval`, `register` subcommands.
  - Keep current `gravion train` behavior compatible where tests expect it.
- `src/gravion_ai/web.py`
  - Add Trainer panel showing dataset reports, job validation, eval scores, and registration status.
- `docs/PRODUCTION.md`
  - Document what Trainer can do locally versus what requires a real training backend/hosted pipeline.
- `tests/test_trainer_integration.py` — new.
- `tests/test_training.py`, `tests/test_training_datasets.py`, `tests/test_bundles.py`, `tests/test_catalog.py` — update.

### Acceptance tests
- Preparing a small dataset writes train/validation/eval splits and a hashed report.
- Validating a job with backend `none` passes validation but run refuses honestly.
- External-command dry-run validates command allowlist and writes a training report without claiming trained weights.
- Eval gate blocks model registration when score is below threshold.
- Passing eval registers a private model variant in deployment `models.json` and bundle descriptor includes dataset/eval hashes.
- Commands:
  - `python -m pytest tests/test_trainer_integration.py tests/test_training.py tests/test_training_datasets.py tests/test_bundles.py tests/test_catalog.py -q`

---

## Phase 9 — Tests, CI, and release readiness

### Goal
Make the upgrade safe to publish: full local tests, CI matrix, smoke tests, docs, and versioning discipline.

### Files to change
- `pyproject.toml`
  - Add dev tools only if adopted (`ruff`, `mypy`) and configure them narrowly.
  - Bump version only when a phase is release-ready.
- `.github/workflows/ci.yml`
  - Keep Python 3.10–3.12 matrix.
  - Add eval smoke step: `gravion eval run --suite core --provider gravion-core --json`.
  - Add web API smoke step using a temporary `GRAVION_HOME`.
  - Keep coverage upload non-blocking unless the project chooses a coverage gate.
- `README.md`
  - Update feature status honestly per phase; do not mark Trainer/backend capabilities as complete before tests prove them.
- `ARCHITECTURE.md`
  - Document final data flow: config → safety → memory → provider router → tools → module → output safety → trace.
- `CHANGELOG.md`
  - Record each phase with user-visible changes and migration notes.
- `docs/PRODUCTION.md`, `docs/WEB.md`, `docs/BOT-SLOT.md`
  - Update operational guidance for traces, memory retention, tool approvals, and Trainer artifacts.
- `tests/conftest.py`
  - Add shared fixtures for temp deployment home, fake endpoint provider, tool registry, memory store, and trace recorder.

### Acceptance tests
- Full suite: `python -m pytest -q --cov=gravion_ai --cov-report=xml`
- CLI smoke:
  - `gravion --help`
  - `GRAVION_HOME=$(mktemp -d) gravion init --name Gravion`
  - `gravion status`
  - `gravion modules list`
  - `gravion chat "hello gravion"`
  - `gravion eval run --suite core --provider gravion-core --json`
- CI is green on the final commit before any publish step.
- No phase is described as shipped in README/ARCHITECTURE until its acceptance tests pass.

## Recommended implementation order

1. Phase 0 baseline tests.
2. Phase 1 provider router.
3. Phase 5 tracing minimal recorder, because later phases need observability.
4. Phase 2 tool registry.
5. Phase 6 safety gates around tools/files/output.
6. Phase 3 memory.
7. Phase 4 eval harness.
8. Phase 8 Trainer integration using evals as the registration gate.
9. Phase 7 web UI polish after APIs stabilize.
10. Phase 9 release readiness.

## Non-goals for this upgrade

- No claim of Gravion-owned trained weights unless a real training backend produces and evaluates them.
- No arbitrary shell execution tool by default.
- No public hosting, account system, or multi-tenant SaaS layer in this local upgrade.
- No weakening of licensing entitlement, signed-module verification, export rights, or audit-log separation.
- No external publishing from this plan; implementation should be reviewed and tested locally first.

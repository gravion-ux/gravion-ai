# Changelog

## Unreleased

- Local training validation: training now rejects prompts and completions with no word tokens (a punctuation-only prompt previously trained an exact-match entry for the empty normalized prompt, making every empty prompt a confident match) and rejects conflicting completions for the same normalized prompt instead of silently keeping the last one. Exact-prompt keys in weights artifacts must be non-empty normalized prompts, and an empty prompt is never a confident match at inference. Removed the unused `DEFAULT_BUCKETS` constant.
- All-in-one routing: directives are now case-insensitive and the space after the colon is optional (`coder:fix this` routes to coder instead of silently falling back to chat). Requests without a directive route by a conservative intent check — a clear verb-plus-object signal for exactly one module (code fence, CVE id, `write a python function`, `build a website`) selects that module; anything ambiguous still goes to chat. Removed two unused licensing imports.

- Local Gravion Core training: Gravion Core now includes a sparse n-gram embedding + direct linear softmax classifier trained only from the bundled Gravion corpus (5,657 examples, 779 answer classes; 10,586,543 parameters counted from stored quantized weights; held-out eval 1,238/1,530, 80.9%, versus 0/1,530 for the earlier 750-parameter seed model), with local `gravion-core` training jobs saving verifiable artifacts under deployment `data/models/` and deterministic fallback when weights are missing, invalid, or not confident. Legacy v1 artifacts remain loadable.
- P0 runtime reliability: model registry now carries explicit `weights_status` values so `available` cannot imply included trained weights; endpoint provider gained bounded retries/backoff, timeout/duration logging with redacted metadata, optional temperature/max_tokens, and Gravion request IDs; runtime now writes a persistent content-free run/session event log at `state/run-events.jsonl`.
- Production hardening: strict deployment config validation (unknown fields, endpoint URL rules, production HTTPS for remote endpoints, no stored API keys in production config), safer endpoint-provider errors (bounded responses, text-only content, HTTP status without upstream body leaks), and `gravion doctor` readiness checks.
- Web UI: `gravion web serve` runs a local browser workspace with module chat and file attachments. Files are stored under the deployment data directory, text is extracted for context, and binary files are stored with metadata only.
- Bot slot: `gravion bot status|check` inspects a drop-in slot for an operator-owned bot. The slot validates an HTTP manifest only; Gravion never executes dropped bot code and never recreates the operator's bot.
- Documentation: production runbook, web UI guide, and bot-slot input checklist.

## 0.11.0

- Completed the module roadmap: chat, support, coder, website-builder, music, image, video, cyber, agentic and all-in-one modules; tier bundles; hosted licensing authority service; governed self-upgrade; training dataset preparation and job validation.
- Licensing: signed licenses and module manifests, tamper-evident audit log, server-side entitlement tokens, safe suspension and export while suspended.

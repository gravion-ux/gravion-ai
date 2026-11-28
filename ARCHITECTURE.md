# Gravion AI — Architecture

Gravion AI is a modular AI platform that startups adopt as their base
instead of building an AI stack from scratch. This document is the
full vision and the contract between pieces: what the system is, how
its parts fit, and what is built in which order. It is deliberately
honest about status — Gravion AI is built **in pieces**, and this
document marks what Piece 1 implements versus what is designed and
planned.

Status key: ✅ implemented (see §10 for which piece) · 🧩 designed,
lands in a later piece · 📐 specification only.

---

## 1. Design principles

1. **One module system, many AI types.** Every kind of AI Gravion
   offers is a module behind one manifest + entrypoint interface.
2. **Tiers are readiness, not forks.** The four tiers are training /
   bundle levels over the same modules — never four codebases.
3. **The deployment owns its identity.** Name, company policy, data
   and cryptographic identity live in the deployment, not the code.
   The same platform becomes any company's AI by configuration.
4. **Gravion models are Gravion's.** The model registry contains
   Gravion's own models and variants. Endpoint plumbing is generic —
   a base URL and a key — and is never the product surface.
5. **Licensing is two layers, kept separate.** Legal entitlement
   (contracts) and technical enforcement (cryptography) are designed
   independently and meet only at a reference number.
6. **Enforcement is safe.** The strongest action the technical layer
   can take is *suspension* — stop inference, keep data, keep export.
   There is no kill switch, by design and by contract.
7. **Self-improvement is governed.** Gravion may fix and enhance
   itself only inside a permission the operator granted once, with
   signed changes, a changelog, and rollback.

## 2. The module system ✅

A module is:

- a **manifest** — name, type, version, entrypoint, description; and
- an **entrypoint** — a class with `run(ctx, prompt)`.

`ctx` (ModuleContext) carries the deployment's system context (name +
company policy), the provider, and the model the registry resolved.

Module types (the catalogue in `registry.py`):

| Type | Module | Purpose |
|---|---|---|
| `chat` | Chat bot | General conversation — the reference module ✅ |
| `support` | Support bot | Answers from company material, escalates to humans ✅ (Piece 2) |
| `coder` | Coder | Writes, reviews and explains code ✅ (Piece 2) |
| `website-builder` | Website builder | Generates and iterates full websites from a brief ✅ (Piece 5) |
| `music` | Music | Tracks, jingles, background scores ✅ (Piece 3) |
| `image` | Image | Generation and editing ✅ (Piece 3) |
| `video` | Video | Generation and editing ✅ (Piece 4); long-form (see §7) 🧩 |
| `cyber` | Cyber/security AI | Defensive analysis, triage, reporting assistance ✅ (Piece 6) |
| `agentic` | Agentic tool developer | Builds and runs tool-using agents ✅ (Piece 6) |
| `translation` | Translation | Hindi/English/Hinglish phrasebook translation ✅ |
| `summarization` | Summarization | Local extractive summaries ✅ |
| `retrieval` | Search/retrieval | Sparse-embedding ranking of supplied passages ✅ |
| `data-analysis` | Data analysis | Descriptive statistics from supplied numbers ✅ |
| `document-qa` | Document Q&A | Extractive answers quoted from a supplied document ✅ |
| `all-in-one` | All-in-one | Every module under one deployment ✅ (Piece 8) |

Modules are distributed individually (variant packages) or as the
all-in-one package. In production, a module's files are verified
against its issuer-signed manifest before any of its code runs (§8.3).

## 3. The four tiers ✅ (as configuration) · 🧩 (as trained bundles)

| Tier | Training level | Intended buyer |
|---|---|---|
| **Tier 1 — Complete** | Fully trained, ready | "Just name it, teach it our company and policy, and use it." |
| **Tier 2 — Basic** | Basic training included | A working base to keep training on your own data |
| **Tier 3 — Intermediate** | Intermediate training included | Pre-trained to a level that takes a startup years to reach in-house |
| **Tier 4 — Advanced** | Advanced training included | The full Gravion platform at its strongest, with headroom to go further |

A tier is recorded in the deployment config and in the license; it
governs which model-registry entries and module bundles the deployment
is entitled to. The training pipelines that *produce* each tier's
module weights are later pieces (§10).

## 4. Naming and company policy ✅

Every deployment has a **name** (default **Gravion**) and a **policy
file** (`policy.md` in the deployment home). At runtime the platform
builds one system context —

> "You are {name}, an AI assistant for {company}. … # Company policy
> {policy text}"

— and every module runs under it. Renaming and re-policying a
deployment is a config edit, never a code change. This is the
mechanism behind "teach it your company and work policy": the policy
is data, versioned with the deployment, included in exports, and
covered by the audit story.

## 5. The Gravion model registry ✅

A structured, editable registry of **Gravion's own models**
(`catalog.py` + `data/models.json`). Each entry:

```json
{
  "id": "gravion-video",
  "name": "Gravion Video",
  "capabilities": ["video"],
  "tiers": ["tier-1-complete", "tier-2-basic",
            "tier-3-intermediate", "tier-4-advanced"],
  "status": "planned",
  "context_window": null,
  "notes": "…"
}
```

- **Capabilities** are module types. A module declares the capability
  it needs; `models_for(capability, tier=…)` resolves candidates.
- **Status** is honest: `gravion-core` is `available` (it is the
  built-in engine); `gravion-chat` and `gravion-code` are `available`
  since Piece 2 as first trained bundles at Tier 2, and
  `gravion-vision` and `gravion-audio` since Piece 3, and
  `gravion-video` since Piece 4, and `gravion-builder` since
  Piece 5, and `gravion-shield` and `gravion-agent` since
  Piece 6, on the same
  terms — their registry entries list all four tiers since
  Piece 8 shipped the Tier 4 (Advanced) bundles with Gravion
  Omni, which is itself `available` since Piece 8 as the Tier 4
  flagship serving every capability. Every model in the registry
  is now available.
- **Gravion Omni** is the all-in-one flagship model: every capability,
  Advanced tier.
- The registry is **data, validated at load** with precise errors
  (unknown capability/tier/status, duplicates, missing fields). A
  deployment overrides it with its own `models.json` — that is how
  private Gravion variants are registered without forking the platform.

### 5.1 Measured model specifications

The registry says *which* models exist. A second, generated file
(`data/model_specs.json`, read by `model_spec.py`) says *what each one
is*: its architecture, embedding width, vocabulary and class counts,
storage dtype, quantization method, parameter count split by dtype,
licence, and a SHA-256 digest of the whole specification.

The split matters:

| Field | Meaning | Example |
|---|---|---|
| `dtype` | how a tensor is **stored** | `int8` |
| `quantization` | how a tensor was **quantized** | `symmetric-int8` |
| `parameters_by_dtype` | counted parameters **per** storage dtype | `{"int8": 10575629, "float32": 10914}` |

Keeping them apart stops a quantization method from being mistaken
for a dtype, and keeps a float32 bias vector from being counted as an
int8 weight. Gravion Core is 10,586,543 parameters: 10,575,629 stored
as int8 and quantized symmetrically, 10,914 stored as float32 in the
bias and transition weights. Across the seventeen bundled models the
registry holds 18,696,282 counted parameters.

Every measured field is computed from the shipped weights artifact by
`tools/build_model_specs.py`, never entered by hand, and the test
suite re-derives each spec and fails if the checked-in file drifts
away from the weights. The licence, the specification version and the
optional base-model lineage are the declared part; a base model may
not reference itself and a lineage may not loop.

`gravion models list` shows the shape beside the size, and
`gravion models list --min-parameters N` selects by the counted
parameters; `gravion models spec <id>` prints the whole
specification.

## 6. Inference plumbing ✅

Two ways to reach a model, behind one `ChatProvider` interface:

- **Gravion Core** (`providers.GravionCoreProvider`) — the built-in
  engine. It uses compact local trained weights when available and
  falls back to a deterministic offline responder that proves the
  full path with zero infrastructure.
- **Endpoint** (`providers.EndpointProvider`) — a generic inference
  endpoint the operator runs or rents: base URL + API key from the
  deployment config, a small chat-completions JSON contract
  (`{model, messages}` → `choices[0].message.content`). The `model`
  sent is the Gravion registry id the catalog resolved, so deployments
  keep one vocabulary — Gravion models — regardless of plumbing.

## 7. Long-form video 📐

The video module's headline requirement — up to ~1 hour of
high-quality video — is a **pipeline**, not a single model call:
script → scene plan → per-scene generation with Gravion Video →
consistency pass (characters, style, palette locked across scenes) →
stitching, audio (Gravion Audio) and mastering. The registry entry
for `gravion-video` notes this; the pipeline is the video module's
piece.

## 8. Licensing architecture ✅

### 8.1 Two layers, designed separately

**Legal entitlement** — contracts between the licensor and the
customer: production use requires a signed commercial agreement;
partnership includes a negotiated **10–20% equity** stake; development
and evaluation need no agreement. The agreement contractually commits
to *no destructive kill switch* and to data-export rights on
suspension. See `docs/LICENSING.md` and `LICENSE-COMMERCIAL.md`
(template, pending legal review).

**Technical enforcement** — this codebase. It never interprets the
contract; it checks cryptographic facts. The only place the layers
meet: every license document carries the **agreement reference** and
the **equity percentage** it mirrors, so the technical record cannot
drift from the legal one.

### 8.2 Deployment identity ✅

Each deployment generates an Ed25519 keypair (`licensing/identity.py`).
The public key's fingerprint is the **deployment ID**. The private key
is created on the deployment machine, stored `0600`, and **never
transmitted** — identity is something the deployment *proves*, not
something it *sends*.

### 8.3 Signed licenses and signed modules ✅

- **License** (`licensing/license_doc.py`): a JSON document —
  deployment id, holder, tier, entitled modules, validity window,
  agreement reference + equity % — signed by the issuer key.
  Verification checks signature, issuer, expiry, deployment binding
  and module entitlement. Issuance refuses equity outside 10–20%.
- **Modules** (`licensing/modules_sig.py`): every shippable module
  carries a manifest of SHA-256 file hashes, issuer-signed. Production
  loading verifies signature + every hash first; a tampered module is
  refused and a `module.tamper` audit event is written.

### 8.4 Server-side entitlement ✅ (protocol + reference client)

Local checks can be patched out, so entitlement is anchored at the
licensing authority:

```
deployment ──► POST /v1/entitlement
               { deployment_id, license_hash, module, requested_at }
authority  ──► verifies the license server-side (signature, expiry,
               deployment binding, module entitlement) and returns a
               signed entitlement token (TTL 900s by default)
deployment ──► verifies the token signature, caches it, runs
```

- **Privacy**: the request is exactly those four fields — a key
  fingerprint, a hash, a module name, a timestamp. No prompts, no
  content, no user data, no telemetry. The field set is pinned in code
  (`authority.REQUEST_FIELDS`) and in tests; the authority *rejects*
  requests carrying extra fields.
- **Availability**: a still-valid cached token authorises runs while
  the authority is unreachable. Suspension happens only when neither a
  fresh nor a cached token can authorise the module.
- Piece 1 ships the protocol spec, a reference client, and an
  in-process **mock authority** (the server's logic minus the network)
  used by tests and offline development. Piece 9 adds the operated
  form: `licensing/authority_server.py`, a standard-library HTTP
  service wrapping the same authority logic (`gravion authority
  serve --issuer-key … --licenses DIR`) — its *protocol* is the
  stable contract; where it is publicly hosted remains an
  operations decision.

### 8.5 Tamper-evident audit ✅

Every licensing event (init, gate allow/suspend, module verification
and tamper, export) is appended to a JSONL log where each entry is
hash-chained to its predecessor (`licensing/audit.py`). Editing,
reordering or deleting history breaks the chain at that point, and
`gravion license audit verify` localises the break. The log contains
licensing metadata only — never prompts or content.

### 8.6 Safe suspension and export ✅

States: `development` (no license needed) · `active` · `suspended`.
Suspension means: module runs refuse, and **nothing else happens**.
Data is never deleted, encrypted or held hostage. `gravion export`
works in every state and produces the full bundle: config, policy,
registry, module list, public identity, license, audit log and the
entire data directory. The private identity key is never exported.

## 9. Governed self-upgrade ✅ (implemented, Piece 10)

Gravion may improve itself — bug fixes and enhancements — under a
strict governor, implemented in `upgrades.py` (`gravion upgrade
grant|revoke|status|apply|rollback`):

1. **One-time permission.** The operator grants (or refuses) a standing
   permission once, recorded in config and in the audit log. No grant,
   no self-changes — ever. Revocable at any time.
2. **Signed updates only.** Update packages are issuer-signed like
   modules; unsigned or tampered updates are refused + audited.
3. **Scoped.** The permission covers bug fixes and enhancements to
   existing modules. It never covers licensing terms, tier, policy or
   identity — those change only through the operator and the agreement.
4. **Logged and reversible.** Every self-change is an audit event with
   before/after hashes; the previous version is retained and
   `rollback` restores it. A failed post-update self-check triggers
   automatic rollback.
5. **Transparent.** A human-readable changelog is written into the
   deployment before activation, not after.

## 10. Piece-by-piece roadmap

| Piece | Contents | Status |
|---|---|---|
| **1** | Core scaffold (config, registry, catalog, plumbing), chat module end-to-end, **licensing subsystem complete** | ✅ done |
| **2** | Support + coder modules; Gravion Chat / Gravion Code first trained bundles (Tier 2) | ✅ done |
| 3 | Image + music modules (Gravion Vision, Gravion Audio) | ✅ done |
| 4 | Video module + local render pipeline (Gravion Video); long-form stitching pipeline lands with the higher-tier bundles | ✅ done |
| 5 | Website-builder module (Gravion Builder) | ✅ done |
| 6 | Cyber module (Gravion Shield) + agentic module (Gravion Agent) | ✅ done |
| 7 | Tier bundles: Tier 1 Complete packaging, Tier 3 Intermediate training pipeline | ✅ done (bundles + training job spec/validation; training execution stays with the hosted pipeline) |
| 8 | Gravion Omni + Tier 4 Advanced platform | ✅ done |
| 9 | Hosted licensing authority service (the §8.4 protocol, operated) | ✅ done (service implementation; public hosting is an operations decision) |
| 10 | Governed self-upgrade service (§9) | ✅ done |
| 11 | Training pipelines for customer fine-tuning on deployment data | ✅ done (dataset preparation local; training execution stays with the hosted pipeline) |

## 11. Non-goals for Piece 1

- No trained model weights (Core is the deterministic scaffold engine).
- No hosted services of any kind (authority, updates) — protocols and
  reference implementations only.
- No UI. The CLI is the operator surface until a console piece is
  justified.

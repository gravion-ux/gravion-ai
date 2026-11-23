# gravion-ai

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-116%20passing-brightgreen.svg)](#testing)
[![codecov](https://codecov.io/gh/gravion-ux/gravion-ai/branch/main/graph/badge.svg)](https://codecov.io/gh/gravion-ux/gravion-ai)

**Gravion AI — one modular AI platform, every kind of AI a startup needs, ready to rename, re-policy and run as your own.**

Startups building an AI product today start from zero: months of plumbing
before the first useful answer. Gravion AI is the base they should start
from instead — a single module system that carries many kinds of AI
(chat, support, coding, website building, music, image, video, security,
agentic tools, and an all-in-one package), offered at four tiers of
readiness, under one licensing and identity layer.

You give it a name (the default is **Gravion**), teach it your company
and your work policy, and run it — on the built-in **Gravion Core**
engine, or on any inference endpoint you operate, with your own API key.

> **Piece 3 status — read this before expecting magic.** Gravion AI is
> built in pieces. Piece 1 shipped the *foundation*: the core scaffold,
> the module registry with one module (chat) wired end-to-end, the
> Gravion model registry with starter data, and the complete licensing
> subsystem. Piece 2 adds the **support** and **coder** modules, wired
> end-to-end on the same path, and the first trained bundles of
> **Gravion Chat** and **Gravion Code** at Tier 2 (Basic). Piece 3 adds
> the **image** and **music** modules on the same path — each turns a
> request into a structured brief (mode/kind, aspect, duration, mood)
> for its model — and the first trained bundles of **Gravion Vision**
> and **Gravion Audio** at Tier 2 (Basic). The
> remaining modules and higher-tier bundles are **planned** and land
> piece by piece — the roadmap in [ARCHITECTURE.md](ARCHITECTURE.md)
> says exactly what exists and what does not, and the code refuses to
> pretend otherwise (planned modules raise instead of faking output).

---

## The four tiers

Tiers are readiness levels over the *same* modules — what changes is
how trained each module is when it reaches you.

| Tier | Name | What you get |
|---|---|---|
| 1 | **Complete** | A complete, ready AI. Name it, teach it your company + policy, use it. |
| 2 | **Basic** | Basic-trained variants of every module — a working base you keep training on your own data. |
| 3 | **Intermediate** | Already trained to a level that takes a startup years to reach in-house. |
| 4 | **Advanced** | The top tier: advanced-trained modules on the full Gravion platform, headroom to go further. |

Every tier: renameable (default name *Gravion*), company-policy
injection, your own endpoint + API key if you want one, and **governed
self-upgrade** — Gravion may propose bug fixes and enhancements, but it
applies them only after your one-time explicit permission, with every
change logged and reversible.

## Modules

| Module | Type | Status |
|---|---|---|
| chat | Chat bot | ✅ implemented (reference module) |
| support | Support bot | ✅ implemented (Piece 2) |
| coder | Coding assistant | ✅ implemented (Piece 2) |
| website-builder | Website builder | planned |
| music | Music generation | ✅ implemented (Piece 3) |
| image | Image generation/editing | ✅ implemented (Piece 3) |
| video | Video generation — incl. long-form up to ~1 hour, high quality | planned |
| cyber | Cyber/security AI | planned |
| agentic | Agentic tool developer | planned |
| all-in-one | Every module, one deployment | planned |

Modules are sold/used individually (variant packages) or together
(all-in-one). Each module declares the capability it needs; the
**Gravion model registry** resolves which Gravion model serves it.

## The Gravion model registry

Gravion's own models, registered as data (`src/gravion_ai/data/models.json`),
not hardcoded logic: **Gravion Core** (the built-in offline engine —
*available now*), **Gravion Chat** and **Gravion Code** (*available
since Piece 2* as Tier 2 first trained bundles), **Gravion Vision** and
**Gravion Audio** (*available since Piece 3* as Tier 2 first trained
bundles), Gravion Builder,
Gravion Video, Gravion Shield, Gravion
Agent, and **Gravion Omni**, the all-in-one flagship at the Advanced tier.
Deployments can register private variants by dropping their own
`models.json` into the deployment home — same schema, same validation.

```bash
gravion models list                     # the whole registry
gravion models list --capability video  # only models that serve video
```

## Quickstart

Requires Python 3.10+.

```bash
git clone https://github.com/gravion-ux/gravion-ai.git
cd gravion-ai
pip install -e .

gravion init --name Gravion --company "Acme Robotics"
gravion status
gravion chat "hello, what are you?"
gravion support "where is my invoice?"
gravion export --out my-deployment.zip
```

Development mode needs **no license** and runs fully offline on
Gravion Core. Edit `policy.md` in the deployment home
(`~/.gravion-ai` by default, or `$GRAVION_HOME`) — that file is your
company policy, injected into every module's system context.

## Licensing — the short version

Gravion AI is built to be a commercial platform, and its licensing is
designed in two separate layers (full detail: [docs/LICENSING.md](docs/LICENSING.md)):

- **Legal entitlement.** Development and evaluation are free. Running
  Gravion in **production** requires a signed commercial agreement,
  which includes a negotiated **10–20% equity partnership**. The
  agreement commits Gravion to **no destructive kill switch**: the
  worst outcome of a lapsed agreement is suspension, never data loss.
- **Technical enforcement.** Every deployment has a cryptographic
  identity (Ed25519). Licenses and module manifests are issuer-signed
  and verified before anything runs in production. A server-side
  entitlement protocol (four fields only — deployment id, license
  hash, module, timestamp; no user data, ever) issues short-lived
  signed tokens, with a signed cache for offline grace. Every
  licensing event lands in a **tamper-evident, hash-chained audit
  log**. Suspended? `gravion export` still gives you everything —
  config, policy, registry, audit log and all your data.

```bash
gravion license identity        # this deployment's cryptographic identity
gravion license audit verify    # check the audit chain
```

## Repository layout

```
src/gravion_ai/
  config.py       deployment config, naming, company-policy injection
  registry.py     the module system (manifest + entrypoint, catalogue)
  catalog.py      the Gravion model registry (loader + validation)
  data/models.json  the registry's starter data
  providers.py    Gravion Core (built-in) + generic endpoint plumbing
  runtime.py      gate → verify → resolve → run; export
  modules/chat/   the reference module, wired end-to-end
  modules/support/  support bot: policy-grounded answers, human escalation
  modules/coder/    coding assistant: generate / review / explain tasks
  modules/image/    image generation/editing: generate / edit briefs
  modules/music/    music generation: track / jingle / score briefs
  licensing/      identity, signed licenses, signed modules, audit log,
                  entitlement protocol + gate — the licensing subsystem
docs/LICENSING.md       licensing design + protocol specification
LICENSE-COMMERCIAL.md   commercial agreement template (summary terms)
ARCHITECTURE.md         the full vision and piece-by-piece roadmap
```

## Testing

```bash
pip install -e ".[dev]"
python -m pytest -q
```

168 tests cover the licensing subsystem end-to-end (identity, signed
licenses, tamper rejection, audit-chain verification, the entitlement
protocol and its pinned privacy field set, gate states, suspension,
export-while-suspended), the Piece 2 support and coder modules
(grounding addendum, human escalation that never reaches the model,
coder task detection, Tier 2 bundle resolution, production runs of
both modules under signed manifests), the Piece 3 image and music
modules (mode/kind detection, aspect/duration/mood brief parsing,
Tier 2 bundle resolution for Gravion Vision and Gravion Audio), plus
config, providers, registry, catalog and the CLI.

## Author

Priyanshu Kumar Thakur — [github.com/gravion-ux](https://github.com/gravion-ux)

## Licence

Source code: MIT (see [LICENSE](LICENSE)). Production deployments are
additionally governed by the commercial agreement —
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md).

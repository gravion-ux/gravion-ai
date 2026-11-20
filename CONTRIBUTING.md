# Contributing to Gravion AI

Thanks for your interest in Gravion AI — a modular AI platform framework:
many AI types, four tiers, one module system, with a cryptographic
licensing and entitlement layer for commercial deployments.

Contributions of every size are welcome: bug reports, documentation,
tests, new dataset entries for the model registry, and code. This file
explains how to get set up and what a mergeable pull request looks like.

## What we are looking for

- **Bug fixes** with a failing test that proves the bug existed.
- **Tests** for behaviour that is currently untested, especially around
  the licensing subsystem (identity, signed licenses, audit chain,
  entitlement gate) and the module runtime path.
- **Documentation and examples** — clearer docs, worked examples, a
  second example company policy next to `examples/policy.md`.
- **Model registry data** — corrections and additions to
  `src/gravion_ai/data/models.json` (Gravion's own models and variants
  only; the registry is data, not hardcoded logic).
- **Module work** that follows the roadmap in
  [ARCHITECTURE.md](ARCHITECTURE.md). Modules land piece by piece, and
  the code is deliberately honest about status: planned modules raise
  instead of faking output. Please keep it that way — never make a
  planned capability pretend to work.

## Ground rules

- **Licensing is load-bearing.** The licensing subsystem (signed
  licenses, signed module manifests, the entitlement gate, the
  hash-chained audit log) is the commercial core of this project.
  Changes that weaken verification, skip the gate, or alter the audit
  chain need an issue and a design discussion *before* any code.
- **Development mode stays free and offline.** Dev mode needs no
  license and runs fully offline on the built-in Gravion Core engine.
  Do not add anything that phones home or requires an account in dev
  mode.
- **One runtime dependency.** The only runtime dependency is
  `cryptography` (it provides the Ed25519 signatures the licensing
  subsystem is built on). Everything else is the Python standard
  library. Do not add new runtime dependencies without discussion.
- **No private keys, ever.** Never commit issuer keys, deployment
  identities, or license files. The bundled module manifests ship
  unsigned on purpose — signing is the project owner's step.

## Development setup

Requires Python 3.10 or newer.

```bash
git clone https://github.com/gravion-ux/gravion-ai.git
cd gravion-ai
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The package uses a `src/` layout (`src/gravion_ai/`). A `conftest.py`
at the repo root puts `src/` on the path, so the test suite also runs
from a bare checkout without installing.

## Running the tests

```bash
python -m pytest -q
```

With coverage (this is what CI runs):

```bash
python -m pytest -q --cov=gravion_ai --cov-report=xml
```

Smoke-test the CLI end to end (development mode needs no license):

```bash
export GRAVION_HOME=$(mktemp -d)
gravion init --name Gravion
gravion status
gravion modules list
gravion chat "hello gravion"
```

## Code style

Match the code that is already here:

- Every module starts with a docstring that explains *what it is for*
  and any deliberate design constraints (read
  `src/gravion_ai/modules/chat/module.py` or the licensing package for
  the flavour).
- `from __future__ import annotations`, type hints on public functions,
  double quotes, small focused functions.
- Comments explain *why*, especially around licensing and security
  decisions — the next reader should never have to guess why a check
  exists.
- Tests are plain pytest functions with descriptive names, in
  `tests/test_<area>.py`. Keep them deterministic: fixed timestamps,
  temporary directories, no network.

## Proposing a bigger change

For anything larger than a small fix — a new module, a registry schema
change, anything touching `src/gravion_ai/licensing/` — **open an issue
first** and outline what you want to build and why. The roadmap in
ARCHITECTURE.md is sequenced piece by piece; an issue keeps your work
aligned with the piece it belongs to and saves you from building
something that cannot be merged yet.

## Pull request workflow

1. Fork the repository and branch from `main`:
   `feat/<short-name>`, `fix/<short-name>`, `docs/<short-name>` or
   `tests/<short-name>`.
2. Make the change, with tests. Run the full suite and the CLI smoke
   test locally.
3. Open the pull request against `main`. In the description say:
   what changed, why, how you tested it, and which roadmap piece it
   belongs to (if any).
4. CI must be green — it runs the suite on Python 3.10, 3.11 and 3.12
   plus the CLI smoke test.
5. One logical change per pull request. Small PRs get reviewed (and
   merged) fastest.

## Licence note for contributors

The source code is MIT (see [LICENSE](LICENSE)). Production
deployments are additionally governed by the commercial agreement in
[LICENSE-COMMERCIAL.md](LICENSE-COMMERCIAL.md) — by contributing you
agree that your contribution is provided under the MIT licence for the
source code.

Questions? Open an issue — asking first is always cheaper than
rebuilding later.

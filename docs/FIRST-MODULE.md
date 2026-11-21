# Building Your First Gravion AI Module

Gravion AI is built around a single module system (`src/gravion_ai/registry.py`): every AI capability—whether `chat`, `support`, `coder`, or a future specialist module—is a package with a `manifest.json` and a Python class implementing `run(ctx, prompt)`.

This guide walks through how existing modules work, the rules every module must follow, and a complete worked example (`echo`) you can register, list via `gravion modules list`, and verify with `pytest`.

---

## 1. Anatomy of a Module Directory

Each built-in module lives in its own subpackage under `src/gravion_ai/modules/<name>/`. Look at the reference `chat` module (`src/gravion_ai/modules/chat/`):

```text
src/gravion_ai/modules/chat/
  __init__.py      # Package marker (module docstring)
  module.py        # Entrypoint class implementing run(ctx, prompt)
  manifest.json    # Module metadata, file SHA-256 hashes, and issuer signature
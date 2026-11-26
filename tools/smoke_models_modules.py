#!/usr/bin/env python3
"""Smoke test: every model in models.json + every module entry point.

Author: Priyanshu Kumar Thakur
Run: .venv/bin/python tools/smoke_models_modules.py
Exit 0 = all pass, 1 = any fail.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from gravion_ai.catalog import ModelCatalog  # noqa: E402
from gravion_ai.core_model import load_bundled_model  # noqa: E402
from gravion_ai.providers import GravionCoreProvider  # noqa: E402
from gravion_ai.registry import ModuleContext, default_registry  # noqa: E402

MODULE_PROMPTS = {
    "chat": "Hello, what can you do?",
    "support": "What are your opening hours?",
    "coder": "Write a python function that adds two numbers",
    "image": "Draw a square logo for a bakery",
    "music": "Compose a calm 30 seconds track",
    "video": "Create a vertical 15 seconds video of a sunrise",
    "website-builder": "Build a landing page for a bakery, pages: home, about",
    "cyber": "Triage: alert from 10.0.0.8 severity: high",
    "agentic": "Plan an agent that checks email and drafts replies",
    "all-in-one": "coder: write a python function that adds two numbers",
}

# model id -> a prompt in its capability area
MODEL_PROMPTS = {
    "gravion-core": "Hello",
    "gravion-chat": "Hello, how are you?",
    "gravion-support": "check the status of an order",
    "gravion-review": "review this code for bugs",
    "gravion-code": "Write a python function",
    "gravion-builder": "Build a landing page",
    "gravion-vision": "Draw a logo",
    "gravion-video": "Create a short video",
    "gravion-audio": "Compose a track",
    "gravion-shield": "Analyse this alert",
    "gravion-agent": "Plan an agent",
    "gravion-omni": "Hello",
}


def main() -> int:
    failures: list[str] = []
    catalog = ModelCatalog.load()
    provider = GravionCoreProvider()
    data_dir = ROOT / "src" / "gravion_ai" / "data"

    print("== Models ==")
    for entry in catalog.list():
        try:
            got = catalog.get(entry.id)
            assert got is not None and got.id == entry.id, "catalog get failed"
            # weights artifact + sha + param count computed from stored weights
            if entry.weights_included:
                assert entry.weights_artifact, "missing weights_artifact"
                wpath = ROOT / "src" / "gravion_ai" / entry.weights_artifact
                assert wpath.exists(), f"weights file missing: {wpath}"
                loaded = load_bundled_model(entry.id)
                # catalog weights_sha256 is the artifact-internal weights
                # hash (core_model._weights_sha256), not the file-bytes hash.
                assert loaded.weights_sha256 == entry.weights_sha256, (
                    f"weights sha mismatch {loaded.weights_sha256} != {entry.weights_sha256}"
                )
                assert loaded is not None, "load_bundled_model returned None"
                assert loaded.parameter_count == entry.parameter_count, (
                    f"param count mismatch loaded={loaded.parameter_count} catalog={entry.parameter_count}"
                )
            # resolution via provider complete()
            prompt = MODEL_PROMPTS.get(entry.id, "Hello")
            out = provider.complete(
                [{"role": "system", "content": "You are Gravion."},
                 {"role": "user", "content": prompt}],
                model=entry.id,
            )
            assert isinstance(out, str) and out.strip(), "empty provider output"
            print(f"PASS model {entry.id} params={entry.parameter_count} out_len={len(out)}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"model {entry.id}: {exc}")
            print(f"FAIL model {entry.id}: {exc}")

    print("\n== Modules ==")
    registry = default_registry()
    # map capability -> first serving model id for context
    for manifest in registry.list():
        name = manifest.name
        try:
            module = registry.instantiate(name)
            # pick a model serving this module type (all-in-one needs omni)
            if name == "all-in-one":
                model_id = "gravion-omni"
            else:
                serving = catalog.models_for(manifest.type.value)
                assert serving, f"no model serves {manifest.type.value}"
                model_id = serving[0].id
            ctx = ModuleContext("You are Gravion. Be helpful.", provider, model=model_id)
            prompt = MODULE_PROMPTS.get(name, "Hello")
            out = module.run(ctx, prompt)
            assert isinstance(out, str) and out.strip(), "empty module output"
            print(f"PASS module {name} model={model_id} out_len={len(out)}")
        except Exception as exc:  # noqa: BLE001
            failures.append(f"module {name}: {exc}")
            print(f"FAIL module {name}: {exc}")

    print(f"\n{'ALL PASS' if not failures else 'FAILURES: ' + '; '.join(failures)}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

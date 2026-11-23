"""The deployment runtime: wires config, registry, catalog, provider
and the entitlement gate together.

One path matters most — running a module:

    gate.require(module)                (licensing, production only)
    → verify signed module manifest     (production only)
    → resolve the model via the registry
    → module.run(ctx)                   (system context = name + policy)

Export is the mirror image of the gate: it is *never* gated. A
suspended deployment can always take its data out.
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from .catalog import ModelCatalog
from .config import MODE_PRODUCTION, GravionConfig
from .errors import ModuleError, ModuleTamperError
from .licensing import AuditLog, EntitlementGate
from .licensing.authority import EntitlementClient, HttpAuthorityClient
from .licensing.keys import KeyPair
from .licensing.modules_sig import verify_module
from .providers import build_provider
from .registry import ModuleContext, ModuleRegistry, default_registry

ISSUER_PUBLIC_FILENAME = "issuer.pub"


def load_issuer_public(config: GravionConfig) -> KeyPair | None:
    """The issuer/authority public key, if the deployment has one
    installed at state/issuer.pub."""
    path = config.state_dir / ISSUER_PUBLIC_FILENAME
    if path.exists():
        return KeyPair.from_public_pem(path.read_bytes())
    return None


class GravionRuntime:
    def __init__(
        self,
        config: GravionConfig,
        *,
        issuer_public: KeyPair | None = None,
        client: EntitlementClient | None = None,
        registry: ModuleRegistry | None = None,
        now_fn=None,
    ):
        self.config = config
        self.audit = AuditLog(config.state_dir / "audit.jsonl")
        self.registry = registry or default_registry()
        self.catalog = ModelCatalog.load_for_home(config.home)
        self.provider = build_provider(config.provider)
        if issuer_public is None:
            issuer_public = load_issuer_public(config)
        if client is None and config.authority_url:
            client = HttpAuthorityClient(config.authority_url)
        self.gate = EntitlementGate(
            config,
            issuer_public=issuer_public,
            client=client,
            audit=self.audit,
            now_fn=now_fn,
        )

    # -- model resolution --------------------------------------------------
    def resolve_model(self, capability: str = "chat") -> str:
        """Resolve the configured model through the Gravion registry.

        If the configured model id is a registry entry, it must serve
        the requested capability. Deployments may also name a custom
        model id their own endpoint understands (registered privately
        or passed through as-is).
        """
        wanted = self.config.provider.model
        entry = self.catalog.get(wanted)
        if entry is not None:
            if not entry.serves(capability):
                raise ModuleError(
                    f"Gravion model {entry.id!r} does not serve capability "
                    f"{capability!r} (serves: {sorted(entry.capabilities)})"
                )
            return entry.id
        return wanted

    # -- module execution ----------------------------------------------------
    def run_module(self, name: str, prompt: str) -> str:
        self.gate.require(name)
        manifest = self.registry.get(name)
        if self.config.mode == MODE_PRODUCTION and manifest.path is not None:
            issuer_public = load_issuer_public(self.config)
            if issuer_public is None:
                raise ModuleTamperError(
                    "production mode requires the issuer public key at "
                    f"state/{ISSUER_PUBLIC_FILENAME} to verify module signatures"
                )
            try:
                verify_module(manifest.path, issuer_public)
            except ModuleTamperError as exc:
                self.audit.append(
                    "module.tamper", {"module": name, "reason": str(exc)}
                )
                raise
            self.audit.append("module.verified", {"module": name})
        module = self.registry.instantiate(name)
        ctx = ModuleContext(
            system_context=self.config.system_context(),
            provider=self.provider,
            model=self.resolve_model(name),
        )
        return module.run(ctx, prompt)

    def chat(self, prompt: str) -> str:
        return self.run_module("chat", prompt)

    def support(self, prompt: str) -> str:
        return self.run_module("support", prompt)

    def coder(self, prompt: str) -> str:
        return self.run_module("coder", prompt)

    def image(self, prompt: str) -> str:
        return self.run_module("image", prompt)

    def music(self, prompt: str) -> str:
        return self.run_module("music", prompt)

    # -- status ----------------------------------------------------------------
    def status(self) -> dict:
        decision = self.gate.check("chat")
        return {
            "name": self.config.name,
            "company": self.config.company,
            "mode": self.config.mode,
            "tier": self.config.tier,
            "provider": self.config.provider.type,
            "model": self.config.provider.model,
            "gate_state": decision.state.value,
            "gate_reason": decision.reason,
            "modules": [
                {"name": m.name, "type": m.type.value, "status": m.status.value}
                for m in self.registry.list()
            ],
        }

    # -- export ------------------------------------------------------------------
    def export_bundle(self, out_path: Path) -> list[str]:
        """Write a full export bundle (zip) and return its file list.

        Available in every gate state — including SUSPENDED. Contains
        configuration, policy, the model registry in use, the module
        list, public identity material, the license (if installed),
        the audit log, and everything under data/. The deployment's
        *private* identity key is never exported.
        """
        cfg = self.config
        self.audit.append("export.bundle", {"out": str(out_path)})
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        included: list[str] = []

        def add_file(source: Path, arcname: str) -> None:
            if source.exists():
                archive.write(source, arcname)
                included.append(arcname)

        with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as archive:
            add_file(cfg.config_path, "gravion.json")
            add_file(cfg.policy_path, "policy.md")
            override = cfg.home / "models.json"
            if override.exists():
                add_file(override, "models.json")
            else:
                from importlib import resources

                raw = resources.files("gravion_ai").joinpath(
                    "data", "models.json"
                ).read_bytes()
                archive.writestr("models.json", raw)
                included.append("models.json")
            modules_doc = {
                "modules": [
                    {
                        "name": m.name,
                        "type": m.type.value,
                        "version": m.version,
                        "status": m.status.value,
                        "description": m.description,
                    }
                    for m in self.registry.list()
                ]
            }
            archive.writestr(
                "modules.json", json.dumps(modules_doc, indent=2) + "\n"
            )
            included.append("modules.json")
            add_file(cfg.state_dir / "identity.json", "state/identity.json")
            add_file(cfg.state_dir / "identity.pub", "state/identity.pub")
            add_file(cfg.state_dir / "license.json", "state/license.json")
            add_file(cfg.state_dir / "audit.jsonl", "state/audit.jsonl")
            if cfg.data_dir.exists():
                for path in sorted(cfg.data_dir.rglob("*")):
                    if path.is_file():
                        arcname = str(Path("data") / path.relative_to(cfg.data_dir))
                        archive.write(path, arcname)
                        included.append(arcname)
        return sorted(included)

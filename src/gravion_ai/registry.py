"""The module system.

A *module* is the unit of Gravion AI: a manifest (name, type, version,
entrypoint) plus the code behind the entrypoint. Startups pick modules
individually (variant packages) or take the whole set (the all-in-one
package); the four tiers describe how trained each module is when it
ships, not different code.

Piece 1 implements one module end-to-end (chat); Piece 2 adds support
and coder. The other module types are registered as *planned* so the
catalogue, tiers and licensing all speak about them honestly from day
one; their implementations land in later pieces (see ARCHITECTURE.md
roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Protocol, Sequence

from .errors import ModuleError
from .providers import Message


class ModuleType(str, Enum):
    CHAT = "chat"
    SUPPORT = "support"
    CODER = "coder"
    WEBSITE_BUILDER = "website-builder"
    MUSIC = "music"
    IMAGE = "image"
    VIDEO = "video"
    CYBER = "cyber"
    AGENTIC = "agentic"
    ALL_IN_ONE = "all-in-one"


class ModuleStatus(str, Enum):
    IMPLEMENTED = "implemented"
    PLANNED = "planned"


@dataclass(frozen=True)
class ModuleManifest:
    name: str
    type: ModuleType
    version: str
    entrypoint: str  # "package.module:ClassName"
    description: str = ""
    status: ModuleStatus = ModuleStatus.PLANNED
    # Directory holding the module's files + signed manifest.json, for
    # filesystem modules whose integrity is verified before loading.
    path: Path | None = field(default=None, compare=False)


class ModuleContext:
    """What a module runs with: the deployment's system context (name +
    company policy), its provider, and the model the catalog resolved
    for it (None = provider default)."""

    def __init__(self, system_context: str, provider, model: str | None = None):
        self.system_context = system_context
        self.provider = provider
        self.model = model

    def chat(self, prompt: str, history: Sequence[Message] = ()) -> str:
        messages = [{"role": "system", "content": self.system_context}]
        messages.extend(history)
        messages.append({"role": "user", "content": prompt})
        return self.provider.complete(messages, model=self.model)


class BaseModule(Protocol):
    manifest: ModuleManifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        ...


def _load_entrypoint(entrypoint: str):
    module_path, _, attr = entrypoint.partition(":")
    if not module_path or not attr:
        raise ModuleError(
            f"entrypoint must look like 'package.module:Class', got {entrypoint!r}"
        )
    import importlib

    module = importlib.import_module(module_path)
    try:
        return getattr(module, attr)
    except AttributeError as exc:
        raise ModuleError(f"entrypoint {entrypoint!r} not found: {exc}") from exc


class ModuleRegistry:
    def __init__(self) -> None:
        self._modules: dict[str, ModuleManifest] = {}

    def register(self, manifest: ModuleManifest) -> None:
        if manifest.name in self._modules:
            raise ModuleError(f"module {manifest.name!r} already registered")
        self._modules[manifest.name] = manifest

    def get(self, name: str) -> ModuleManifest:
        try:
            return self._modules[name]
        except KeyError:
            raise ModuleError(
                f"unknown module {name!r}; known: {sorted(self._modules)}"
            ) from None

    def list(self) -> list[ModuleManifest]:
        return [self._modules[name] for name in sorted(self._modules)]

    def by_type(self, module_type: ModuleType) -> list[ModuleManifest]:
        return [m for m in self.list() if m.type == module_type]

    def instantiate(self, name: str) -> BaseModule:
        manifest = self.get(name)
        if manifest.status is not ModuleStatus.IMPLEMENTED:
            raise ModuleError(
                f"module {name!r} is planned for a later piece and has no "
                f"implementation yet"
            )
        cls = _load_entrypoint(manifest.entrypoint)
        return cls(manifest)


def default_registry() -> ModuleRegistry:
    """The catalogue: chat (Piece 1) plus support and coder (Piece 2)
    implemented, the rest planned."""
    from .modules import chat as chat_module_pkg
    from .modules import coder as coder_module_pkg
    from .modules import support as support_module_pkg

    registry = ModuleRegistry()
    registry.register(
        ModuleManifest(
            name="chat",
            type=ModuleType.CHAT,
            version="0.1.0",
            entrypoint="gravion_ai.modules.chat.module:ChatModule",
            description="General-purpose chat bot — the reference module.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(chat_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="support",
            type=ModuleType.SUPPORT,
            version="0.1.0",
            entrypoint="gravion_ai.modules.support.module:SupportModule",
            description="Customer support bot: answers from company docs and policy, escalates to humans.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(support_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="coder",
            type=ModuleType.CODER,
            version="0.1.0",
            entrypoint="gravion_ai.modules.coder.module:CoderModule",
            description="Coding assistant: writes, reviews and explains code in your stack.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(coder_module_pkg.__file__).parent,
        )
    )
    planned = [
        ("website-builder", ModuleType.WEBSITE_BUILDER, "Website builder: generates and iterates full sites from a brief."),
        ("music", ModuleType.MUSIC, "Music generation: tracks, jingles and background scores."),
        ("image", ModuleType.IMAGE, "Image generation and editing."),
        ("video", ModuleType.VIDEO, "Video generation, including long-form up to ~1 hour in high quality."),
        ("cyber", ModuleType.CYBER, "Cyber/security AI: defensive analysis, triage and reporting assistance."),
        ("agentic", ModuleType.AGENTIC, "Agentic tool developer: builds and runs tool-using agents for repeatable work."),
        ("all-in-one", ModuleType.ALL_IN_ONE, "The all-in-one package: every module above under one deployment."),
    ]
    for name, mtype, desc in planned:
        registry.register(
            ModuleManifest(
                name=name,
                type=mtype,
                version="0.0.0",
                entrypoint="",
                description=desc,
                status=ModuleStatus.PLANNED,
            )
        )
    return registry

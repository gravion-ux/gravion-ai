"""The module system.

A *module* is the unit of Gravion AI: a manifest (name, type, version,
entrypoint) plus the code behind the entrypoint. Startups pick modules
individually (variant packages) or take the whole set (the all-in-one
package); the four tiers describe how trained each module is when it
ships, not different code.

Piece 1 implements one module end-to-end (chat); Piece 2 adds support
and coder; Piece 3 adds image and music; Piece 4 adds video; Piece 5
adds website-builder; Piece 6 adds cyber and agentic; Piece 8
adds all-in-one. The other
module types are registered as *planned* so the catalogue, tiers and
licensing all speak about them honestly from day one; their
implementations land in later pieces (see ARCHITECTURE.md roadmap).
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
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    RETRIEVAL = "retrieval"
    DATA_ANALYSIS = "data-analysis"
    DOCUMENT_QA = "document-qa"
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
    """The catalogue: chat (Piece 1), support and coder (Piece 2),
    image and music (Piece 3), video (Piece 4), website-builder
    (Piece 5), cyber and agentic (Piece 6) and all-in-one
    (Piece 8) — every module implemented."""
    from .modules import agentic as agentic_module_pkg
    from .modules import all_in_one as all_in_one_module_pkg
    from .modules import chat as chat_module_pkg
    from .modules import coder as coder_module_pkg
    from .modules import cyber as cyber_module_pkg
    from .modules import data_analysis as data_analysis_module_pkg
    from .modules import document_qa as document_qa_module_pkg
    from .modules import image as image_module_pkg
    from .modules import music as music_module_pkg
    from .modules import retrieval as retrieval_module_pkg
    from .modules import summarization as summarization_module_pkg
    from .modules import support as support_module_pkg
    from .modules import translation as translation_module_pkg
    from .modules import video as video_module_pkg
    from .modules import website_builder as website_builder_module_pkg

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
    registry.register(
        ModuleManifest(
            name="image",
            type=ModuleType.IMAGE,
            version="0.1.0",
            entrypoint="gravion_ai.modules.image.module:ImageModule",
            description="Image generation and editing: turns a request into a structured brief for Gravion Vision.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(image_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="music",
            type=ModuleType.MUSIC,
            version="0.1.0",
            entrypoint="gravion_ai.modules.music.module:MusicModule",
            description="Music generation: tracks, jingles and background scores, briefed for Gravion Audio.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(music_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="video",
            type=ModuleType.VIDEO,
            version="0.1.0",
            entrypoint="gravion_ai.modules.video.module:VideoModule",
            description="Video generation and editing: turns a request into a structured brief for Gravion Video, with a local short-clip render pipeline.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(video_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="website-builder",
            type=ModuleType.WEBSITE_BUILDER,
            version="0.1.0",
            entrypoint="gravion_ai.modules.website_builder.module:WebsiteBuilderModule",
            description="Website builder: generates and iterates full sites from a brief, with a local static-site builder.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(website_builder_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="cyber",
            type=ModuleType.CYBER,
            version="0.1.0",
            entrypoint="gravion_ai.modules.cyber.module:CyberModule",
            description="Cyber/security AI: defensive analysis, triage and reporting assistance, with deterministic indicator extraction.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(cyber_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="agentic",
            type=ModuleType.AGENTIC,
            version="0.1.0",
            entrypoint="gravion_ai.modules.agentic.module:AgenticModule",
            description="Agentic tool developer: builds and runs tool-using agents for repeatable work, with a local plan writer.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(agentic_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="translation",
            type=ModuleType.TRANSLATION,
            version="0.1.0",
            entrypoint="gravion_ai.modules.translation.module:TranslationModule",
            description="Translation: Hindi, English and Hinglish phrasebook translation with a Gravion-owned glossary, briefed for Gravion Translate.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(translation_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="summarization",
            type=ModuleType.SUMMARIZATION,
            version="0.1.0",
            entrypoint="gravion_ai.modules.summarization.module:SummarizationModule",
            description="Summarization: local extractive summaries that keep the source's own sentences, briefed for Gravion Summary.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(summarization_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="retrieval",
            type=ModuleType.RETRIEVAL,
            version="0.1.0",
            entrypoint="gravion_ai.modules.retrieval.module:RetrievalModule",
            description="Search and retrieval: sparse embedding ranking of supplied passages, briefed for Gravion Search.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(retrieval_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="data-analysis",
            type=ModuleType.DATA_ANALYSIS,
            version="0.1.0",
            entrypoint="gravion_ai.modules.data_analysis.module:DataAnalysisModule",
            description="Data analysis: descriptive statistics and trend checks computed locally from supplied numbers, briefed for Gravion Analyst.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(data_analysis_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="document-qa",
            type=ModuleType.DOCUMENT_QA,
            version="0.1.0",
            entrypoint="gravion_ai.modules.document_qa.module:DocumentQAModule",
            description="Document Q&A: extractive answers quoted from a supplied document, briefed for Gravion DocQA.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(document_qa_module_pkg.__file__).parent,
        )
    )
    registry.register(
        ModuleManifest(
            name="all-in-one",
            type=ModuleType.ALL_IN_ONE,
            version="0.1.0",
            entrypoint="gravion_ai.modules.all_in_one.module:AllInOneModule",
            description="The all-in-one package: every module above under one deployment, routing by directive, served by Gravion Omni.",
            status=ModuleStatus.IMPLEMENTED,
            path=Path(all_in_one_module_pkg.__file__).parent,
        )
    )
    return registry

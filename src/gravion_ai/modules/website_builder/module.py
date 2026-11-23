"""WebsiteBuilderModule — website generation and iteration (Piece 5).

The website-builder module turns a plain-language brief into a
structured **website brief** for Gravion Builder, following the same
shape as the other modules (manifest in, `run(ctx, prompt)` out) and
adding deterministic logic of its own:

1. **Mode detection.** A request is either *build* (a new site from
   a brief) or *iterate* (change an existing site), detected from
   explicit phrasing exactly as the image module detects edits — an
   "update:" / "revise:" style prefix, or a request to update /
   change / edit *this* or *the* site.

2. **Brief parsing.** Three pieces of intent are extracted: the
   **site type** (landing page, portfolio, blog, online store,
   documentation, restaurant, agency, product/SaaS — from explicit
   keywords, in a fixed order); the **pages** (an explicit
   "pages: home, about, pricing" list wins; otherwise pages named
   in the brief — "a pricing page", "an about page" — are
   collected, with home always first); and the **style** (dark,
   light, minimal, bold or modern). Anything not stated is recorded
   in the brief as the model default rather than invented.

The brief is appended to the deployment's system context together
with the mode guidance. The module is served by **Gravion
Builder**, whose first trained bundle ships at Tier 2 in this
piece (see data/models.json); the runtime's registry check refuses
a model that does not serve the website-builder capability instead
of silently answering with the wrong model.

`run` is the model path. `build` is the development path: it hands
the brief to the local site builder (see builder.py), which writes
a real static website (HTML + CSS + JS) offline — honestly
labelled as the local builder, never as the trained model.
"""

from __future__ import annotations

import re
from pathlib import Path

from ...registry import ModuleContext, ModuleManifest

MODE_BUILD = "build"
MODE_ITERATE = "iterate"

MODE_GUIDANCE = {
    MODE_BUILD: """

# Website mode: build
Build the website described in the brief below. Follow the
deployment's brand and policy context above — colours, tone and
content rules there govern every site this deployment produces.
Produce the full site: every page in the brief, consistent
navigation and styling across pages, real content written for the
company in the brief — never lorem ipsum.""",
    MODE_ITERATE: """

# Website mode: iterate
Change the existing website as described in the brief below.
Change only what the request asks to change; preserve everything
else — pages, structure and the deployment's brand elements —
exactly as it is. Follow the deployment's brand and policy context
above.""",
}

_ITERATE_PREFIXES = (
    "iterate:",
    "iterate on",
    "update:",
    "update this",
    "update the",
    "please update",
    "revise:",
    "revise this",
    "revise the",
    "please revise",
    "change:",
    "change the",
    "edit:",
    "edit this",
    "edit the",
    "modify:",
    "modify the",
)

# Site types, in the order they are checked: the first keyword set
# that matches wins, so more specific types (online store) are
# checked before generic ones.
_SITE_TYPES = (
    (("online store", "e-commerce", "ecommerce", "shop"), "online store"),
    (("landing page", "landing"), "landing page"),
    (("portfolio",), "portfolio"),
    (("documentation", "docs site", "docs"), "documentation"),
    (("restaurant", "cafe", "café"), "restaurant"),
    (("blog",), "blog"),
    (("agency",), "agency"),
    (("saas", "startup", "app"), "product"),
)

# Pages the brief can name, in the order they are collected after
# the home page. "home" itself is always first and never repeated.
_NAMED_PAGES = ("about", "pricing", "features", "blog", "gallery", "contact")

_PAGES_LIST_RE = re.compile(r"pages?\s*:\s*([a-z0-9 ,\-]+)", re.IGNORECASE)

_STYLES = ("dark", "light", "minimal", "bold", "modern")


def detect_mode(prompt: str) -> str:
    """Classify a website prompt as build / iterate."""
    text = prompt.strip().lower()
    if text.startswith(_ITERATE_PREFIXES):
        return MODE_ITERATE
    return MODE_BUILD


def parse_site_type(prompt: str) -> str | None:
    """Extract the site type from explicit keywords, or None."""
    text = prompt.lower()
    for keywords, site_type in _SITE_TYPES:
        if any(keyword in text for keyword in keywords):
            return site_type
    return None


def _slugify(page: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", page.strip().lower()).strip("-")


def parse_pages(prompt: str) -> list[str]:
    """Collect the site's pages, home first.

    An explicit "pages: home, about, pricing" list wins (slugs,
    deduplicated, home forced first). Otherwise pages named in the
    brief ("a pricing page", "an about page") are collected in a
    fixed order after home. No named pages means just home.
    """
    match = _PAGES_LIST_RE.search(prompt)
    if match:
        pages: list[str] = []
        for raw in match.group(1).split(","):
            slug = _slugify(raw)
            if slug and slug not in pages:
                pages.append(slug)
        if "home" in pages:
            pages.remove("home")
        return ["home", *pages]
    text = prompt.lower()
    pages = ["home"]
    for page in _NAMED_PAGES:
        if f"{page} page" in text and page not in pages:
            pages.append(page)
    return pages


def parse_style(prompt: str) -> str | None:
    """Extract the requested style keyword, or None."""
    text = prompt.lower()
    for style in _STYLES:
        if style in text:
            return style
    return None


def build_brief(prompt: str) -> str:
    """Render the structured website brief for a request."""
    site_type = parse_site_type(prompt)
    style = parse_style(prompt)
    return (
        "\n\n# Website brief\n"
        f"Mode: {detect_mode(prompt)}\n"
        f"Site type: {site_type if site_type else 'model default'}\n"
        f"Pages: {', '.join(parse_pages(prompt))}\n"
        f"Style: {style if style else 'model default'}\n"
        f"Request: {prompt.strip()}"
    )


class WebsiteBuilderModule:
    def __init__(self, manifest: ModuleManifest):
        self.manifest = manifest

    def run(self, ctx: ModuleContext, prompt: str) -> str:
        guidance = MODE_GUIDANCE[detect_mode(prompt)]
        messages = [
            {
                "role": "system",
                "content": ctx.system_context + guidance + build_brief(prompt),
            },
            {"role": "user", "content": prompt},
        ]
        return ctx.provider.complete(messages, model=ctx.model)

    def build(self, prompt: str, out_dir, site_builder=None):
        """Build a real static website for the brief via the local
        builder.

        This is the development path (see builder.py): the output
        is a static site from the local builder, honestly labelled
        with its provider name — not output of the trained Gravion
        Builder model, which serves requests through `run`.
        """
        from .builder import LocalSiteBuilder, job_from_prompt

        job = job_from_prompt(prompt)
        builder = site_builder or LocalSiteBuilder()
        return builder.build(job, Path(out_dir))

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
# The menu/services/team/faq pages are named just as plainly in
# briefs ("a menu page and a services page" for a bakery): with
# only the original six listed, those requests built a home page
# alone and silently dropped the pages that were asked for. The
# booking, cart and checkout pages are named the same way ("menu
# and booking pages" for a restaurant, "cart and checkout" for a
# shop) and were dropped the same way.
_NAMED_PAGES = (
    "about",
    "pricing",
    "features",
    "blog",
    "gallery",
    "contact",
    "menu",
    "services",
    "team",
    "faq",
    "booking",
    "cart",
    "checkout",
)

# A negated page, style or site type is ruled out, not requested:
# "no pricing page", "not dark, use a light style" and "not a
# shop, a portfolio site" each recorded the negated item as the
# brief's choice, so the site was built with exactly what the
# request refused. A negator cancels a match only when it governs
# it: same clause (commas break clauses, so "not dark, use a
# light style" still takes the light style) and within the few
# words immediately before the match, mirroring the other
# modules' negation guards.
_NEGATOR_RE = re.compile(r"(?:\bno\b|\bnot\b|\bnever\b|\bwithout\b|n't\b)")
_CLAUSE_BREAKS = ".!?;,"
_NEGATION_WINDOW_WORDS = 3


def _is_negated(text: str, match_start: int) -> bool:
    clause_start = (
        max(text.rfind(break_char, 0, match_start) for break_char in _CLAUSE_BREAKS)
        + 1
    )
    words = text[clause_start:match_start].split()
    tail = " ".join(words[-_NEGATION_WINDOW_WORDS:])
    return bool(tail) and _NEGATOR_RE.search(tail) is not None

_PAGES_LIST_RE = re.compile(r"pages?\s*:\s*([a-z0-9 ,\-]+)", re.IGNORECASE)

_STYLES = ("dark", "light", "minimal", "bold", "modern")


def detect_mode(prompt: str) -> str:
    """Classify a website prompt as build / iterate."""
    text = prompt.strip().lower()
    if text.startswith(_ITERATE_PREFIXES):
        return MODE_ITERATE
    return MODE_BUILD


def _keyword_re(keyword: str) -> re.Pattern[str]:
    # Keywords match on word boundaries: "shop" inside "workshop"
    # is not a store, and "app" inside "happy" or "application"
    # does not make the site a product site. A trailing plural
    # "s" still names the keyword ("landing pages", "restaurants").
    return re.compile(rf"\b{re.escape(keyword)}s?\b")


_SITE_TYPE_RES = tuple(
    (tuple(_keyword_re(keyword) for keyword in keywords), site_type)
    for keywords, site_type in _SITE_TYPES
)

_STYLE_RES = tuple((style, _keyword_re(style)) for style in _STYLES)


def parse_site_type(prompt: str) -> str | None:
    """Extract the site type from explicit keywords, or None.

    A negated keyword states what the site is not ("not a shop,
    a portfolio site"), so only a non-negated match counts.
    """
    text = prompt.lower()
    for patterns, site_type in _SITE_TYPE_RES:
        if any(
            not _is_negated(text, match.start())
            for pattern in patterns
            for match in pattern.finditer(text)
        ):
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
        # A " with …" clause after the list states style or
        # content, not another page: "pages: home, about with a
        # dark style" names two pages, not a page called
        # "about-with-a-dark-style".
        listing = re.split(r"\s+with\s+", match.group(1), maxsplit=1)[0]
        pages: list[str] = []
        for raw in listing.split(","):
            # "pages: home, about and contact" joins the last
            # two pages with "and"; split the join only when
            # both sides name known pages, so a page called
            # "terms and conditions" stays one page.
            parts = re.split(r"\s+and\s+", raw)
            if len(parts) > 1 and all(
                _slugify(part) in ("home", *_NAMED_PAGES) for part in parts
            ):
                items = parts
            else:
                items = [raw]
            for item in items:
                slug = _slugify(item)
                if slug and slug not in pages:
                    pages.append(slug)
        if "home" in pages:
            pages.remove("home")
        return ["home", *pages]
    text = prompt.lower()
    pages = ["home"]
    # An enumeration before the word "pages" names its pages
    # just as surely as "about page" does: "home, about and
    # contact pages" names about and contact, in that order.
    # The list structure (a comma or an "and") is required, so
    # a stray page word in prose ("a site about cats") never
    # collects a page. A space-separated run of known page
    # names directly before "pages" ("home about contact
    # pages") is the same enumeration without the punctuation:
    # every word of the run names a page, so prose still
    # collects nothing, and a single trailing page word keeps
    # the named-page handling below.
    for pages_match in re.finditer(r"\bpages\b", text):
        clause_start = (
            max(text.rfind(break_char, 0, pages_match.start()) for break_char in ".;")
            + 1
        )
        segment = text[clause_start : pages_match.start()]
        if "," not in segment and " and " not in segment:
            run: list[tuple[int, str]] = []
            for word_match in reversed(list(re.finditer(r"\S+", segment))):
                word = word_match.group(0).strip(".,;:!?\"'()")
                if word not in ("home", *_NAMED_PAGES):
                    break
                run.append((word_match.start(), word))
            if len(run) < 2:
                continue
            for pos, page in reversed(run):
                if page != "home" and page not in pages and not _is_negated(
                    segment, pos
                ):
                    pages.append(page)
            continue
        found = [
            (word_match.start(), page)
            for page in _NAMED_PAGES
            for word_match in re.finditer(rf"\b{page}\b", segment)
            if not _is_negated(segment, word_match.start())
        ]
        for _pos, page in sorted(found):
            if page not in pages:
                pages.append(page)
    for page in _NAMED_PAGES:
        # "about us" and "contact us" name the about and contact
        # pages just as "about page" does. A negated naming ("no
        # pricing page", "without a contact page") rules the page
        # out instead of requesting it, so only a non-negated
        # occurrence collects the page.
        named = (f"{page} page", f"{page} us page", f"{page} us")
        if page in pages:
            continue
        if any(
            not _is_negated(text, match.start())
            for form in named
            for match in re.finditer(rf"\b{re.escape(form)}s?\b", text)
        ):
            pages.append(page)
    return pages


def parse_style(prompt: str) -> str | None:
    """Extract the requested style keyword, or None.

    A negated style ("not dark, use a light style") is the one
    style the request refuses, so only a non-negated match
    counts.
    """
    text = prompt.lower()
    for style, pattern in _STYLE_RES:
        # Word boundaries: "light" inside "slight" or
        # "lightweight" is not a light style request.
        if any(
            not _is_negated(text, match.start())
            for match in pattern.finditer(text)
        ):
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

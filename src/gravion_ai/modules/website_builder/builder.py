"""The local site builder for the website-builder module (Piece 5).

`WebsiteBuilderModule.run` briefs **Gravion Builder** — the
trained model — exactly like the other modules brief theirs. This
builder is the module's other path: turning a brief into an actual
static website on the machine the deployment runs on, with no
network and no trained weights. It exists so a deployment can
develop, demo and test its website workflow end-to-end before the
trained bundle is installed.

What the local builder is, honestly:

- It is a **development builder**, not the trained Gravion Builder
  model, and it does not pretend to be: the result and the site's
  `site.json` record the provider name (`gravion-local-builder`)
  so nothing downstream can mistake its output for the model's.
- It writes a small, real static site: one HTML page per page in
  the brief (home is `index.html`), a shared `style.css`, a
  `script.js`, and `site.json` describing the build. Content is
  template copy written for the brief's company and site type —
  deterministic (same brief, same site), never lorem ipsum, and
  never a claim of design quality beyond "a working starting
  point". Iterating on copy and design is the trained model's job,
  served through `run`.
- An empty brief is an error, not an empty site: the builder never
  reports success for a site it did not actually write.
"""

from __future__ import annotations

import hashlib
import html
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...errors import ProviderError
from .module import detect_mode, parse_pages, parse_site_type, parse_style

DEFAULT_SITE_TYPE = "general"
DEFAULT_STYLE = "modern"
DEFAULT_TITLE = "My Website"

# Accent colours the local builder picks from, by prompt hash.
_PALETTE = (
    "#1F3B73",
    "#7B2D8B",
    "#0E7C7B",
    "#C75146",
    "#2E5EAA",
    "#8C5E10",
    "#274156",
    "#5B2A86",
)

# One line of honest template copy per site type: what the page
# says until the trained model (or the company) writes real copy.
_TYPE_COPY = {
    "general": "Welcome to {title}. This is our home on the web — who we are, what we do, and how to reach us.",
    "landing page": "{title} — the short version: what it is, why it matters, and how to get started today.",
    "portfolio": "Selected work by {title}: projects, outcomes, and the story behind each one.",
    "blog": "Writing from {title}: notes, announcements and things we are learning in public.",
    "online store": "Shop {title}: browse the range, check prices, and order in a couple of clicks.",
    "documentation": "{title} documentation: start with the quickstart, then dive into the guides and reference.",
    "restaurant": "{title}: seasonal menus, honest cooking, and a table waiting for you. Book ahead on weekends.",
    "agency": "{title} is an agency for brands that want the work to speak first: strategy, design, delivery.",
    "product": "{title}: the product tour — what it does, who it is for, and why teams switch to it.",
}

_QUOTED_RE = re.compile(r"[\"']([^\"']{2,60})[\"']")
_FOR_RE = re.compile(
    r"\bfor\s+([A-Za-z][A-Za-z0-9&' .-]{1,48}?)(?:\s+(?:with|in|using)\b|[.,]|$)"
)


def parse_title(prompt: str) -> str:
    """Derive the site's title (the company / owner name).

    A quoted name wins; otherwise the phrase after "for" ("a
    website for Acme Robotics"); otherwise the honest default.
    """
    quoted = _QUOTED_RE.search(prompt)
    if quoted:
        return quoted.group(1).strip()
    match = _FOR_RE.search(prompt)
    if match:
        title = match.group(1).strip()
        # "for our bakery" / "for my band" name no company — the
        # brief's owner is the deployment itself, so the honest
        # default applies instead of inventing a name from "our".
        if title and not title.lower().startswith(("our ", "my ")):
            return title if title[0].isupper() else title.title()
    return DEFAULT_TITLE


@dataclass(frozen=True)
class SiteJob:
    """A fully-specified local build: everything the builder
    needs, derived from the brief by `job_from_prompt`."""

    prompt: str
    mode: str
    title: str
    site_type: str
    pages: tuple[str, ...]
    style: str


@dataclass(frozen=True)
class SiteResult:
    """What a build produced: the directory, who built it, and the
    spec it was built to."""

    path: Path
    provider: str
    title: str
    pages: tuple[str, ...]
    files: tuple[str, ...]


def job_from_prompt(prompt: str) -> SiteJob:
    """Derive a build job from a plain-language brief, using the
    module's own parsers. Unstated type/style fall back to the
    honest defaults (general, modern) rather than being invented
    per-request."""
    return SiteJob(
        prompt=prompt.strip(),
        mode=detect_mode(prompt),
        title=parse_title(prompt),
        site_type=parse_site_type(prompt) or DEFAULT_SITE_TYPE,
        pages=tuple(parse_pages(prompt)),
        style=parse_style(prompt) or DEFAULT_STYLE,
    )


@runtime_checkable
class SiteBuilder(Protocol):
    """A site builder: takes a job, writes a static site, reports
    what it produced. The trained Gravion Builder bundle and the
    local development builder both sit behind this interface."""

    name: str

    def build(self, job: SiteJob, out_dir: Path) -> SiteResult:
        ...


def _page_filename(page: str) -> str:
    return "index.html" if page == "home" else f"{page}.html"


def _css(job: SiteJob, accent: str) -> str:
    dark = job.style == "dark"
    bg = "#101418" if dark else "#ffffff"
    fg = "#e8ecef" if dark else "#1a1d21"
    muted = "#9aa4ad" if dark else "#5b6470"
    return (
        ":root {\n"
        f"  --accent: {accent};\n"
        f"  --bg: {bg};\n"
        f"  --fg: {fg};\n"
        f"  --muted: {muted};\n"
        "}\n"
        "* { box-sizing: border-box; }\n"
        "body { margin: 0; font-family: system-ui, sans-serif; "
        "background: var(--bg); color: var(--fg); }\n"
        "nav { display: flex; gap: 1rem; padding: 1rem 2rem; "
        "border-bottom: 2px solid var(--accent); }\n"
        "nav a { color: var(--fg); text-decoration: none; }\n"
        "nav a[aria-current=\"page\"] { color: var(--accent); "
        "font-weight: 700; }\n"
        "main { max-width: 60rem; margin: 0 auto; padding: 2rem; }\n"
        ".hero h1 { font-size: 2.5rem; margin-bottom: 0.5rem; }\n"
        ".hero p, section p { color: var(--muted); line-height: 1.6; }\n"
        "footer { padding: 1rem 2rem; color: var(--muted); "
        "border-top: 1px solid var(--muted); }\n"
    )


def _nav_link(other: str, page: str) -> str:
    aria = ' aria-current="page"' if other == page else ""
    return (
        f'<a href="{_page_filename(other)}"{aria}>'
        f"{html.escape(other.replace('-', ' ').title())}</a>"
    )


def _page_html(job: SiteJob, page: str) -> str:
    title = html.escape(job.title)
    copy = html.escape(_TYPE_COPY[job.site_type].format(title=job.title))
    nav = "\n      ".join(
        _nav_link(other, page)
        for other in job.pages
    )
    if page == "home":
        body = (
            f'    <section class="hero">\n'
            f"      <h1>{title}</h1>\n"
            f"      <p>{copy}</p>\n"
            f"    </section>"
        )
    else:
        heading = html.escape(page.replace("-", " ").title())
        body = (
            f"    <section>\n"
            f"      <h1>{heading}</h1>\n"
            f"      <p>{copy}</p>\n"
            f"    </section>"
        )
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        '  <meta name="viewport" content="width=device-width, '
        'initial-scale=1">\n'
        f"  <title>{title} — "
        f"{html.escape(page.replace('-', ' ').title())}</title>\n"
        '  <link rel="stylesheet" href="style.css">\n'
        "</head>\n"
        "<body>\n"
        "  <nav>\n"
        f"      {nav}\n"
        "  </nav>\n"
        "  <main>\n"
        f"{body}\n"
        "  </main>\n"
        f"  <footer>Built for {title}.</footer>\n"
        '  <script src="script.js"></script>\n'
        "</body>\n"
        "</html>\n"
    )


_SCRIPT_JS = (
    "// Local builder scaffold: marks the current year in the footer.\n"
    "document.addEventListener('DOMContentLoaded', function () {\n"
    "  var footer = document.querySelector('footer');\n"
    "  if (footer) { footer.dataset.year = new Date().getFullYear(); }\n"
    "});\n"
)


class LocalSiteBuilder:
    """The offline development builder.

    See the module docstring for what it is and is not: a small,
    deterministic, honestly-labelled static site — never a
    stand-in claiming to be the trained model.
    """

    name = "gravion-local-builder"

    def build(self, job: SiteJob, out_dir: Path) -> SiteResult:
        if not job.prompt:
            raise ProviderError(
                "the local site builder needs a non-empty brief; "
                "describe the site (company, type, pages) or use "
                "the trained Gravion Builder bundle"
            )
        out_dir = Path(out_dir)
        if out_dir.exists() and not out_dir.is_dir():
            raise ProviderError(
                f"cannot build a site at {out_dir}: path exists and "
                f"is not a directory"
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256(job.prompt.encode("utf-8")).digest()
        accent = _PALETTE[digest[0] % len(_PALETTE)]

        files: list[str] = []
        for page in job.pages:
            filename = _page_filename(page)
            (out_dir / filename).write_text(
                _page_html(job, page), encoding="utf-8"
            )
            files.append(filename)
        (out_dir / "style.css").write_text(_css(job, accent), encoding="utf-8")
        files.append("style.css")
        (out_dir / "script.js").write_text(_SCRIPT_JS, encoding="utf-8")
        files.append("script.js")
        site_doc = {
            "title": job.title,
            "site_type": job.site_type,
            "style": job.style,
            "mode": job.mode,
            "pages": list(job.pages),
            "provider": self.name,
            "files": files,
        }
        (out_dir / "site.json").write_text(
            json.dumps(site_doc, indent=2) + "\n", encoding="utf-8"
        )
        files.append("site.json")
        return SiteResult(
            path=out_dir,
            provider=self.name,
            title=job.title,
            pages=job.pages,
            files=tuple(files),
        )


def build_site(
    prompt: str,
    out_dir: Path,
    builder: SiteBuilder | None = None,
) -> SiteResult:
    """Build a website for a brief: derive the job, then build it
    with the given builder (the local development builder by
    default)."""
    job = job_from_prompt(prompt)
    return (builder or LocalSiteBuilder()).build(job, Path(out_dir))

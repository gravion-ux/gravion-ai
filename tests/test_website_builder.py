import json
from pathlib import Path

import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError, ProviderError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.website_builder.builder import (
    LocalSiteBuilder,
    build_site,
    job_from_prompt,
    parse_title,
)
from gravion_ai.modules.website_builder.module import (
    MODE_BUILD,
    MODE_ITERATE,
    build_brief,
    detect_mode,
    parse_pages,
    parse_site_type,
    parse_style,
)
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry
from gravion_ai.runtime import GravionRuntime


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None):
        self.calls.append((list(messages), model))
        return "recorded reply"


def _ctx(provider, model="gravion-builder"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def test_website_builder_instantiates_and_answers():
    module = default_registry().instantiate("website-builder")
    out = module.run(_ctx(GravionCoreProvider()), "a website for our bakery")
    assert "a website for our bakery" in out


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("update: change the hero headline", MODE_ITERATE),
        ("Update the pricing page to add a new plan", MODE_ITERATE),
        ("please update this site with our new logo", MODE_ITERATE),
        ("revise: fix the contact form", MODE_ITERATE),
        ("change the footer colour", MODE_ITERATE),
        ("edit this site to add a blog", MODE_ITERATE),
        ("a website for our bakery", MODE_BUILD),
        ("build a landing page for our app", MODE_BUILD),
    ],
)
def test_detect_mode(prompt, expected):
    assert detect_mode(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("an online store for our merch", "online store"),
        ("a landing page for the launch", "landing page"),
        ("a portfolio for our designer", "portfolio"),
        ("a documentation site for the API", "documentation"),
        ("a website for our restaurant", "restaurant"),
        ("a blog for our travels", "blog"),
        ("an agency site for our studio", "agency"),
        ("a saas marketing site", "product"),
        ("a website for our bakery", None),
        # Regression: keywords used to match as substrings, so a
        # "workshop" became an online store and "happy" /
        # "application" became a product site via "app".
        ("build a site for my woodworking workshop", None),
        ("build a happy birthday site", None),
        ("build a job application tracker site", None),
        ("a shop for our merch", "online store"),
        ("a website for our app", "product"),
    ],
)
def test_parse_site_type(prompt, expected):
    assert parse_site_type(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a website with pages: home, about, pricing", ["home", "about", "pricing"]),
        ("pages: pricing, about", ["home", "pricing", "about"]),
        ("a site with a pricing page and a contact page",
         ["home", "pricing", "contact"]),
        ("a site with an about page", ["home", "about"]),
        ("a website for our bakery", ["home"]),
        # Regression: a trailing "with …" style clause became a
        # page named "about-with-a-dark-style".
        ("build a site, pages: home, about with a dark style",
         ["home", "about"]),
        # "about us" / "contact us" name the about and contact
        # pages.
        ("build a site with an about us page and contact us page",
         ["home", "about", "contact"]),
        # Regression: an enumeration before "pages" dropped
        # every page but the last ("about and contact pages"
        # named only contact), and an explicit list joined by
        # "and" became one slug ("about-and-contact").
        ("build a portfolio site with home, about and contact pages",
         ["home", "about", "contact"]),
        ("a site with pricing, blog and gallery pages",
         ["home", "pricing", "blog", "gallery"]),
        ("pages: home, about and contact", ["home", "about", "contact"]),
        # A page whose own name contains "and" stays one page,
        # and a stray page word in prose collects nothing.
        ("pages: home, terms and conditions",
         ["home", "terms-and-conditions"]),
        ("a site about cats with photos", ["home"]),
        # Regression: the booking, cart and checkout pages
        # were dropped the same way the menu page once was,
        # so a restaurant brief lost its booking page and a
        # shop brief lost its cart and checkout pages.
        ("a restaurant site with menu and booking pages",
         ["home", "menu", "booking"]),
        ("a shop site with a cart page and a checkout page",
         ["home", "cart", "checkout"]),
        ("a site with no booking page, home and menu pages",
         ["home", "menu"]),
        # Regression: a space-separated enumeration before
        # "pages" ("home about contact pages") dropped every
        # page but the last, because only a comma or an "and"
        # counted as list structure.
        ("a website with home about contact pages",
         ["home", "about", "contact"]),
        ("a site with menu booking pages", ["home", "menu", "booking"]),
        ("a site about cats and dogs", ["home"]),
    ],
)
def test_parse_pages(prompt, expected):
    assert parse_pages(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a dark themed portfolio", "dark"),
        ("a light, airy blog", "light"),
        ("a minimal landing page", "minimal"),
        ("a bold agency site", "bold"),
        ("a modern website for our app", "modern"),
        ("a website for our bakery", None),
        # Regression: "light" inside "slight" read as a light
        # style request.
        ("a slight variation of the design", None),
    ],
)
def test_parse_style(prompt, expected):
    assert parse_style(prompt) == expected


def test_brief_records_mode_type_pages_style_and_request():
    brief = build_brief("a dark landing page for Acme with a pricing page")
    assert "Mode: build" in brief
    assert "Site type: landing page" in brief
    assert "Pages: home, pricing" in brief
    assert "Style: dark" in brief
    assert "Request: a dark landing page for Acme with a pricing page" in brief
    brief = build_brief("a website for our bakery")
    assert "Site type: model default" in brief
    assert "Style: model default" in brief
    assert "Pages: home" in brief


@pytest.mark.parametrize(
    "prompt,marker",
    [
        ("a website for our bakery", "Website mode: build"),
        ("update: change the hero headline", "Website mode: iterate"),
    ],
)
def test_website_guidance_and_brief_reach_the_provider(prompt, marker):
    provider = RecordingProvider()
    module = default_registry().instantiate("website-builder")
    out = module.run(_ctx(provider), prompt)
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-builder"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert marker in messages[0]["content"]
    assert "# Website brief" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": prompt}


def test_website_runtime_dev_end_to_end_with_gravion_builder(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-builder"))
    assert runtime.resolve_model("website-builder") == "gravion-builder"
    out = runtime.website_builder("a website for our bakery")
    assert "a website for our bakery" in out


def test_website_runtime_rejects_model_without_builder_capability(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.website_builder("a website for our bakery")


def test_website_manifest_signs_and_verifies(issuer, website_builder_module_copy):
    payload = build_manifest(
        website_builder_module_copy,
        name="website-builder",
        module_type="website-builder",
        version="0.1.0",
        entrypoint="gravion_ai.modules.website_builder.module:WebsiteBuilderModule",
    )
    write_signed_manifest(website_builder_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(website_builder_module_copy, issuer)
    assert verified["name"] == "website-builder"
    assert verified["type"] == "website-builder"
    assert "module.py" in verified["files"]
    assert "builder.py" in verified["files"]


def test_bundled_website_manifest_is_unsigned(issuer):
    import gravion_ai.modules.website_builder as website_builder_pkg

    with pytest.raises(ModuleTamperError):
        verify_module(Path(website_builder_pkg.__file__).parent, issuer)


# -- the local builder ------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a landing page for Acme Robotics", "Acme Robotics"),
        ("a portfolio for 'Jane Doe'", "Jane Doe"),
        ("a website for our bakery", "My Website"),
    ],
)
def test_parse_title(prompt, expected):
    assert parse_title(prompt) == expected


def test_job_from_prompt_defaults():
    job = job_from_prompt("a website for our bakery")
    assert job.title == "My Website"
    assert job.site_type == "general"
    assert job.style == "modern"
    assert job.pages == ("home",)


def test_local_builder_writes_a_real_site(tmp_path):
    result = build_site(
        "a dark landing page for Acme Robotics with a pricing page",
        tmp_path / "site",
    )
    assert result.provider == "gravion-local-builder"
    assert result.title == "Acme Robotics"
    assert result.pages == ("home", "pricing")
    out = tmp_path / "site"
    for name in ("index.html", "pricing.html", "style.css", "script.js", "site.json"):
        assert (out / name).exists(), name
    index = (out / "index.html").read_text(encoding="utf-8")
    assert "Acme Robotics" in index
    assert 'href="pricing.html"' in index
    site_doc = json.loads((out / "site.json").read_text(encoding="utf-8"))
    assert site_doc["provider"] == "gravion-local-builder"
    assert site_doc["pages"] == ["home", "pricing"]
    assert site_doc["style"] == "dark"


def test_local_builder_is_deterministic(tmp_path):
    prompt = "a portfolio for 'Jane Doe' with a gallery page"
    first = build_site(prompt, tmp_path / "a")
    second = build_site(prompt, tmp_path / "b")
    for name in first.files:
        assert (tmp_path / "a" / name).read_bytes() == (tmp_path / "b" / name).read_bytes()


def test_local_builder_refuses_an_empty_brief(tmp_path):
    with pytest.raises(ProviderError, match="non-empty brief"):
        LocalSiteBuilder().build(job_from_prompt("   "), tmp_path / "site")


def test_local_builder_refuses_a_file_as_out_dir(tmp_path):
    blocker = tmp_path / "blocker"
    blocker.write_text("not a directory", encoding="utf-8")
    with pytest.raises(ProviderError, match="not a directory"):
        build_site("a website for our bakery", blocker)


def test_runtime_build_website_end_to_end(make_config, tmp_path):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-builder"))
    result = runtime.build_website(
        "a website for Acme Robotics", tmp_path / "site"
    )
    assert result.provider == "gravion-local-builder"
    assert (tmp_path / "site" / "index.html").exists()


def test_negated_pages_styles_and_site_types_are_refused_not_requested():
    # Regression: "no pricing page" collected the pricing page,
    # "not dark, use a light style" recorded the dark style and
    # "not a shop, a portfolio site" recorded an online store —
    # each brief carried exactly what the request refused.
    assert parse_pages("landing page for a gym, no pricing page") == ["home"]
    assert parse_pages("website without a contact page") == ["home"]
    assert parse_pages("build a site, not a blog page, just home") == ["home"]
    assert parse_style("not dark, use a light style") == "light"
    assert parse_style("no dark style, keep it light") == "light"
    assert parse_style("dark style, not minimal") == "dark"
    assert parse_site_type("not a shop, a portfolio site") == "portfolio"
    # Non-negated namings still collect, including the menu,
    # services, team and faq pages briefs name just as plainly.
    assert parse_pages("home, about and menu pages for a bakery") == [
        "home",
        "about",
        "menu",
    ]
    assert parse_pages("a menu page and a services page") == [
        "home",
        "menu",
        "services",
    ]
    assert parse_pages("team page and faq page please") == ["home", "team", "faq"]
    assert parse_site_type("landing pages for gyms") == "landing page"

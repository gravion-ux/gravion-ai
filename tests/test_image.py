import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.image.module import (
    MODE_EDIT,
    MODE_GENERATE,
    build_brief,
    detect_mode,
    parse_aspect,
    resolve_size,
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


def _ctx(provider, model="gravion-vision"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def test_image_instantiates_and_answers():
    module = default_registry().instantiate("image")
    out = module.run(_ctx(GravionCoreProvider()), "a logo for our app")
    assert "a logo for our app" in out


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("edit: make the sky darker", MODE_EDIT),
        ("Edit this image to remove the watermark", MODE_EDIT),
        ("please edit the product photo", MODE_EDIT),
        ("modify: swap the background for a beach", MODE_EDIT),
        ("Modify the logo to be bolder", MODE_EDIT),
        ("a portrait of our founder", MODE_GENERATE),
        ("generate a banner for the launch", MODE_GENERATE),
        ("draw a square icon for settings", MODE_GENERATE),
    ],
)
def test_detect_mode(prompt, expected):
    assert detect_mode(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a banner at 1920x1080 for the site", "1920x1080"),
        ("an icon, 512 x 512", "512x512"),
        # "800 by 600" states a pixel size just as "800x600" does.
        ("a poster, 800 by 600", "800x600"),
        ("a square logo", "1:1"),
        ("a portrait of the founder", "3:4"),
        ("a landscape photo of the office", "16:9"),
        ("a wide banner for the homepage", "21:9"),
        # An explicit size beats a named shape.
        ("a square image at 1200x628", "1200x628"),
        # Explicit ratios beat named shapes, not pixel sizes.
        ("a 4:5 poster for the launch", "4:5"),
        ("a 3:2 photo of the office", "3:2"),
        ("a square image at 16:9", "16:9"),
        ("a 4:5 image at 1200x628", "1200x628"),
        # Clock readings and durations are not aspects.
        ("a photo at 10:30", None),
        ("a 1:30 exposure", None),
        ("a logo for our app", None),
    ],
)
def test_parse_aspect(prompt, expected):
    assert parse_aspect(prompt) == expected


def test_resolve_size_uses_explicit_ratio():
    assert resolve_size("a 4:5 poster") == (819, 1024)
    assert resolve_size("a 3:2 photo") == (1024, 682)
    assert resolve_size("a poster, 800 by 600") == (800, 600)


def test_brief_records_mode_aspect_and_request():
    brief = build_brief("edit: darken the sky in this landscape shot")
    assert "Mode: edit" in brief
    assert "Aspect: 16:9" in brief
    assert "Request: edit: darken the sky in this landscape shot" in brief
    brief = build_brief("a logo for our app")
    assert "Mode: generate" in brief
    assert "Aspect: model default" in brief


@pytest.mark.parametrize(
    "prompt,marker",
    [
        ("a logo for our app", "Image mode: generate"),
        ("edit: remove the background", "Image mode: edit"),
    ],
)
def test_image_guidance_and_brief_reach_the_provider(prompt, marker):
    provider = RecordingProvider()
    module = default_registry().instantiate("image")
    out = module.run(_ctx(provider), prompt)
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-vision"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert marker in messages[0]["content"]
    assert "# Image brief" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": prompt}


def test_image_runtime_dev_end_to_end_with_gravion_vision(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-vision"))
    assert runtime.resolve_model("image") == "gravion-vision"
    out = runtime.image("a square logo for our app")
    assert "a square logo for our app" in out


def test_image_runtime_rejects_model_without_image_capability(make_config):
    # Gravion Core serves chat + support, not image: the registry
    # check must refuse rather than answer with the wrong model.
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.image("a square logo for our app")


def test_image_manifest_signs_and_verifies(issuer, image_module_copy):
    payload = build_manifest(
        image_module_copy,
        name="image",
        module_type="image",
        version="0.1.0",
        entrypoint="gravion_ai.modules.image.module:ImageModule",
    )
    write_signed_manifest(image_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(image_module_copy, issuer)
    assert verified["name"] == "image"
    assert verified["type"] == "image"
    assert "module.py" in verified["files"]


def test_bundled_image_manifest_is_unsigned(issuer):
    import gravion_ai.modules.image as image_pkg
    from pathlib import Path

    with pytest.raises(ModuleTamperError):
        verify_module(Path(image_pkg.__file__).parent, issuer)

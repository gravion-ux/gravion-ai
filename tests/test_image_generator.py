"""Tests for the local HQ image generator."""

import hashlib
import io

import pytest
from PIL import Image

from gravion_ai.modules.image.generator import (
    QUALITY_LEVELS,
    detect_palette,
    extract_text,
    generate_image,
    generate_to_file,
)
from gravion_ai.modules.image.module import resolve_size
from gravion_ai.registry import default_registry


def _png_bytes(result) -> bytes:
    buf = io.BytesIO()
    result.image.save(buf, format="PNG")
    return buf.getvalue()


def test_generates_requested_dimensions():
    result = generate_image("a sunset over mountains", width=640, height=360)
    assert result.image.size == (640, 360)
    assert result.width == 640 and result.height == 360
    assert result.palette == "sunset"


@pytest.mark.parametrize("quality", QUALITY_LEVELS)
def test_every_quality_level_produces_exact_size(quality):
    result = generate_image("ocean waves at dawn", width=320, height=320,
                            quality=quality)
    assert result.image.size == (320, 320)
    assert result.quality == quality


def test_generation_is_deterministic_for_prompt_and_seed():
    a = generate_image("a starry night over pine forest", width=256, height=256, seed=7)
    b = generate_image("a starry night over pine forest", width=256, height=256, seed=7)
    assert hashlib.sha256(_png_bytes(a)).hexdigest() == hashlib.sha256(
        _png_bytes(b)).hexdigest()
    c = generate_image("a starry night over pine forest", width=256, height=256, seed=8)
    assert _png_bytes(a) != _png_bytes(c)


def test_seed_defaults_from_prompt():
    a = generate_image("desert dunes", width=128, height=128)
    b = generate_image("desert dunes", width=128, height=128)
    assert a.seed == b.seed


def test_image_is_detailed_not_blank():
    result = generate_image("sunset mountains with clouds", width=512, height=512)
    colours = result.image.getcolors(maxcolors=1_000_000)
    assert colours is not None and len(colours) > 200


def test_quoted_text_is_rendered():
    result = generate_image('a neon poster that says "GRAVION"', width=640, height=640)
    assert result.text_rendered == "GRAVION"
    assert extract_text('poster "Hello World" please') == "Hello World"
    # White text pixels exist in the centre band of the image.
    centre = result.image.crop((0, 200, 640, 440))
    raw = centre.tobytes()
    whites = sum(1 for i in range(0, len(raw), 3)
                 if raw[i] > 245 and raw[i + 1] > 245 and raw[i + 2] > 245)
    assert whites > 50


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("sunset over the valley", "sunset"),
        ("ocean waves", "ocean"),
        ("deep forest pines", "forest"),
        ("starry night sky", "night"),
        ("desert dunes", "desert"),
        ("neon cyber city", "neon"),
        ("snow winter peaks", "snow"),
        ("spring meadow", "meadow"),
        ("a logo", "sunset"),
    ],
)
def test_detect_palette(prompt, expected):
    assert detect_palette(prompt) == expected


def test_generate_to_file_writes_png(tmp_path):
    out = tmp_path / "scene.png"
    result = generate_to_file("snow mountains", out, width=480, height=270)
    assert out.exists()
    with Image.open(out) as img:
        assert img.format == "PNG"
        assert img.size == (480, 270)
    assert result.palette == "snow"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"width": 10},
        {"width": 5000},
        {"height": 0},
        {"quality": "maximum"},
    ],
)
def test_invalid_options_raise(kwargs):
    with pytest.raises(ValueError):
        generate_image("a sunset", **kwargs)


def test_empty_prompt_raises():
    with pytest.raises(ValueError):
        generate_image("   ")


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a banner at 1920x1080", (1920, 1080)),
        ("a square logo", (1024, 1024)),
        ("a landscape photo", (1024, 576)),
        ("a portrait shot", (768, 1024)),
        ("a logo", (1024, 1024)),
    ],
)
def test_resolve_size(prompt, expected):
    assert resolve_size(prompt) == expected


def test_module_generate_local_wiring(tmp_path):
    module = default_registry().instantiate("image")
    out = tmp_path / "poster.png"
    result = module.generate_local('a landscape sunset poster "Gravion AI"',
                                   quality="standard", output_path=out)
    assert result.image.size == (1024, 576)
    assert result.text_rendered == "Gravion AI"
    assert out.exists()


def test_human_prompt_draws_figure():
    result = generate_image("human portrait at sunset", width=512, height=512,
                            quality="ultra", seed=7)
    assert result.human_figure is True
    # The figure occupies the centre foreground: dark silhouette pixels
    # must be present in the middle band above the ground line.
    band = result.image.crop((200, 150, 312, 430))
    raw = band.tobytes()
    dark = sum(1 for i in range(0, len(raw), 3)
               if raw[i] < 45 and raw[i + 1] < 35 and raw[i + 2] < 60)
    assert dark > 300


def test_landscape_prompt_has_no_figure():
    result = generate_image("sunset over mountains", width=256, height=256)
    assert result.human_figure is False


def test_human_generation_is_deterministic():
    a = generate_image("a person at dusk", width=256, height=256, seed=3)
    b = generate_image("a person at dusk", width=256, height=256, seed=3)
    assert a.human_figure and b.human_figure
    assert hashlib.sha256(_png_bytes(a)).hexdigest() == hashlib.sha256(
        _png_bytes(b)).hexdigest()

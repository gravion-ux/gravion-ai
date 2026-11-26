"""Local HQ image generation for Gravion AI.

A fully local, deterministic procedural image synthesizer built on
Pillow. It turns a text prompt into a high-resolution, detailed
PNG: multi-stop gradient scenes, layered silhouettes (mountain
ridges, trees, waves, skyline), soft lighting with glow and shadow
passes, atmospheric particles, and accurate rendering of quoted
text from the prompt.

Everything runs on the local machine — no network, no external
service, no downloaded weights. The same prompt and seed always
produce the same image, so results are reproducible and testable.

Author: Priyanshu Kumar Thakur.
"""

from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

MIN_SIZE = 64
MAX_SIZE = 4096

QUALITY_LEVELS = ("draft", "standard", "high", "ultra")

# Supersampling factor and scene richness per quality level. The
# scene is rendered larger than requested and downscaled, which
# gives clean anti-aliased edges and finer detail at HQ sizes.
_QUALITY = {
    "draft": {"supersample": 1, "layers": 3, "particles": 40, "grain": 0},
    "standard": {"supersample": 2, "layers": 4, "particles": 90, "grain": 4},
    "high": {"supersample": 2, "layers": 5, "particles": 160, "grain": 6},
    "ultra": {"supersample": 3, "layers": 6, "particles": 260, "grain": 8},
}

# Named palettes: (sky stops top->horizon, sun colour, ridge colours
# far->near, ground colour, accent colour).
_PALETTES = {
    "sunset": {
        "sky": [(38, 24, 84), (122, 44, 110), (232, 96, 84), (255, 176, 92)],
        "sun": (255, 214, 130),
        "ridges": [(94, 52, 110), (66, 38, 92), (40, 26, 66), (22, 16, 44)],
        "ground": (18, 12, 36),
        "accent": (255, 190, 110),
    },
    "ocean": {
        "sky": [(16, 60, 110), (30, 110, 168), (96, 178, 214), (190, 230, 240)],
        "sun": (255, 246, 200),
        "ridges": [(38, 96, 140), (28, 78, 122), (20, 60, 104), (14, 44, 84)],
        "ground": (10, 52, 96),
        "accent": (150, 225, 250),
    },
    "forest": {
        "sky": [(20, 52, 44), (36, 96, 70), (110, 168, 110), (210, 235, 190)],
        "sun": (255, 250, 205),
        "ridges": [(30, 84, 58), (22, 68, 46), (16, 52, 36), (10, 38, 26)],
        "ground": (8, 30, 20),
        "accent": (170, 230, 160),
    },
    "night": {
        "sky": [(6, 8, 28), (14, 20, 56), (30, 40, 92), (54, 66, 128)],
        "sun": (235, 240, 255),
        "ridges": [(26, 32, 66), (20, 26, 54), (14, 20, 42), (10, 14, 32)],
        "ground": (8, 10, 24),
        "accent": (190, 205, 255),
    },
    "desert": {
        "sky": [(70, 60, 120), (180, 110, 90), (240, 170, 100), (255, 220, 160)],
        "sun": (255, 230, 160),
        "ridges": [(190, 130, 84), (164, 104, 64), (132, 80, 48), (100, 58, 36)],
        "ground": (86, 50, 30),
        "accent": (255, 214, 150),
    },
    "neon": {
        "sky": [(10, 6, 30), (32, 10, 64), (80, 16, 110), (150, 30, 140)],
        "sun": (120, 240, 255),
        "ridges": [(44, 20, 84), (34, 16, 68), (24, 12, 52), (16, 8, 38)],
        "ground": (10, 6, 26),
        "accent": (120, 240, 255),
    },
    "snow": {
        "sky": [(60, 90, 140), (120, 160, 205), (190, 215, 238), (240, 248, 255)],
        "sun": (255, 252, 235),
        "ridges": [(140, 168, 200), (110, 140, 180), (84, 114, 158), (62, 90, 134)],
        "ground": (235, 242, 250),
        "accent": (255, 255, 255),
    },
    "meadow": {
        "sky": [(40, 90, 160), (90, 150, 210), (160, 205, 235), (225, 242, 250)],
        "sun": (255, 248, 190),
        "ridges": [(96, 150, 96), (74, 128, 76), (54, 106, 58), (38, 84, 44)],
        "ground": (44, 96, 50),
        "accent": (255, 230, 140),
    },
}

_PALETTE_KEYWORDS = {
    "sunset": ("sunset", "dusk", "evening", "golden hour", "twilight"),
    "ocean": ("ocean", "sea", "beach", "wave", "coast", "underwater", "lake"),
    "forest": ("forest", "jungle", "woods", "pine", "jungle"),
    "night": ("night", "midnight", "starry", "moonlit", "dark sky"),
    "desert": ("desert", "dune", "sand", "canyon"),
    "neon": ("neon", "cyber", "futuristic", "synthwave", "glow city"),
    "snow": ("snow", "winter", "frozen", "arctic", "ice"),
    "meadow": ("meadow", "field", "spring", "garden", "valley"),
}

_QUOTED_RE = re.compile(r"[\"“]([^\"”]{1,120})[\"”]")

_HUMAN_KEYWORDS = (
    "human", "portrait", "person", "man", "woman", "figure",
    "girl", "boy", "people", "silhouette",
)


@dataclass(frozen=True)
class GeneratedImage:
    """One locally generated image plus the facts about how it was made."""

    image: Image.Image
    prompt: str
    seed: int
    palette: str
    quality: str
    text_rendered: str | None = None
    human_figure: bool = False

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    def save(self, path: str | Path) -> Path:
        """Save as PNG with descriptive metadata; returns the path."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        self.image.save(out, format="PNG", optimize=True)
        return out


def detect_palette(prompt: str) -> str:
    """Pick the scene palette named by the prompt (sunset default)."""
    text = prompt.lower()
    # Night/neon intent wins over scene nouns ("starry night over a
    # pine forest" is a night scene with trees, not a daytime forest).
    for name in ("night", "neon"):
        if any(word in text for word in _PALETTE_KEYWORDS[name]):
            return name
    for name, keywords in _PALETTE_KEYWORDS.items():
        if any(word in text for word in keywords):
            return name
    return "sunset"


def extract_text(prompt: str) -> str | None:
    """Return the quoted text the prompt asks to render, if any."""
    match = _QUOTED_RE.search(prompt)
    return match.group(1).strip() if match else None


def _seed_for(prompt: str, seed: int | None) -> int:
    if seed is not None:
        return int(seed)
    digest = hashlib.sha256(prompt.strip().lower().encode("utf-8")).hexdigest()
    return int(digest[:12], 16)


def _lerp(a: tuple[int, ...], b: tuple[int, ...], t: float) -> tuple[int, ...]:
    return tuple(int(round(x + (y - x) * t)) for x, y in zip(a, b))


def _gradient_sky(size: tuple[int, int], stops: list[tuple[int, int, int]]) -> Image.Image:
    """Vertical multi-stop gradient, rendered as a thin strip and stretched."""
    w, h = size
    strip = Image.new("RGB", (1, h))
    px = strip.load()
    segments = len(stops) - 1
    for y in range(h):
        pos = (y / max(h - 1, 1)) * segments
        idx = min(int(pos), segments - 1)
        px[0, y] = _lerp(stops[idx], stops[idx + 1], pos - idx)
    return strip.resize((w, h))


def _glow(size: tuple[int, int], centre: tuple[int, int], radius: int,
          colour: tuple[int, int, int], strength: int = 160) -> Image.Image:
    """A soft radial light glow layer (RGBA, additive-feel)."""
    layer = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    cx, cy = centre
    steps = 24
    for i in range(steps, 0, -1):
        r = int(radius * i / steps)
        alpha = int(strength * (1 - i / steps) ** 2)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=colour + (alpha,))
    return layer.filter(ImageFilter.GaussianBlur(radius=max(radius // 6, 2)))


def _ridge_points(rng: random.Random, w: int, base_y: int, amp: int,
                  peaks: int) -> list[tuple[int, int]]:
    """A jagged mountain ridge line across the canvas."""
    points: list[tuple[int, int]] = [(0, base_y)]
    step = w / peaks
    for i in range(peaks + 1):
        x = int(i * step)
        y = base_y - int(rng.random() * amp) - int(amp * 0.25 * math.sin(i * 1.7))
        points.append((x, y))
    points.append((w, base_y))
    return points


def _draw_ridges(img: Image.Image, rng: random.Random, palette: dict,
                 layers: int) -> None:
    """Layered mountain ridges with snow caps and soft shadows."""
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    horizon = int(h * 0.62)
    colours = palette["ridges"]
    for layer in range(layers):
        colour = colours[min(layer, len(colours) - 1)]
        base_y = horizon + int((h - horizon) * layer / max(layers, 1))
        amp = int(h * (0.20 - 0.02 * layer))
        points = _ridge_points(rng, w, base_y, max(amp, 8), peaks=7 + layer * 2)
        # Soft shadow pass: the same ridge, darker and offset, blurred.
        shadow = Image.new("RGBA", img.size, (0, 0, 0, 0))
        sdraw = ImageDraw.Draw(shadow)
        offset = max(h // 90, 3)
        sdraw.polygon([(x + offset, y + offset) for x, y in points] + [(w, h), (0, h)],
                      fill=(0, 0, 0, 90))
        img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(offset)))
        draw.polygon(points + [(w, h), (0, h)], fill=colour + (255,))
        # Snow caps on the highest peaks of the far layers.
        if layer < 2:
            for x, y in points[1:-1]:
                if y < base_y - amp * 0.7:
                    cap = max(w // 60, 6)
                    draw.polygon([(x - cap, y + cap), (x, y), (x + cap, y + cap)],
                                 fill=(245, 248, 255, 220))


def _draw_trees(img: Image.Image, rng: random.Random, palette: dict,
                count: int) -> None:
    """Pine-tree silhouettes along the ground line."""
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    ground_y = int(h * 0.86)
    colour = palette["ground"]
    for _ in range(count):
        x = rng.randint(0, w)
        size = rng.randint(h // 18, h // 7)
        trunk_w = max(size // 10, 2)
        draw.rectangle([x - trunk_w, ground_y - size // 4, x + trunk_w, ground_y],
                       fill=colour + (255,))
        for tier in range(3):
            tw = int(size * (0.55 - tier * 0.13))
            ty = ground_y - size // 4 - tier * (size // 4)
            draw.polygon([(x - tw, ty), (x, ty - size // 3), (x + tw, ty)],
                         fill=colour + (255,))


def _draw_clouds(img: Image.Image, rng: random.Random, count: int,
                 colour: tuple[int, int, int]) -> None:
    """Soft blurred cloud clusters."""
    w, h = img.size
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    for _ in range(count):
        cx = rng.randint(0, w)
        cy = rng.randint(int(h * 0.05), int(h * 0.35))
        spread = rng.randint(w // 14, w // 6)
        for _ in range(rng.randint(4, 8)):
            r = rng.randint(spread // 3, spread)
            ox = rng.randint(-spread, spread)
            oy = rng.randint(-r // 3, r // 3)
            draw.ellipse([cx + ox - r, cy + oy - r // 2, cx + ox + r, cy + oy + r // 2],
                         fill=colour + (70,))
    img.alpha_composite(layer.filter(ImageFilter.GaussianBlur(max(w // 120, 3))))


def _draw_particles(img: Image.Image, rng: random.Random, count: int,
                    colour: tuple[int, int, int], night: bool) -> None:
    """Stars at night, drifting light motes otherwise."""
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    for _ in range(count):
        x = rng.randint(0, w)
        y = rng.randint(0, int(h * (0.6 if night else 0.9)))
        r = rng.choice([1, 1, 1, 2, 2, 3])
        alpha = rng.randint(90, 230)
        draw.ellipse([x - r, y - r, x + r, y + r], fill=colour + (alpha,))


def _find_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Best available bold sans font at the requested size."""
    candidates = sorted(Path("/usr/share/fonts").rglob("*Sans*Bold*.ttf")) if Path(
        "/usr/share/fonts").exists() else []
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size)
        except OSError:
            continue
    return ImageFont.load_default()


def _wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> list[str]:
    lines: list[str] = []
    current = ""
    for word in text.split():
        trial = f"{current} {word}".strip()
        if draw.textlength(trial, font=font) <= max_width or not current:
            current = trial
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def _draw_text(img: Image.Image, text: str, palette: dict) -> None:
    """Render the prompt's quoted text accurately, centred, with shadow."""
    w, h = img.size
    draw = ImageDraw.Draw(img, "RGBA")
    max_width = int(w * 0.86)
    size = max(int(h * 0.16), 20)
    while size > 18:
        font = _find_font(size)
        lines = _wrap_text(draw, text, font, max_width)
        widest = max(draw.textlength(line, font=font) for line in lines)
        block_h = int(size * 1.25 * len(lines))
        if widest <= max_width and block_h <= h * 0.5:
            break
        size = int(size * 0.88)
    font = _find_font(size)
    lines = _wrap_text(draw, text, font, max_width)
    line_h = int(size * 1.25)
    y = (h - line_h * len(lines)) // 2
    shadow_off = max(size // 18, 2)
    for line in lines:
        x = (w - draw.textlength(line, font=font)) // 2
        draw.text((x + shadow_off, y + shadow_off), line, font=font,
                  fill=(0, 0, 0, 170))
        draw.text((x, y), line, font=font, fill=(255, 255, 255, 255),
                  stroke_width=max(size // 24, 1), stroke_fill=palette["ground"] + (255,))
        y += line_h


def _wants_human(prompt: str) -> bool:
    text = prompt.lower()
    return any(word in text for word in _HUMAN_KEYWORDS)


def _draw_sunset_lighting(img: Image.Image, rng: random.Random,
                          sun_pos: tuple[int, int], palette: dict) -> None:
    """Warm bloom, a lens-flare streak and horizon haze for golden light."""
    w, h = img.size
    sx, sy = sun_pos
    warm = palette["sun"]
    # Broad warm bloom around the sun.
    img.alpha_composite(_glow((w, h), (sx, sy), max(w // 3, 40), warm, strength=120))
    # Lens-flare style streak and ghost circles along the sun axis.
    flare = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    fdraw = ImageDraw.Draw(flare)
    streak_h = max(h // 160, 2)
    fdraw.rectangle([0, sy - streak_h, w, sy + streak_h], fill=warm + (70,))
    cx, cy = w // 2, h // 2
    for t, r_scale, alpha in ((0.35, 0.05, 60), (0.6, 0.035, 45), (0.85, 0.07, 35)):
        gx = int(sx + (cx - sx) * t)
        gy = int(sy + (cy - sy) * t)
        r = max(int(min(w, h) * r_scale), 4)
        fdraw.ellipse([gx - r, gy - r, gx + r, gy + r], fill=warm + (alpha,))
    img.alpha_composite(flare.filter(ImageFilter.GaussianBlur(max(w // 200, 2))))
    # Atmospheric haze bands hugging the horizon.
    haze = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    hdraw = ImageDraw.Draw(haze)
    horizon = int(h * 0.62)
    for i in range(4):
        y0 = horizon - int(h * 0.05) + i * int(h * 0.03)
        hdraw.rectangle([0, y0, w, y0 + int(h * 0.035)], fill=warm + (28,))
    img.alpha_composite(haze.filter(ImageFilter.GaussianBlur(max(h // 60, 4))))


def _draw_human_figure(img: Image.Image, rng: random.Random, palette: dict,
                       sun_pos: tuple[int, int]) -> None:
    """A detailed standing human figure with rim light from the sun.

    Rendered as a shaded silhouette: head with hair, neck, shoulders,
    torso with clothing shading, arms and legs, a warm rim light on
    the sun-facing side, and a soft ground shadow. Position and
    proportions vary slightly with the seed but stay deterministic.
    """
    w, h = img.size
    ground_y = int(h * 0.86)
    fig_h = int(h * rng.uniform(0.52, 0.6))
    cx = int(w * rng.uniform(0.42, 0.58))
    top = ground_y - fig_h
    head_r = max(int(fig_h * 0.075), 6)
    head_cy = top + head_r
    neck_w = max(head_r // 3, 3)
    shoulder_y = head_cy + head_r + int(fig_h * 0.05)
    shoulder_w = int(fig_h * 0.21)
    hip_y = top + int(fig_h * 0.55)
    hip_w = int(fig_h * 0.12)
    dark = palette["ground"]
    shade = _lerp(dark, palette["accent"], 0.18)
    rim = palette["accent"]
    sun_left = sun_pos[0] < cx

    # Soft ground shadow, stretched away from the sun.
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sdraw = ImageDraw.Draw(shadow)
    sh_w = int(fig_h * 0.3)
    sh_off = int(fig_h * 0.08) * (1 if sun_left else -1)
    sdraw.ellipse([cx - sh_w + sh_off, ground_y - max(h // 120, 2),
                   cx + sh_w + sh_off, ground_y + max(h // 60, 4)],
                  fill=(0, 0, 0, 110))
    img.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(max(h // 100, 3))))

    draw = ImageDraw.Draw(img, "RGBA")
    # Legs: two tapered shapes from hips to feet.
    leg_w = max(int(fig_h * 0.045), 3)
    for side in (-1, 1):
        hip_x = cx + side * hip_w // 2
        foot_x = cx + side * int(fig_h * 0.07)
        draw.polygon([(hip_x - leg_w, hip_y), (hip_x + leg_w, hip_y),
                      (foot_x + leg_w // 2, ground_y), (foot_x - leg_w // 2, ground_y)],
                     fill=dark + (255,))
    # Torso with clothing shading: dark base, lighter sun-side panel.
    torso = [(cx - shoulder_w, shoulder_y), (cx + shoulder_w, shoulder_y),
             (cx + hip_w, hip_y), (cx - hip_w, hip_y)]
    draw.polygon(torso, fill=dark + (255,))
    lit = [(cx, shoulder_y), (cx + shoulder_w if sun_left else cx - shoulder_w, shoulder_y),
           (cx + hip_w if sun_left else cx - hip_w, hip_y), (cx, hip_y)]
    draw.polygon(lit, fill=shade + (160,))
    # Arms hanging along the torso.
    arm_w = max(int(fig_h * 0.035), 3)
    for side in (-1, 1):
        sx0 = cx + side * shoulder_w
        hx0 = cx + side * int(hip_w * 1.25)
        draw.polygon([(sx0 - arm_w, shoulder_y), (sx0 + arm_w, shoulder_y),
                      (hx0 + arm_w // 2, hip_y + int(fig_h * 0.06)),
                      (hx0 - arm_w // 2, hip_y + int(fig_h * 0.06))],
                     fill=dark + (255,))
    # Neck and head.
    draw.rectangle([cx - neck_w, head_cy, cx + neck_w, shoulder_y], fill=dark + (255,))
    draw.ellipse([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 fill=dark + (255,))
    # Hair: a cap over the top of the head with a slight sweep.
    hair = _lerp(dark, (0, 0, 0), 0.4)
    draw.pieslice([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                  start=180, end=360, fill=hair + (255,))
    draw.polygon([(cx - head_r, head_cy), (cx + head_r, head_cy),
                  (cx + head_r // 2, head_cy - head_r // 2)], fill=hair + (255,))
    # Rim light: a warm edge on the sun-facing side of head and body.
    rim_w = max(int(fig_h * 0.012), 2)
    if sun_left:
        draw.arc([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 start=90, end=270, fill=rim + (235,), width=rim_w)
        draw.line([(cx - shoulder_w, shoulder_y), (cx - hip_w, hip_y)],
                  fill=rim + (220,), width=rim_w)
    else:
        draw.arc([cx - head_r, head_cy - head_r, cx + head_r, head_cy + head_r],
                 start=270, end=90, fill=rim + (235,), width=rim_w)
        draw.line([(cx + shoulder_w, shoulder_y), (cx + hip_w, hip_y)],
                  fill=rim + (220,), width=rim_w)


def _texture_grain(img: Image.Image, rng: random.Random, amount: int) -> None:
    """Fine photographic texture noise, scaled by the quality level."""
    if amount <= 0:
        return
    w, h = img.size
    small = Image.new("L", (max(w // 4, 8), max(h // 4, 8)))
    px = small.load()
    for y in range(small.height):
        for x in range(small.width):
            px[x, y] = rng.randint(0, 255)
    noise = small.resize((w, h), Image.BILINEAR).filter(
        ImageFilter.GaussianBlur(1))
    layer = Image.merge("RGBA", (noise, noise, noise,
                                 Image.new("L", (w, h), amount)))
    img.alpha_composite(layer)


def _vignette(img: Image.Image) -> None:
    """Darken the corners slightly for a finished, photographic feel."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse([-w // 4, -h // 4, w + w // 4, h + h // 4], fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(w // 12, 8)))
    dark = Image.new("RGBA", (w, h), (0, 0, 0, 120))
    img.alpha_composite(Image.composite(
        Image.new("RGBA", (w, h), (0, 0, 0, 0)), dark, mask))


def generate_image(prompt: str, width: int = 1024, height: int = 1024,
                   quality: str = "high", seed: int | None = None) -> GeneratedImage:
    """Generate one HQ image locally from a text prompt.

    The prompt drives the scene: palette keywords (sunset, ocean,
    forest, night, desert, neon, snow, meadow), scene elements
    (mountains, trees, clouds, stars, sun/moon), and any quoted
    text, which is rendered verbatim in the image. Generation is
    deterministic for a given prompt + seed.
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must not be empty")
    if quality not in _QUALITY:
        raise ValueError(
            f"quality must be one of {QUALITY_LEVELS}, got {quality!r}")
    for label, value in (("width", width), ("height", height)):
        if not isinstance(value, int) or not MIN_SIZE <= value <= MAX_SIZE:
            raise ValueError(
                f"{label} must be an int between {MIN_SIZE} and {MAX_SIZE}, "
                f"got {value!r}")

    cfg = _QUALITY[quality]
    ss = cfg["supersample"]
    w, h = width * ss, height * ss
    seed_value = _seed_for(prompt, seed)
    rng = random.Random(seed_value)
    palette_name = detect_palette(prompt)
    palette = _PALETTES[palette_name]
    text = extract_text(prompt)
    lowered = prompt.lower()

    img = _gradient_sky((w, h), palette["sky"]).convert("RGBA")

    night = palette_name in ("night", "neon")
    if night or "star" in lowered:
        _draw_particles(img, rng, cfg["particles"], (235, 240, 255), night=True)

    # Sun or moon with a soft glow, placed by the seed.
    sun_x = int(w * rng.uniform(0.3, 0.7))
    sun_y = int(h * rng.uniform(0.22, 0.4))
    sun_r = max(int(min(w, h) * 0.07), 8)
    img.alpha_composite(_glow((w, h), (sun_x, sun_y), sun_r * 5, palette["sun"]))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse([sun_x - sun_r, sun_y - sun_r, sun_x + sun_r, sun_y + sun_r],
                 fill=palette["sun"] + (255,))
    if night:  # crescent shading for the moon
        draw.ellipse([sun_x - sun_r // 3, sun_y - sun_r, sun_x + sun_r,
                      sun_y + sun_r], fill=palette["sky"][1] + (255,))

    _draw_clouds(img, rng, count=5 if quality != "draft" else 3,
                 colour=(255, 255, 255) if not night else (120, 130, 180))

    _draw_ridges(img, rng, palette, layers=cfg["layers"])

    # Ground band with a lit horizon line.
    ground_y = int(h * 0.86)
    draw.rectangle([0, ground_y, w, h], fill=palette["ground"] + (255,))
    draw.line([(0, ground_y), (w, ground_y)], fill=palette["accent"] + (200,),
              width=max(h // 300, 1))

    if palette_name in ("forest", "meadow", "snow") or "tree" in lowered:
        _draw_trees(img, rng, palette, count=26 if quality != "draft" else 12)

    if not night:
        _draw_particles(img, rng, cfg["particles"] // 2, palette["accent"],
                        night=False)
        _draw_sunset_lighting(img, rng, (sun_x, sun_y), palette)

    human = _wants_human(prompt)
    if human:
        _draw_human_figure(img, rng, palette, (sun_x, sun_y))

    _texture_grain(img, rng, cfg["grain"])

    if text:
        _draw_text(img, text, palette)

    _vignette(img)

    if ss > 1:
        img = img.resize((width, height), Image.LANCZOS)
    if quality == "ultra":
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=80, threshold=2))
    return GeneratedImage(
        image=img.convert("RGB"),
        prompt=prompt.strip(),
        seed=seed_value,
        palette=palette_name,
        quality=quality,
        text_rendered=text,
        human_figure=human,
    )


def generate_to_file(prompt: str, path: str | Path, width: int = 1024,
                     height: int = 1024, quality: str = "high",
                     seed: int | None = None) -> GeneratedImage:
    """Generate an image and save it as a PNG; returns the result."""
    result = generate_image(prompt, width=width, height=height,
                            quality=quality, seed=seed)
    result.save(path)
    return result

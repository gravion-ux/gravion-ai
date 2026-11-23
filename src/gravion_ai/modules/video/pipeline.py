"""The local render pipeline for the video module (Piece 4).

`VideoModule.run` briefs **Gravion Video** — the trained model —
exactly like the other modules brief theirs. This pipeline is the
module's other path: turning a request into an actual MP4 on the
machine the deployment runs on, with no network and no trained
weights. It exists so a deployment can develop, demo and test its
video workflow end-to-end before the trained bundle is installed.

What the local renderer is, honestly:

- It is a **development renderer**, not the trained Gravion Video
  model, and it does not pretend to be: the result records the
  provider name (`gravion-local-renderer`) so nothing downstream
  can mistake its output for the model's.
- With no reference frames it renders an animated gradient clip
  whose colours and seed derive deterministically from the prompt
  (same prompt, same clip spec); with reference frames it animates
  the first frame with a slow zoom (Ken Burns style).
- It renders **short clips only** — up to
  `MAX_LOCAL_DURATION_SECONDS`. Long-form output (up to ~1 hour)
  is produced by the scene-planning and stitching pipeline that
  ships with the trained Gravion Video bundle; asking the local
  renderer for more than the cap is an error, not a silent
  truncation.
- The long edge of a render is capped at 1280 px: this is a
  development preview, not a mastering tool.

Rendering shells out to ffmpeg (libx264). A missing ffmpeg, a
missing reference frame, or an over-cap duration raises
`ProviderError` — the pipeline never reports success for a file it
did not actually write.
"""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

from ...errors import ProviderError
from .module import detect_mode, parse_aspect, parse_duration, parse_reference_frames

DEFAULT_DURATION_SECONDS = 4
DEFAULT_ASPECT = "16:9"
DEFAULT_FPS = 24
MAX_LOCAL_DURATION_SECONDS = 60
MAX_LONG_EDGE = 1280

# Named aspect ratios the renderer has fixed dimensions for.
ASPECT_DIMENSIONS = {
    "1:1": (640, 640),
    "3:4": (720, 960),
    "4:3": (960, 720),
    "16:9": (1280, 720),
    "9:16": (720, 1280),
    "21:9": (1280, 548),
}

# Gradient colours the local renderer picks from, by prompt hash.
_PALETTE = (
    "0x1F3B73",
    "0x7B2D8B",
    "0x0E7C7B",
    "0xC75146",
    "0x2E5EAA",
    "0x8C5E10",
    "0x274156",
    "0x5B2A86",
)

_SIZE_RE = re.compile(r"^(\d{3,5})x(\d{3,5})$")
_RATIO_RE = re.compile(r"^(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)$")


def _even(value: float) -> int:
    """Round down to an even pixel count (libx264 requires even
    dimensions), never below 16."""
    return max(16, int(value) // 2 * 2)


def _dimensions_for(aspect: str) -> tuple[int, int]:
    """Resolve an aspect (named ratio, ratio string or explicit
    pixel size) to the renderer's (width, height)."""
    if aspect in ASPECT_DIMENSIONS:
        return ASPECT_DIMENSIONS[aspect]
    size = _SIZE_RE.match(aspect)
    if size:
        width, height = int(size.group(1)), int(size.group(2))
    else:
        ratio = _RATIO_RE.match(aspect)
        if not ratio:
            return ASPECT_DIMENSIONS[DEFAULT_ASPECT]
        rw, rh = float(ratio.group(1)), float(ratio.group(2))
        width, height = MAX_LONG_EDGE, MAX_LONG_EDGE * rh / rw
    scale = min(1.0, MAX_LONG_EDGE / max(width, height))
    return _even(width * scale), _even(height * scale)


@dataclass(frozen=True)
class VideoJob:
    """A fully-specified local render: everything the renderer
    needs, derived from the request by `job_from_prompt`."""

    prompt: str
    mode: str
    duration_seconds: int
    aspect: str
    width: int
    height: int
    reference_frames: tuple[str, ...] = ()
    fps: int = DEFAULT_FPS


@dataclass(frozen=True)
class VideoResult:
    """What a render produced: the file, who rendered it, and the
    spec it was rendered to."""

    path: Path
    provider: str
    duration_seconds: int
    width: int
    height: int


def job_from_prompt(prompt: str) -> VideoJob:
    """Derive a render job from a plain-language request, using the
    module's own parsers. Unstated duration/aspect fall back to the
    honest defaults (4 seconds, 16:9) rather than being invented
    per-request."""
    aspect = parse_aspect(prompt) or DEFAULT_ASPECT
    duration = parse_duration(prompt)
    if duration is None:
        duration = DEFAULT_DURATION_SECONDS
    width, height = _dimensions_for(aspect)
    return VideoJob(
        prompt=prompt.strip(),
        mode=detect_mode(prompt),
        duration_seconds=duration,
        aspect=aspect,
        width=width,
        height=height,
        reference_frames=tuple(parse_reference_frames(prompt)),
        fps=DEFAULT_FPS,
    )


@runtime_checkable
class VideoProvider(Protocol):
    """A video renderer: takes a job, writes an MP4, reports what
    it produced. The trained Gravion Video bundle and the local
    development renderer both sit behind this interface."""

    name: str

    def render(self, job: VideoJob, out_path: Path) -> VideoResult:
        ...


class LocalVideoProvider:
    """The offline development renderer (ffmpeg under the hood).

    See the module docstring for what it is and is not: short,
    deterministic, honestly-labelled preview clips — never a
    stand-in claiming to be the trained model.
    """

    name = "gravion-local-renderer"

    def render(self, job: VideoJob, out_path: Path) -> VideoResult:
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise ProviderError(
                "the local video renderer needs ffmpeg on PATH; "
                "install ffmpeg or use the trained Gravion Video bundle"
            )
        if job.duration_seconds < 1:
            raise ProviderError(
                f"video duration must be at least 1 second, got "
                f"{job.duration_seconds}"
            )
        if job.duration_seconds > MAX_LOCAL_DURATION_SECONDS:
            raise ProviderError(
                f"the local renderer renders clips of up to "
                f"{MAX_LOCAL_DURATION_SECONDS} seconds; a "
                f"{job.duration_seconds}-second render needs the "
                f"long-form scene-planning and stitching pipeline, "
                f"which ships with the trained Gravion Video bundle"
            )
        for frame in job.reference_frames:
            if not Path(frame).exists():
                raise ProviderError(f"reference frame not found: {frame}")
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg,
            "-y",
            *self._source_args(job),
            "-t",
            str(job.duration_seconds),
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0 or not out_path.exists() or out_path.stat().st_size == 0:
            detail = proc.stderr.strip().splitlines()
            tail = detail[-1] if detail else "no output produced"
            raise ProviderError(f"ffmpeg failed to render {out_path}: {tail}")
        return VideoResult(
            path=out_path,
            provider=self.name,
            duration_seconds=job.duration_seconds,
            width=job.width,
            height=job.height,
        )

    def _source_args(self, job: VideoJob) -> list[str]:
        if job.reference_frames:
            frames = job.duration_seconds * job.fps
            zoom = (
                f"scale={job.width}:{job.height}"
                f":force_original_aspect_ratio=increase,"
                f"crop={job.width}:{job.height},"
                f"zoompan=z='min(zoom+0.0015,1.5)'"
                f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
                f":d={frames}:s={job.width}x{job.height}:fps={job.fps}"
            )
            return ["-i", job.reference_frames[0], "-vf", zoom]
        digest = hashlib.sha256(job.prompt.encode("utf-8")).digest()
        colours = [_PALETTE[digest[i] % len(_PALETTE)] for i in range(3)]
        seed = int.from_bytes(digest[:4], "big") % (2**31)
        source = (
            f"gradients=s={job.width}x{job.height}:r={job.fps}"
            f":c0={colours[0]}:c1={colours[1]}:c2={colours[2]}"
            f":nb_colors=3:speed=0.02:seed={seed}"
        )
        return ["-f", "lavfi", "-i", source]


def render_video(
    prompt: str,
    out_path: Path,
    provider: VideoProvider | None = None,
) -> VideoResult:
    """Render a request to an MP4: derive the job, then render it
    with the given provider (the local development renderer by
    default)."""
    job = job_from_prompt(prompt)
    return (provider or LocalVideoProvider()).render(job, Path(out_path))

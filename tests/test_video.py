import json
import shutil
import subprocess

import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError, ProviderError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.video.module import (
    MODE_EDIT,
    MODE_GENERATE,
    build_brief,
    detect_mode,
    parse_aspect,
    parse_duration,
    parse_reference_frames,
)
from gravion_ai.modules.video.pipeline import (
    LocalVideoProvider,
    job_from_prompt,
    render_video,
)
from gravion_ai.providers import GravionCoreProvider
from gravion_ai.registry import ModuleContext, default_registry
from gravion_ai.runtime import GravionRuntime

requires_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH"
)


class RecordingProvider:
    name = "recording"

    def __init__(self):
        self.calls = []

    def complete(self, messages, model=None):
        self.calls.append((list(messages), model))
        return "recorded reply"


def _ctx(provider, model="gravion-video"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def _assert_real_mp4(path):
    assert path.exists()
    data = path.read_bytes()
    assert len(data) > 0
    assert data[4:8] == b"ftyp"  # MP4 box header — a real container


def _probe(path):
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        return None
    out = subprocess.run(
        [ffprobe, "-v", "error", "-show_entries",
         "stream=codec_type,width,height:format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True,
    )
    return json.loads(out.stdout)


def test_video_instantiates_and_answers():
    module = default_registry().instantiate("video")
    out = module.run(_ctx(GravionCoreProvider()), "a teaser for our app launch")
    assert "a teaser for our app launch" in out


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("edit: trim the intro", MODE_EDIT),
        ("Edit this video to remove the watermark", MODE_EDIT),
        ("please edit the product demo", MODE_EDIT),
        ("modify: swap the ending for the new logo", MODE_EDIT),
        ("Modify the launch video to be shorter", MODE_EDIT),
        ("a teaser for our app launch", MODE_GENERATE),
        ("make a vertical video for the launch", MODE_GENERATE),
        ("generate a product demo", MODE_GENERATE),
    ],
)
def test_detect_mode(prompt, expected):
    assert detect_mode(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a 30 second teaser", 30),
        ("a 45-second sting", 45),
        ("a 2 minute product video", 120),
        ("a 3-minute explainer", 180),
        ("a teaser for our app", None),
        ("a video for the launch", None),
    ],
)
def test_parse_duration(prompt, expected):
    assert parse_duration(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a teaser at 1920x1080 for the site", "1920x1080"),
        ("a clip, 512 x 512", "512x512"),
        ("a square video for the feed", "1:1"),
        ("a vertical video for the launch", "9:16"),
        ("a portrait video of the founder", "9:16"),
        ("a landscape video of the office", "16:9"),
        ("a wide banner video for the homepage", "21:9"),
        # An explicit size beats a named shape.
        ("a vertical video at 1200x628", "1200x628"),
        ("a teaser for our app", None),
    ],
)
def test_parse_aspect(prompt, expected):
    assert parse_aspect(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("animate frames/intro.png into a teaser", ["frames/intro.png"]),
        (
            "build a video from frames/a.png and frames/b.jpg",
            ["frames/a.png", "frames/b.jpg"],
        ),
        ("use cover.webp as the opening frame", ["cover.webp"]),
        ("edit this video: replace the frame shot.jpeg", ["shot.jpeg"]),
        # A repeated frame is recorded once.
        ("animate logo.png and logo.png again", ["logo.png"]),
        ("a teaser for our app", []),
    ],
)
def test_parse_reference_frames(prompt, expected):
    assert parse_reference_frames(prompt) == expected


def test_brief_records_mode_duration_aspect_frames_and_request():
    brief = build_brief("edit: trim the intro of this 30 second vertical video")
    assert "Mode: edit" in brief
    assert "Duration: 30 seconds" in brief
    assert "Aspect: 9:16" in brief
    assert "Reference frames: none" in brief
    assert "Request: edit: trim the intro of this 30 second vertical video" in brief
    brief = build_brief("animate frames/intro.png into a teaser")
    assert "Mode: generate" in brief
    assert "Duration: model default" in brief
    assert "Aspect: model default" in brief
    assert "Reference frames: frames/intro.png" in brief


@pytest.mark.parametrize(
    "prompt,marker",
    [
        ("a teaser for our app", "Video mode: generate"),
        ("edit: trim the intro", "Video mode: edit"),
    ],
)
def test_video_guidance_and_brief_reach_the_provider(prompt, marker):
    provider = RecordingProvider()
    module = default_registry().instantiate("video")
    out = module.run(_ctx(provider), prompt)
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-video"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert marker in messages[0]["content"]
    assert "# Video brief" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": prompt}


def test_video_runtime_dev_end_to_end_with_gravion_video(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-video"))
    assert runtime.resolve_model("video") == "gravion-video"
    out = runtime.video("a 30 second teaser for our app")
    assert "a 30 second teaser for our app" in out


def test_video_runtime_rejects_model_without_video_capability(make_config):
    # Gravion Core serves chat + support, not video: the registry
    # check must refuse rather than answer with the wrong model.
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.video("a 30 second teaser for our app")


def test_video_manifest_signs_and_verifies(issuer, video_module_copy):
    payload = build_manifest(
        video_module_copy,
        name="video",
        module_type="video",
        version="0.1.0",
        entrypoint="gravion_ai.modules.video.module:VideoModule",
    )
    write_signed_manifest(video_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(video_module_copy, issuer)
    assert verified["name"] == "video"
    assert verified["type"] == "video"
    assert "module.py" in verified["files"]
    assert "pipeline.py" in verified["files"]


def test_bundled_video_manifest_is_unsigned(issuer):
    import gravion_ai.modules.video as video_pkg
    from pathlib import Path

    with pytest.raises(ModuleTamperError):
        verify_module(Path(video_pkg.__file__).parent, issuer)


# ---------------------------------------------------------------------------
# the render pipeline
# ---------------------------------------------------------------------------

def test_job_from_prompt_defaults():
    job = job_from_prompt("a teaser for our app")
    assert job.mode == MODE_GENERATE
    assert job.duration_seconds == 4
    assert job.aspect == "16:9"
    assert (job.width, job.height) == (1280, 720)
    assert job.reference_frames == ()
    assert job.fps == 24


def test_job_from_prompt_derives_the_full_spec():
    job = job_from_prompt("edit: extend frames/opening.png into a 6 second vertical video")
    assert job.mode == MODE_EDIT
    assert job.duration_seconds == 6
    assert job.aspect == "9:16"
    assert (job.width, job.height) == (720, 1280)
    assert job.reference_frames == ("frames/opening.png",)


def test_job_from_prompt_explicit_size_is_capped_for_the_local_renderer():
    job = job_from_prompt("a 10 second video at 1920x1080")
    assert job.aspect == "1920x1080"
    assert (job.width, job.height) == (1280, 720)


def test_job_derivation_is_deterministic():
    prompt = "a 5 second square video from frames/a.png"
    assert job_from_prompt(prompt) == job_from_prompt(prompt)


@requires_ffmpeg
def test_local_render_produces_a_real_mp4(tmp_path):
    out = tmp_path / "teaser.mp4"
    result = LocalVideoProvider().render(
        job_from_prompt("a 2 second teaser for our app"), out
    )
    _assert_real_mp4(out)
    assert result.path == out
    assert result.provider == "gravion-local-renderer"
    assert result.duration_seconds == 2
    assert (result.width, result.height) == (1280, 720)
    probe = _probe(out)
    if probe is not None:
        (stream,) = probe["streams"]
        assert stream["codec_type"] == "video"
        assert (stream["width"], stream["height"]) == (1280, 720)
        assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=0.6)


@requires_ffmpeg
def test_reference_frame_render(tmp_path):
    frame = tmp_path / "opening.png"
    subprocess.run(
        [shutil.which("ffmpeg"), "-y", "-f", "lavfi",
         "-i", "color=c=red:s=320x240", "-frames:v", "1", str(frame)],
        capture_output=True, text=True, check=True,
    )
    out = tmp_path / "from-frame.mp4"
    result = render_video(f"animate {frame} into a 2 second video", out)
    _assert_real_mp4(out)
    assert result.provider == "gravion-local-renderer"
    probe = _probe(out)
    if probe is not None:
        assert float(probe["format"]["duration"]) == pytest.approx(2.0, abs=0.6)


def test_missing_reference_frame_fails(tmp_path):
    job = job_from_prompt("animate frames/nope.png into a video")
    with pytest.raises(ProviderError, match="reference frame not found"):
        LocalVideoProvider().render(job, tmp_path / "clip.mp4")


def test_over_cap_duration_fails(tmp_path):
    # Long-form is the trained bundle's stitching pipeline, not the
    # local renderer's job — it must refuse, not truncate.
    job = job_from_prompt("a 2 minute product video")
    assert job.duration_seconds == 120
    with pytest.raises(ProviderError, match="long-form"):
        LocalVideoProvider().render(job, tmp_path / "clip.mp4")


@requires_ffmpeg
def test_module_render_delegates_to_the_pipeline(tmp_path):
    module = default_registry().instantiate("video")
    out = tmp_path / "clip.mp4"
    result = module.render("a 2 second teaser for our app", out)
    _assert_real_mp4(out)
    assert result.provider == "gravion-local-renderer"


@requires_ffmpeg
def test_runtime_render_video_goes_through_the_gate_path(make_config, tmp_path):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-video"))
    out = tmp_path / "clip.mp4"
    result = runtime.render_video("a 2 second teaser for our app", out)
    _assert_real_mp4(out)
    assert result.provider == "gravion-local-renderer"
    assert (result.width, result.height) == (1280, 720)

import pytest

from gravion_ai.errors import ModuleError, ModuleTamperError
from gravion_ai.licensing.modules_sig import (
    build_manifest,
    sign_manifest,
    verify_module,
    write_signed_manifest,
)
from gravion_ai.modules.music.module import (
    KIND_JINGLE,
    KIND_SCORE,
    KIND_TRACK,
    build_brief,
    detect_kind,
    parse_duration,
    parse_moods,
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


def _ctx(provider, model="gravion-audio"):
    return ModuleContext("You are Gravion, an AI assistant for Acme.", provider, model=model)


def test_music_instantiates_and_answers():
    module = default_registry().instantiate("music")
    out = module.run(_ctx(GravionCoreProvider()), "an upbeat track for our launch video")
    assert "Track:" in out


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a jingle for our radio ad", KIND_JINGLE),
        ("write a Jingle for the brand", KIND_JINGLE),
        ("background music for the product demo", KIND_SCORE),
        ("an underscore for the training video", KIND_SCORE),
        ("a soundtrack for our documentary", KIND_SCORE),
        ("bgm for the lobby playlist", KIND_SCORE),
        # A jingle is named explicitly, so it wins over score phrasing.
        ("a jingle to use as background music in ads", KIND_JINGLE),
        ("an upbeat track for the launch", KIND_TRACK),
        ("a calm piano piece", KIND_TRACK),
    ],
)
def test_detect_kind(prompt, expected):
    assert detect_kind(prompt) == expected


@pytest.mark.parametrize(
    "prompt,expected",
    [
        ("a 30 second jingle", 30),
        ("a 45-second sting", 45),
        ("a 2 minute track", 120),
        ("a 3-minute background score", 180),
        # Regression: compound durations kept only their first
        # part, fractional minutes were misread ("1.5 minutes"
        # became 300 via the "5"), and hours were not a unit.
        ("a 2 minutes 30 seconds track", 150),
        ("a 1.5 minutes track", 90),
        ("a 1 hour mix", 3600),
        ("a jingle for our ad", None),
        ("a track for the launch", None),
    ],
)
def test_parse_duration(prompt, expected):
    assert parse_duration(prompt) == expected


def test_parse_moods_in_request_order():
    assert parse_moods("a calm and upbeat track") == ["calm", "upbeat"]
    assert parse_moods("an epic cinematic score") == ["epic", "cinematic"]
    assert parse_moods("a lo-fi beat") == ["lo-fi"]
    assert parse_moods("a track for the launch") == []
    # A mood word embedded in a longer word is not a mood.
    assert parse_moods("a swarm of bees over the epicentre") == []
    assert parse_moods("darken the warmup track") == []


def test_brief_records_kind_duration_mood_and_request():
    brief = build_brief("a 30 second upbeat jingle")
    assert "Kind: jingle" in brief
    assert "Duration: 30 seconds" in brief
    assert "Mood: upbeat" in brief
    assert "Request: a 30 second upbeat jingle" in brief
    brief = build_brief("a track for the launch")
    assert "Kind: track" in brief
    assert "Duration: model default" in brief
    assert "Mood: model default" in brief


@pytest.mark.parametrize(
    "prompt,marker",
    [
        ("a track for the launch", "Music mode: track"),
        ("a jingle for our ad", "Music mode: jingle"),
        ("background music for the demo", "Music mode: score"),
    ],
)
def test_music_guidance_and_brief_reach_the_provider(prompt, marker):
    provider = RecordingProvider()
    module = default_registry().instantiate("music")
    out = module.run(_ctx(provider), prompt)
    assert out == "recorded reply"
    (messages, model), = provider.calls
    assert model == "gravion-audio"
    assert messages[0]["role"] == "system"
    assert messages[0]["content"].startswith("You are Gravion")
    assert marker in messages[0]["content"]
    assert "# Music brief" in messages[0]["content"]
    assert messages[1] == {"role": "user", "content": prompt}


def test_music_runtime_dev_end_to_end_with_gravion_audio(make_config):
    runtime = GravionRuntime(make_config(mode="development", model="gravion-audio"))
    assert runtime.resolve_model("music") == "gravion-audio"
    out = runtime.music("a 30 second jingle")
    assert "a 30 second jingle" in out


def test_music_runtime_rejects_model_without_music_capability(make_config):
    # Gravion Core serves chat + support, not music: the registry
    # check must refuse rather than answer with the wrong model.
    runtime = GravionRuntime(make_config(mode="development", model="gravion-core"))
    with pytest.raises(ModuleError, match="does not serve"):
        runtime.music("a 30 second jingle")


def test_music_manifest_signs_and_verifies(issuer, music_module_copy):
    payload = build_manifest(
        music_module_copy,
        name="music",
        module_type="music",
        version="0.1.0",
        entrypoint="gravion_ai.modules.music.module:MusicModule",
    )
    write_signed_manifest(music_module_copy, sign_manifest(payload, issuer))
    verified = verify_module(music_module_copy, issuer)
    assert verified["name"] == "music"
    assert verified["type"] == "music"
    assert "module.py" in verified["files"]


def test_bundled_music_manifest_is_unsigned(issuer):
    import gravion_ai.modules.music as music_pkg
    from pathlib import Path

    with pytest.raises(ModuleTamperError):
        verify_module(Path(music_pkg.__file__).parent, issuer)

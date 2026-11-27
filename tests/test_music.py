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
        # Regression: kind phrases matched as substrings, so
        # "bgm" buried in a longer word read as a score, and a
        # negated kind was recorded as the kind requested.
        ("subgmusic vibes for the set", KIND_TRACK),
        ("not a jingle, a full track for the launch", KIND_TRACK),
        ("no background music, a standalone track", KIND_TRACK),
        # Regression: music's negator list lacked "without", so
        # a kind named only to be ruled out won, and a "-free"
        # suffix read as the kind it excludes.
        ("background music without a jingle feel", KIND_SCORE),
        ("a jingle-free track for our app", KIND_TRACK),
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
        # Compact units and clock forms state the same lengths;
        # an aspect ratio is never read as a clock duration.
        ("a 30s sting", 30),
        ("a 1m 30s track", 90),
        ("a 1m30s track", 90),
        ("a 1:30 jingle", 90),
        ("a 0:45 sting", 45),
        ("a 1:02:30 mix", 3750),
        ("a 9:16 track", None),
        ("a jingle for our ad", None),
        ("a track for the launch", None),
        # Word-form durations state the same lengths as digit
        # forms; the article counts for minutes/hours only, so
        # an ordinal "second version" is not one second.
        ("thirty seconds of calm piano", 30),
        ("two minutes of lofi", 120),
        ("a minute long jingle", 60),
        ("twenty-five seconds", 25),
        ("one hour of background music", 3600),
        ("half a minute of music", 30),
        ("two minutes thirty seconds", 150),
        ("make a second version, calm", None),
    ],
)
def test_parse_duration(prompt, expected):
    assert parse_duration(prompt) == expected


def test_parse_moods_in_request_order():
    assert parse_moods("a calm and upbeat track") == ["calm", "upbeat"]
    assert parse_moods("an epic cinematic score") == ["epic", "cinematic"]
    assert parse_moods("a lo-fi beat") == ["lo-fi"]
    # Regression: the plain feeling words a request states
    # first ("a sad piano melody", "a cheerful jingle")
    # recorded no mood at all.
    assert parse_moods("a sad piano melody") == ["sad"]
    assert parse_moods("a cheerful jingle") == ["cheerful"]
    assert parse_moods("a happy and sad song") == ["happy", "sad"]
    assert parse_moods("a track for the launch") == []
    # A mood word embedded in a longer word is not a mood.
    assert parse_moods("a swarm of bees over the epicentre") == []
    assert parse_moods("darken the warmup track") == []


def test_parse_moods_skips_negated_moods():
    # Regression: moods had no negation guard, so a ruled-out
    # mood was recorded in the brief as one the request asked for.
    assert parse_moods("not calm, make it upbeat") == ["upbeat"]
    assert parse_moods("no dark tones, warm and playful") == ["warm", "playful"]
    assert parse_moods("calm but not dark") == ["calm"]
    assert parse_moods("never dark, always warm") == ["warm"]
    assert parse_moods("don't make it epic, keep it calm") == ["calm"]
    assert parse_moods("no lofi please, cinematic only") == ["cinematic"]
    # Regression: "without" was missing from the negators, so a
    # mood named only to be ruled out was recorded as requested.
    assert parse_moods("a track without dark tones, warm and playful") == [
        "warm", "playful",
    ]
    assert parse_moods("without calm, upbeat only") == ["upbeat"]
    # A mood named again without a negator still counts, in the
    # order of that later mention.
    assert parse_moods("not dark, actually dark and epic") == ["dark", "epic"]
    # The brief records only the moods the request keeps.
    assert "Mood: upbeat" in build_brief("not calm, an upbeat track")


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


def test_plain_feeling_moods():
    # Regression: the plain feeling words beside
    # sad/happy/cheerful recorded no mood at all.
    assert parse_moods("energetic workout music") == ["energetic"]
    assert parse_moods("romantic song") == ["romantic"]
    assert parse_moods("peaceful piano") == ["peaceful"]
    assert parse_moods("calm relaxing tune") == ["calm", "relaxing"]
    assert parse_moods("angry metal song") == ["angry"]
    assert parse_moods("happy but not energetic") == ["happy"]

import pytest

from better_meeting import asr
from better_meeting.asr.core import merge_passes, pipeline_meta, transcribe, transcribe_range
from better_meeting.asr.langid import guess_lang, label_segments
from better_meeting.utils import BmError, save


def seg(start, end, text, lang, score=-0.5, nospeech=0.0):
    return {
        "start": start,
        "end": end,
        "text": text,
        "lang": lang,
        "score": score,
        "nospeech": nospeech,
    }


# --- merge -----------------------------------------------------------------


def test_merge_picks_most_confident_pass_per_stretch():
    uk = [seg(0, 5, "uk-a", "uk", -0.1), seg(5, 10, "uk-b", "uk", -0.9)]
    en = [seg(0, 5, "en-a", "en", -0.8), seg(5, 10, "en-b", "en", -0.2)]
    assert [s["text"] for s in merge_passes([uk, en])] == ["uk-a", "en-b"]


def test_merge_drops_hallucinations():
    passes = [[seg(0, 3, "real", "en", -0.3), seg(3, 6, "ghost", "en", -1.5, nospeech=0.9)]]
    assert [s["text"] for s in merge_passes(passes)] == ["real"]


def test_merge_jumps_over_long_gaps():
    passes = [[seg(0, 2, "a", "en"), seg(30, 32, "b", "en")]]
    assert [s["text"] for s in merge_passes(passes)] == ["a", "b"]


def test_merge_empty():
    assert merge_passes([[], []]) == []


# --- language heuristic ------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "prev", "expected"),
    [
        ("Let's ship it", None, "en"),
        ("Це вже готово, дякую", None, "uk"),
        ("Это уже готово, спасибо", None, "ru"),
        ("Окей", "ru", "ru"),  # no markers: inherit
        ("Окей", None, "uk"),  # no markers, no context: default
        ("12:40", "uk", "uk"),  # no letters: inherit
    ],
)
def test_guess_lang(text, prev, expected):
    assert guess_lang(text, prev) == expected


def test_label_segments_carries_context():
    segs = [{"text": "Это важно"}, {"text": "Ага"}, {"text": "OK then"}]
    label_segments(segs)
    assert [s["lang"] for s in segs] == ["ru", "ru", "en"]


# --- registry ----------------------------------------------------------------


def test_auto_backend_is_platform_default(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Linux")
    assert asr.resolve_backend_name("auto") == "faster"
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    assert asr.resolve_backend_name("auto") == "mlx"


def test_unknown_backend():
    with pytest.raises(BmError, match="unknown ASR backend"):
        asr.get_backend("nope")


def test_parakeet_model_resolution():
    from better_meeting.asr.providers.parakeet import PARAKEET_MODEL, ParakeetBackend

    p = ParakeetBackend()
    assert p.resolve_model(None) == PARAKEET_MODEL
    assert p.resolve_model("large-v3-turbo") == PARAKEET_MODEL
    assert p.resolve_model("org/custom") == "org/custom"
    with pytest.raises(BmError):
        p.resolve_model("tiny")


# --- transcribe: per-language passes ----------------------------------------


def test_auto_runs_one_cached_pass_per_language(tmp_path, fake_asr):
    out = transcribe(tmp_path / "a.wav", "auto", ["uk", "en"], None, "fake", tmp_path)
    assert [(s["text"], s["lang"]) for s in out] == [("uk 0", "uk"), ("en 5", "en")]
    assert set(out[0]) == {"start", "end", "text", "lang"}  # scores stripped
    assert (tmp_path / "pass_uk.json").exists() and (tmp_path / "pass_en.json").exists()

    transcribe(tmp_path / "a.wav", "auto", ["uk", "en"], None, "fake", tmp_path)
    assert len(fake_asr.calls) == 2  # second run fully cached

    transcribe(
        tmp_path / "a.wav", "auto", ["uk", "en"], None, "fake", tmp_path, {"temperature": 0.2}
    )
    assert len(fake_asr.calls) == 4  # new options -> new passes


def test_pinned_language_is_one_pass(tmp_path, fake_asr):
    out = transcribe(tmp_path / "a.wav", "uk", ["uk", "en"], None, "fake", tmp_path)
    assert {s["lang"] for s in out} == {"uk"}
    assert fake_asr.calls == [("uk", {})]


# --- transcribe: single-pass providers --------------------------------------


def test_single_pass_runs_once_and_relabels(tmp_path, fake_single):
    auto = transcribe(tmp_path / "a.wav", "auto", ["uk", "ru", "en"], None, "fake1", tmp_path)
    assert [s["lang"] for s in auto] == ["en", "uk"]
    pinned = transcribe(tmp_path / "a.wav", "uk", [], None, "fake1", tmp_path)
    assert {s["lang"] for s in pinned} == {"uk"}
    assert fake_single.calls == 1  # pass cache ignores --lang
    assert (tmp_path / "pass_fake1.json").exists()


def test_single_pass_rejects_options(tmp_path, fake_single):
    with pytest.raises(BmError, match="-O options"):
        transcribe(tmp_path / "a.wav", "auto", [], None, "fake1", tmp_path, {"x": 1})


# --- transcribe_range reuses the pipeline cache ------------------------------


def test_range_sliced_from_pipeline_cache(tmp_path, fake_asr):
    merged = transcribe(tmp_path / "a.wav", "auto", ["uk", "en"], None, "fake", tmp_path)
    save(tmp_path / "transcript.json", merged)
    save(
        tmp_path / "transcript.meta.json",
        pipeline_meta("auto", ["uk", "en"], "fake-model", "fake", {}),
    )
    calls = len(fake_asr.calls)

    video = tmp_path / "missing.mp4"  # never opened: no ASR run, no audio extraction
    auto = transcribe_range(video, 6, None, None, None, "fake", tmp_path, {}, False)
    assert [s["text"] for s in auto] == ["en 5"]
    pinned = transcribe_range(video, 0, 4, "uk", None, "fake", tmp_path, {}, False)
    assert [s["text"] for s in pinned] == ["uk 0"]
    assert len(fake_asr.calls) == calls

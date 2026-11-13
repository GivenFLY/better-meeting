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


def test_merge_does_not_repeat_a_line_another_pass_already_covered():
    # "A again" is the same sentence from another pass. It adds 0.5 s past "A"
    # and outscores "B", the next real line.
    passes = [
        [seg(3, 9.5, "A", "uk", -0.2)],
        [seg(3, 10, "A again", "ru", -0.25)],
        [seg(10, 16, "B", "en", -0.3)],
    ]
    assert [s["text"] for s in merge_passes(passes)] == ["A", "B"]


def test_merge_does_not_repeat_a_line_that_ends_slightly_later_before_a_pause():
    # As above, "A again" adds 0.5 s past "A", but nothing follows within 2 s.
    passes = [
        [seg(3, 9.5, "A", "uk", -0.2)],
        [seg(3, 10, "A again", "ru", -0.25)],
        [seg(20, 26, "B", "en", -0.3)],
    ]
    assert [s["text"] for s in merge_passes(passes)] == ["A", "B"]


def test_merge_takes_a_segment_that_mostly_continues_past_the_cursor():
    passes = [[seg(0, 5, "X", "uk", -0.3)], [seg(4, 12, "Y", "en", -0.2)]]
    assert [s["text"] for s in merge_passes(passes)] == ["X", "Y"]


def test_merge_keeps_a_short_line_before_a_more_confident_one():
    # The long line outscores "Так." in both passes and starts within 2 s of the cursor.
    uk = [seg(0, 1.5, "Так.", "uk", -0.6), seg(1.5, 6, "uk-long", "uk", -0.2)]
    en = [seg(0, 1.5, "Tuck.", "en", -0.9), seg(1.5, 6, "en-long", "en", -0.8)]
    assert [s["text"] for s in merge_passes([uk, en])] == ["Так.", "uk-long"]


def test_merge_keeps_a_line_of_200_ms():
    passes = [[seg(0, 2, "a", "en"), seg(10, 10.2, "b", "en")]]
    assert [s["text"] for s in merge_passes(passes)] == ["a", "b"]


def test_merge_keeps_speech_only_a_crossing_segment_covers():
    # Nothing but "A" covers 6-10 s.
    passes = [[seg(0, 10, "A", "uk", -0.3)], [seg(0, 6, "B", "en", -0.2)]]
    assert [s["text"] for s in merge_passes(passes)] == ["B", "A"]


def test_merge_keeps_a_crossing_segment_when_the_winner_would_skip_its_stretch():
    # "Мм." takes 0-2 s from the English sentence. The next English line starts at
    # 3.5 s, so only the sentence covers 2-3.5 s. The two English lines have the
    # same score, as lines of one Whisper window do, and the earlier one wins.
    uk = [seg(0, 2, "Мм.", "uk", -0.8)]
    en = [seg(0, 3.5, "sentence", "en", -0.9), seg(3.5, 7, "next", "en", -0.9)]
    assert [s["text"] for s in merge_passes([uk, en])] == ["Мм.", "sentence", "next"]


def test_merge_does_not_repeat_a_stretch_for_a_more_confident_crossing_segment():
    # "en 6-10" starts at the cursor, so 6-10 s has an offer. "uk 0-10" is more
    # confident, but taking it prints 0-6 s a second time.
    uk = [seg(0, 10, "uk 0-10", "uk", -0.35), seg(10, 15, "uk next", "uk", -0.2)]
    en = [
        seg(0, 6, "en 0-6", "en", -0.15),
        seg(6, 10, "en 6-10", "en", -0.9),
        seg(10, 15, "en next", "en", -0.9),
    ]
    assert [s["text"] for s in merge_passes([uk, en])] == ["en 0-6", "en 6-10", "uk next"]


def test_merge_keeps_a_line_that_lost_to_shorter_lines():
    # The cursor passes 0-2 s and 2.3-3 s of the sentence, but no line covers them.
    uk = [seg(2, 2.3, "Мм.", "uk", -0.8), seg(3, 3.3, "Ага.", "uk", -0.8)]
    en = [seg(0, 4, "sentence", "en", -0.9)]
    assert [s["text"] for s in merge_passes([uk, en])] == ["sentence", "Мм.", "Ага."]


def test_merge_returns_segments_in_start_order():
    # The sentence is taken last, after both short lines, but it starts first.
    uk = [seg(2, 2.3, "Мм.", "uk", -0.8), seg(3, 3.3, "Ага.", "uk", -0.8)]
    en = [seg(0, 4, "sentence", "en", -0.9), seg(4, 8, "next", "en", -0.9)]
    starts = [s["start"] for s in merge_passes([uk, en])]
    assert starts == sorted(starts)
    assert starts == [0, 2, 3, 4]


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

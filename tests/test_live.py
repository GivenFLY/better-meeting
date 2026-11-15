"""Live sessions: pause segmenter, session loop, recording and the ``bm live`` command."""

from __future__ import annotations

import io
import signal
import subprocess
import wave
from pathlib import Path
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from better_meeting import asr, live  # noqa: E402
from better_meeting.asr.base import AsrBackend, RawSegment  # noqa: E402
from better_meeting.cli import main  # noqa: E402
from better_meeting.live.capture import Capture, Window  # noqa: E402
from better_meeting.live.echo import EchoFilter  # noqa: E402
from better_meeting.live.recording import crop, shrink  # noqa: E402
from better_meeting.live.segmenter import RATE, Segmenter  # noqa: E402
from better_meeting.live.session import LiveOptions, _stop_on_signals, leveled, run_session  # noqa: E402
from better_meeting.render import render_timeline, render_transcript  # noqa: E402
from better_meeting.utils import BmError, load  # noqa: E402


def tone(seconds: float) -> Any:
    t = np.arange(round(seconds * RATE)) / RATE
    return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def quiet(seconds: float) -> Any:
    return np.zeros(round(seconds * RATE), dtype=np.float32)


def feed(seg: Segmenter, audio: Any, block: int = 1600) -> list[Any]:
    out = []
    for i in range(0, len(audio), block):
        out += seg.feed(audio[i : i + block])
    return out


def test_segmenter_cuts_at_pauses():
    audio = np.concatenate([quiet(1), tone(2), quiet(1), tone(3), quiet(1)])
    utts = feed(Segmenter(), audio)
    assert len(utts) == 2
    assert utts[0].start == pytest.approx(0.8, abs=0.07)  # 0.2 s of padding before the speech
    assert utts[0].end == pytest.approx(3.2, abs=0.07)
    assert utts[1].start == pytest.approx(3.8, abs=0.07)
    assert utts[1].end == pytest.approx(7.2, abs=0.07)
    assert len(utts[0].samples) == round((utts[0].end - utts[0].start) * RATE)


def test_segmenter_short_pause_does_not_cut():
    utts = feed(Segmenter(silence=0.6), np.concatenate([tone(2), quiet(0.3), tone(2), quiet(1)]))
    assert len(utts) == 1


def test_segmenter_drops_a_click():
    assert feed(Segmenter(), np.concatenate([quiet(1), tone(0.1), quiet(1)])) == []


def test_segmenter_cuts_long_speech_at_its_quietest_frame():
    dip = tone(0.09) * 0.1  # still speech, but the quietest stretch
    seg = Segmenter(max_len=5.0)
    utts = feed(seg, np.concatenate([tone(3), dip, tone(4)]))
    assert len(utts) == 1
    assert utts[0].end == pytest.approx(3.05, abs=0.07)
    rest = seg.flush()
    assert rest[0].start == pytest.approx(utts[0].end)  # nothing is lost at the cut
    assert rest[0].end == pytest.approx(7.09, abs=0.07)


def test_segmenter_open_and_flush():
    seg = Segmenter()
    feed(seg, np.concatenate([quiet(0.5), tone(1)]))
    assert seg.open().end == pytest.approx(1.5, abs=0.07)
    assert len(seg.flush()) == 1
    assert seg.open() is None


def test_segmenter_gap_in_the_source_closes_the_utterance():
    seg = Segmenter()
    assert seg.feed(tone(1), 0.0) == []
    utts = seg.feed(tone(1), 10.0)  # the source skipped 9 s
    assert [round(u.end) for u in utts] == [1]
    assert seg.flush()[0].start == pytest.approx(10.0, abs=0.07)


def test_utterance_cut_and_speech_seconds():
    (utt,) = feed(Segmenter(), np.concatenate([tone(2), quiet(1)]))
    assert utt.speech_seconds(utt.start, utt.end) == pytest.approx(2.0, abs=0.07)
    part = utt.cut(0.5, 1.0)
    assert (part.start, part.end) == (pytest.approx(0.51, abs=0.02), pytest.approx(0.99, abs=0.02))
    assert len(part.samples) == round((part.end - part.start) * RATE)


class EchoAsr(AsrBackend):
    """One segment per decoded clip: its text is the clip's length in seconds."""

    name = "echo"
    default_model = "echo-model"
    single_pass = True
    #: Segments end this early, as if the provider skipped the rest of the clip.
    covers: float | None = None

    def __init__(self) -> None:
        self.clips: list[float] = []

    def transcribe(
        self, wav: Path, lang: str | None, model: str, opts: dict[str, Any]
    ) -> list[RawSegment]:
        with wave.open(str(wav)) as w:
            seconds = w.getnframes() / w.getframerate()
            silent = not np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").any()
        self.clips.append(seconds)
        if silent:
            return []  # the warm-up clip
        end = min(seconds, self.covers) if self.covers else seconds
        return [
            {
                "start": 0.0,
                "end": end,
                "text": f"{seconds:.1f}s",
                "lang": lang or "en",
                "score": 0.9,
                "nospeech": 0.0,
            }
        ]


class FakeCapture(Capture):
    """Delivers prepared blocks from ``start`` and then ends, like a closed window."""

    name = "fakecap"
    audio: dict[str, Any] = {}  # noqa: RUF012 (set per test)
    has_video = False

    def __init__(self) -> None:
        super().__init__()
        self.stopped = False

    def windows(self) -> list[Window]:
        return [Window(1, "Meet", "Weekly sync"), Window(2, "Editor", "notes.md")]

    def start(self, window, tracks, video, on_audio) -> None:
        self.window, self.tracks = window, tracks
        for track in tracks:
            audio = self.audio[track]
            for i in range(0, len(audio), 1600):
                on_audio(track, i / RATE, audio[i : i + 1600])
        self.finished.set()

    def stop(self) -> None:
        self.stopped = True


@pytest.fixture
def echo() -> EchoAsr:
    backend = EchoAsr()
    asr.register_backend("echo", lambda: backend)
    return backend


@pytest.fixture
def fake_capture() -> type[FakeCapture]:
    live.register_capture("fakecap", FakeCapture)
    FakeCapture.audio = {
        "app": np.concatenate([quiet(0.5), tone(2), quiet(3)]),
        "mic": np.concatenate([quiet(3.5), tone(1), quiet(1)]),
    }
    return FakeCapture


def session(tmp_path, echo, tracks, **options):
    out = io.StringIO()
    capture = FakeCapture()
    transcript = run_session(
        capture,
        None,
        tracks,
        tmp_path,
        echo,
        "echo-model",
        LiveOptions(**options),
        out=out,
        err=io.StringIO(),
    )
    return capture, transcript, out.getvalue().splitlines()


def test_session_transcribes_each_track(tmp_path, echo, fake_capture):
    capture, transcript, lines = session(tmp_path, echo, ("mic", "app"), draft=False)
    assert capture.stopped
    assert capture.tracks == ("app", "mic")  # always in the order of the recording's streams
    assert [(s["track"], round(s["start"], 1)) for s in transcript] == [("app", 0.3), ("mic", 3.3)]
    assert lines == ["[00:00] [app] [en] 2.5s", "[00:03] [mic] [en] 1.4s"]
    for track, seconds in (("app", 5.5), ("mic", 5.5)):
        with wave.open(str(tmp_path / f"track_{track}.wav")) as w:
            assert (w.getframerate(), w.getnchannels()) == (RATE, 1)
            assert w.getnframes() / RATE == pytest.approx(seconds)
    assert not (tmp_path / "live_utterance.wav").exists()


def test_session_single_track_has_no_track_tag(tmp_path, echo, fake_capture):
    _, transcript, lines = session(tmp_path, echo, ("app",), lang="uk")
    assert lines == ["[00:00] [uk] 2.5s"]
    assert transcript[0]["track"] == "app"


def test_session_decodes_uncovered_speech_again(tmp_path, echo, fake_capture):
    fake_capture.audio = {"app": np.concatenate([tone(6), quiet(1)])}
    echo.covers = 2.0  # the provider "skips" the last 4 s of speech
    _, transcript, _ = session(tmp_path, echo, ("app",), draft=False)
    assert [round(s["start"], 1) for s in transcript] == [0.0, 1.8]
    assert echo.clips[1:] == [pytest.approx(6.2, abs=0.07), pytest.approx(4.4, abs=0.07)]


def test_render_tags_tracks_only_when_there_are_two():
    one = [{"start": 0.0, "end": 1.0, "text": "hi", "lang": "en", "track": "app"}]
    assert "[app]" not in render_transcript(one)
    two = [*one, {"start": 2.0, "end": 3.0, "text": "yo", "lang": "en", "track": "mic"}]
    assert "[00:00] [app] hi" in render_transcript(two)
    assert "[00:02] SPEECH: [mic] yo" in render_timeline(two, [], {}, 8.0)
    assert "[mic] is the microphone" in render_timeline(two, [], {}, 8.0)


def test_live_list_windows(fake_capture, capsys):
    assert main(["live", "--capture", "fakecap", "--list-windows"]) == 0
    assert capsys.readouterr().out == "Meet\tWeekly sync\nEditor\tnotes.md\n"


def test_live_needs_a_choice_without_a_terminal(fake_capture, echo, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    args = ["live", "--capture", "fakecap", "--asr-backend", "echo", "--out-root", str(tmp_path)]
    assert main(args) == 1
    assert "pass the choice as an option" in capsys.readouterr().err


def test_live_window_must_match_one(fake_capture, echo, tmp_path, capsys):
    args = ["live", "--capture", "fakecap", "--asr-backend", "echo", "--out-root", str(tmp_path)]
    assert main([*args, "--source", "app", "--window", "e"]) == 1
    assert "matches 2 windows" in capsys.readouterr().err
    assert main([*args, "--source", "app", "--window", "zoom"]) == 1
    assert "no window to capture matches 'zoom'" in capsys.readouterr().err


@pytest.mark.ffmpeg
def test_live_end_to_end_without_video(fake_capture, echo, ffmpeg, tmp_path, capsys):
    args = ["live", "--capture", "fakecap", "--asr-backend", "echo", "--out-root", str(tmp_path)]
    assert main([*args, "--source", "app+mic", "--window", "meet", "--name", "sync"]) == 0
    assert capsys.readouterr().out.splitlines() == [
        "[00:00] [app] [en] 2.5s",
        "[00:03] [mic] [en] 1.4s",
    ]
    work = tmp_path / "sync"
    assert (tmp_path / "sync.m4a").exists()  # the fake capture writes no video
    assert not list(work.glob("track_*.wav"))
    assert [s["track"] for s in load(work / "transcript.json")] == ["app", "mic"]
    assert "[00:03] [mic] 1.4s" in (work / "artifacts" / "transcript.md").read_text()
    assert "SPEECH: [app] 2.5s" in (work / "artifacts" / "timeline.md").read_text()
    # The stored transcript is the transcription stage's cache: extract does not run ASR.
    clips = len(echo.clips)
    extract = ["extract", str(tmp_path / "sync.m4a"), "--out", str(work), "--only", "transcribe"]
    assert main([*extract, "--asr-backend", "echo"]) == 0
    assert len(echo.clips) == clips
    # One folder per recording.
    assert main([*args, "--source", "mic", "--name", "sync"]) == 1
    assert "already holds a recording" in capsys.readouterr().err


def test_get_capture_unknown():
    with pytest.raises(BmError, match="unknown capture provider"):
        live.get_capture("nope")


def echo_session(app: Any, mic: Any, block: int = 800) -> EchoFilter:
    """Both tracks fed in the order a capture delivers them."""
    echo_filter = EchoFilter()
    for i in range(0, len(app), block):
        echo_filter.played(i / RATE, app[i : i + block])
        echo_filter.heard(i / RATE, mic[i : i + block])
    return echo_filter


def mic_utterances(mic: Any) -> list[Any]:
    seg = Segmenter()
    return feed(seg, mic) + seg.flush()


def test_echo_filter_drops_the_application_heard_through_speakers():
    app = np.concatenate([quiet(1), tone(6), quiet(2)])
    # The speakers reach the microphone 0.2 s later at a tenth of the level.
    mic = 0.1 * np.concatenate([quiet(0.2), app])[: len(app)]
    echo_filter = echo_session(app, mic)
    (utt,) = mic_utterances(mic)
    assert echo_filter.user_part(utt) is None


def test_echo_filter_keeps_the_user_talking_over_the_application():
    app = np.concatenate([quiet(1), tone(8), quiet(2)])
    mic = 0.1 * np.concatenate([quiet(0.2), app])[: len(app)]
    mic[5 * RATE : 6 * RATE] += tone(1)  # the user, for one second
    echo_filter = echo_session(app, mic)
    (utt,) = mic_utterances(mic)
    mine = echo_filter.user_part(utt)
    assert mine is not None
    assert mine.start == pytest.approx(4.7, abs=0.1)
    assert mine.end == pytest.approx(6.3, abs=0.1)


def test_echo_filter_leaves_speech_alone_while_the_application_is_silent():
    app = np.concatenate([quiet(1), tone(4), quiet(5)])
    mic = 0.1 * np.concatenate([quiet(0.2), app])[: len(app)]
    mic[7 * RATE : 8 * RATE] += 0.05 * tone(1)  # quieter than the echo was, but nothing plays
    echo_filter = echo_session(app, mic)
    echo_only, mine = mic_utterances(mic)
    assert echo_filter.user_part(echo_only) is None
    assert echo_filter.user_part(mine) is mine


def test_session_drops_the_echo_on_the_microphone(tmp_path, echo, fake_capture):
    app = np.concatenate([quiet(0.5), tone(6), quiet(2)])
    FakeCapture.audio = {"app": app, "mic": 0.1 * np.concatenate([quiet(0.2), app])[: len(app)]}
    _, transcript, _ = session(tmp_path, echo, ("app", "mic"), draft=False)
    assert [s["track"] for s in transcript] == ["app"]


def test_leveled_amplifies_quiet_audio_only():
    assert np.sqrt(np.mean(leveled(0.05 * tone(1)) ** 2)) == pytest.approx(0.1, rel=0.05)
    assert np.abs(leveled(0.0001 * tone(1))).max() == pytest.approx(0.0001 * 0.3 * 20, rel=0.01)
    loud = tone(1)
    assert leveled(loud) is loud


def test_crop_shows_less_of_the_frame_without_encoding_again(ffmpeg, tmp_path):
    video = tmp_path / "video.mp4"
    source = ["-f", "lavfi", "-i", "testsrc=size=320x200:rate=5:duration=1"]
    encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p", str(video)]
    subprocess.run(["ffmpeg", "-loglevel", "error", *source, *encode], check=True)
    size = ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0"]

    crop(video, (320, 200))  # the whole frame: nothing to do
    assert subprocess.check_output([*size, str(video)], text=True).strip() == "320,200"
    crop(video, (120, 90))
    assert subprocess.check_output([*size, str(video)], text=True).strip() == "120,90"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["video.mp4"]


def test_second_interrupt_does_not_stop_the_finishing():
    restore = _stop_on_signals()
    try:
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)
        signal.raise_signal(signal.SIGINT)  # answered with a note, nothing raised
        signal.raise_signal(signal.SIGTERM)
    finally:
        restore()


def test_shrink_scales_the_video_and_keeps_the_audio(ffmpeg, tmp_path):
    recording = tmp_path / "meeting.mp4"
    source = ["-f", "lavfi", "-i", "testsrc=size=320x200:rate=5:duration=2"]
    source += ["-f", "lavfi", "-i", "sine=duration=2"]
    first = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(recording)]
    subprocess.run(["ffmpeg", "-loglevel", "error", *source, *first], check=True)
    probe = ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,nb_frames"]
    probe += ["-of", "csv=p=0", str(recording)]
    audio = subprocess.check_output(probe, text=True).splitlines()[1]

    shrink(recording, None, 1.0)  # nothing asked: the file is left alone
    assert subprocess.check_output(probe, text=True).splitlines()[0] == "video,320,10"
    shrink(recording, "100k", 0.5)
    assert subprocess.check_output(probe, text=True).splitlines() == ["video,160,10", audio]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["meeting.mp4"]


@pytest.mark.parametrize("flags", [["--fps", "0"], ["--scale", "1.5"], ["--video-bitrate", "fast"]])
def test_live_rejects_bad_video_options(fake_capture, echo, tmp_path, capsys, flags):
    argv = ["live", "--capture", "fakecap", "--asr-backend", "echo", "--source", "mic"]
    assert main([*argv, "--out-root", str(tmp_path), *flags]) == 1
    assert flags[0] in capsys.readouterr().err

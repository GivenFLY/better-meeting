"""What the capture providers share: resampling, the track clock, the video writer."""

from __future__ import annotations

import itertools
import subprocess
import sys
import time
from typing import Any

import pytest

np = pytest.importorskip("numpy")

from better_meeting import live  # noqa: E402
from better_meeting.live.capture import registry  # noqa: E402
from better_meeting.live.capture.audio import Resampler, TrackFeed, mono  # noqa: E402
from better_meeting.live.capture.video import VideoPipe  # noqa: E402
from better_meeting.live.segmenter import RATE  # noqa: E402
from better_meeting.utils import BmError  # noqa: E402


def sine(hz: float, seconds: float, rate: int) -> Any:
    return np.sin(2 * np.pi * hz * np.arange(round(seconds * rate)) / rate).astype(np.float32)


def test_resampler_keeps_speech_and_drops_what_would_fold_back():
    def level(hz: float) -> float:
        resample = Resampler(48000)
        blocks = np.split(sine(hz, 1, 48000), 100)  # block borders must not be heard
        out = np.concatenate([resample(b) for b in blocks])
        assert len(out) == RATE
        return float(np.sqrt(np.mean(out[1000:] ** 2)))

    assert level(1000) == pytest.approx(0.707, rel=0.02)
    assert level(15000) < 0.01  # would land at 1 kHz without the filter


def test_resampler_passes_the_session_rate_through():
    block = sine(440, 0.1, RATE)
    assert Resampler(RATE)(block) is block
    assert len(Resampler(44100)(sine(440, 1, 44100))) == RATE


def test_mono_reads_interleaved_pcm():
    stereo = np.array([[0.5, -0.5], [1.0, 0.0]], dtype="<f4")
    assert mono(stereo.tobytes(), "<f4", 2).tolist() == [0.0, 0.5]
    pcm = np.array([16384, -16384, 0], dtype="<i2")
    assert mono(pcm.tobytes(), "<i2", 1).tolist() == [0.5, -0.5, 0.0]


def test_track_feed_places_blocks_on_the_session_clock():
    got: list[tuple[str, float, int]] = []
    feed = TrackFeed("app", 48000, lambda track, t, s: got.append((track, t, len(s))), t0=100.0)
    block = sine(440, 0.1, 48000)

    feed.push(block, now=100.5)  # ended at 0.5, so it started at 0.4
    feed.push(block, now=100.63)  # a little late: still follows the first one
    feed.push(block, now=103.0)  # the source was quiet in between
    assert [(track, n) for track, _, n in got] == [("app", 1600)] * 3
    assert [t for _, t, _ in got] == pytest.approx([0.4, 0.5, 2.9])


def test_track_feed_never_goes_before_the_start_of_the_session():
    got: list[float] = []
    feed = TrackFeed("mic", RATE, lambda track, t, s: got.append(t), t0=10.0)
    feed.push(sine(440, 0.5, RATE), now=10.2)
    assert got == [0.0]


def probe(video: Any, entries: str) -> list[str]:
    cmd = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries]
    return subprocess.check_output([*cmd, "-of", "csv=p=0", str(video)], text=True).split()


def test_video_pipe_writes_changes_and_keeps_the_clock(ffmpeg, tmp_path):
    video = tmp_path / "video.mp4"
    pipe = VideoPipe(video, (320, 200), fps=10)
    red = np.zeros((120, 200, 4), dtype=np.uint8)
    red[..., 2] = 255
    assert pipe.show(red) == (200, 120)
    assert pipe.started.wait(5)
    for shade in range(30):  # far more changes than frames
        red[..., 1] = shade
        pipe.show(red)
        time.sleep(0.01)
    time.sleep(1.3)  # nothing changes: one frame a second keeps the video going
    assert pipe.show(np.full((400, 400, 4), 255, dtype=np.uint8)) == (320, 200)  # cut to the frame
    assert pipe.show(np.full((50, 60, 4), 255, dtype=np.uint8)) == (60, 50)
    pipe.close()
    length = time.monotonic() - pipe.t0

    assert probe(video, "stream=codec_name,width,height") == ["h264,320,200"]
    times = [float(t.strip(",")) for t in probe(video, "frame=pts_time")]
    assert times[0] == 0.0 and 4 <= len(times) <= 10
    assert times[-1] == pytest.approx(length, abs=0.5)
    assert min(b - a for a, b in itertools.pairwise(times)) > 0.03

    # The last picture is smaller than the one before it: black around it.
    shot = tmp_path / "frames.rgb"
    raw = ["-i", str(video), "-fps_mode", "passthrough", "-f", "rawvideo", "-pix_fmt", "rgb24"]
    subprocess.run(["ffmpeg", "-loglevel", "error", *raw, str(shot)], check=True)
    frame = np.frombuffer(shot.read_bytes(), dtype=np.uint8).reshape(-1, 200, 320, 3)[-1]
    assert frame[20, 20].min() > 200 and frame[100, 150].max() < 40


@pytest.mark.parametrize(
    ("system", "name"), [("Darwin", "screencapturekit"), ("Windows", "windows")]
)
def test_default_capture_follows_the_platform(monkeypatch, system, name):
    monkeypatch.setattr(registry.platform, "system", lambda: system)
    assert registry.default_capture_name() == name


def test_no_default_capture_on_other_platforms(monkeypatch):
    monkeypatch.setattr(registry.platform, "system", lambda: "Linux")
    with pytest.raises(BmError, match="macOS 15\\+ or Windows 10"):
        live.get_capture("auto")


@pytest.mark.skipif(sys.platform == "win32", reason="the provider loads there")
def test_windows_capture_says_what_it_needs_elsewhere():
    with pytest.raises(BmError, match="on Windows"):
        live.get_capture("windows")

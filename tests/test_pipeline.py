"""End-to-end `bm extract` on a generated clip, with fake ASR/OCR providers."""

import json
import subprocess

import pytest

from better_meeting.cli import main

pytestmark = pytest.mark.ffmpeg


@pytest.fixture
def clip(tmp_path, ffmpeg):
    """8 s test pattern that changes every 2 s, with a tone."""
    video = tmp_path / "demo.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "testsrc2=size=320x240:rate=10:duration=8",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:duration=8",
            "-vf",
            "hue=h=t*45",
            "-shortest",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    return video


def run_extract(video, out, *extra):
    return main(
        [
            "-q",
            "extract",
            str(video),
            "--out",
            str(out),
            "--asr-backend",
            "fake",
            "--langs",
            "uk,en",
            "--ocr",
            "fakeocr",
            "--frame-interval",
            "1",
            *extra,
        ]
    )


def test_extract_end_to_end_and_cache(tmp_path, clip, fake_asr, fake_ocr):
    fake_ocr.screens = [f"Slide {i}\nitem {i}" for i in range(50)]
    out = tmp_path / "run"
    assert run_extract(clip, out) == 0

    art = out / "artifacts"
    timeline = (art / "timeline.md").read_text(encoding="utf-8")
    assert timeline.startswith("# Timeline")
    assert "SPEECH: [uk] uk 0" in timeline
    assert "SCREEN (screenshot: screens/" in timeline
    assert list((art / "screens").glob("*.jpg"))
    assert json.loads((out / "transcript.meta.json").read_text())["backend"] == "fake"

    asr_calls, ocr_calls = len(fake_asr.calls), len(fake_ocr.seen)
    assert run_extract(clip, out) == 0  # everything cached
    assert (len(fake_asr.calls), len(fake_ocr.seen)) == (asr_calls, ocr_calls)

    assert run_extract(clip, out, "--ocr-sim", "0.5") == 0  # OCR params changed
    assert len(fake_asr.calls) == asr_calls
    assert len(fake_ocr.seen) > ocr_calls


def test_changed_frame_interval_rebuilds_frames(tmp_path, clip, fake_asr, fake_ocr):
    out = tmp_path / "run"
    assert run_extract(clip, out, "--ocr", "none") == 0
    first = sorted(p.name for p in (out / "thumbs").iterdir())
    assert (
        main(
            [
                "-q",
                "extract",
                str(clip),
                "--out",
                str(out),
                "--asr-backend",
                "fake",
                "--langs",
                "uk,en",
                "--ocr",
                "none",
                "--frame-interval",
                "4",
            ]
        )
        == 0
    )
    second = sorted(p.name for p in (out / "thumbs").iterdir())
    assert len(second) < len(first)  # no stale thumbnails left

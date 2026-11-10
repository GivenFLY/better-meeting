"""Thumbnail sampling and frame grabbing on generated clips."""

import subprocess

import pytest
from PIL import Image

from better_meeting.frames import extract_full_frames, grab_frame, pick_keyframes, sample_thumbs

pytestmark = pytest.mark.ffmpeg


def make_clip(path, luma, *options):
    """8 s at 10 fps. ``luma`` is an ffmpeg expression of the time ``T``: 16 is black, 235 white."""
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
            f"color=c=black:s=64x64:r=10:d=8,geq=lum='{luma}':cb=128:cr=128",
            *options,
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )


def is_white(path):
    with Image.open(path) as im:
        return im.convert("L").resize((1, 1)).getpixel((0, 0)) > 128


def test_keyframe_frame_shows_the_screen_its_thumbnail_showed(tmp_path, ffmpeg):
    """The screen turns white at 2.5 s, between two thumbnail times."""
    video = tmp_path / "change.mp4"
    make_clip(video, "if(gte(T,2.5),235,16)")
    thumbs = sample_thumbs(video, tmp_path / "thumbs", 2.0)
    keyframes = pick_keyframes(thumbs, cell_delta=14, min_cells=3, max_gap=90.0)
    frames = extract_full_frames(video, keyframes, tmp_path / "frames", 320)

    thumb_at = {th["t"]: th["path"] for th in thumbs}
    assert [is_white(thumb_at[f["t"]]) for f in frames] == [False, True]  # change detected
    assert [is_white(f["path"]) for f in frames] == [False, True]  # and the new screen grabbed


def test_grab_frame_returns_the_frame_on_screen_during_a_frame_gap(tmp_path, ffmpeg):
    """No frames from 1 s to 5 s, as when a recorder skips a still screen. White starts at 5 s."""
    video = tmp_path / "gap.mkv"  # Matroska keeps the gap without a sync option
    make_clip(video, "if(gte(T,5),235,16)", "-vf", "select='lt(t,1.05)+gt(t,4.95)'")
    shot = tmp_path / "shot.jpg"
    grab_frame(video, 3.0, shot, 320)
    assert not is_white(shot)

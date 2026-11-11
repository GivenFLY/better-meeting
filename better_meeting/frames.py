"""Stage 3: cheap thumbnails -> keyframes by grid diff -> full-size frames."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .utils import log, need, run

GRID = 16  # thumbnails are compared as GRID x GRID grayscale cells
SPIKE_DELTA = 60  # one cell changing this much keeps the frame on its own


def sample_thumbs(video: Path, out_dir: Path, interval: float) -> list[dict[str, Any]]:
    """One 320 px thumbnail every ``interval`` seconds -> ``[{t, path}]``.

    The thumbnail at ``t`` is the frame that was on screen at ``t``, the same
    frame :func:`grab_frame` returns. That needs ``round=up``: with its default
    rounding the ``fps`` filter takes a frame from up to half an interval after
    ``t``. ``start_time=0`` keeps the first thumbnail at 0 when the video starts
    later than the audio.

    ``out_dir`` is emptied first: leftovers from a run with another interval
    would otherwise be picked up with wrong timestamps.
    """
    need("ffmpeg")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    log(f"sampling a thumbnail every {interval:g}s")
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video),
            "-vf",
            f"fps=1/{interval}:round=up:start_time=0,scale=320:-2",
            "-q:v",
            "6",
            str(out_dir / "%06d.jpg"),
        ]
    )
    files = sorted(out_dir.glob("*.jpg"))
    return [{"t": i * interval, "path": str(p)} for i, p in enumerate(files)]


def grid_signature(path: str, n: int = GRID) -> list[int]:
    from PIL import Image

    with Image.open(path) as im:
        return list(im.convert("L").resize((n, n), Image.BILINEAR).tobytes())


def is_change(sig: list[int], prev: list[int], cell_delta: int, min_cells: int) -> bool:
    """True if at least ``min_cells`` cells moved by more than ``cell_delta``, or one
    cell moved by more than ``SPIKE_DELTA``. Local changes (typing in a terminal)
    count, so a mostly static screen still yields keyframes."""
    diffs = [abs(a - b) for a, b in zip(sig, prev, strict=True)]
    strong = sum(1 for d in diffs if d > cell_delta)
    return strong >= min_cells or max(diffs) > SPIKE_DELTA


def pick_keyframes(
    thumbs: list[dict[str, Any]], cell_delta: int, min_cells: int, max_gap: float
) -> list[dict[str, Any]]:
    """Thumbnails where the screen changed, plus one at least every ``max_gap`` seconds."""
    log("looking for frames where the screen changed")
    kept: list[dict[str, Any]] = []
    prev_sig: list[int] | None = None
    prev_t = 0.0
    for th in thumbs:
        sig = grid_signature(th["path"])
        keep = (
            prev_sig is None
            or is_change(sig, prev_sig, cell_delta, min_cells)
            or th["t"] - prev_t >= max_gap
        )
        if keep:
            kept.append({"t": th["t"]})
            prev_sig, prev_t = sig, th["t"]
    log(f"{len(kept)} keyframes out of {len(thumbs)} thumbnails")
    return kept


def grab_frame(video: Path, t: float, path: Path, width: int) -> None:
    """The frame that was on screen at ``t`` seconds, scaled to ``width``.

    ``-ss`` alone gives the first frame at or after ``t``. In a recording that
    has no frames while the screen is still, that is the next change. So decoding
    starts at the keyframe before ``t`` (``-noaccurate_seek``, which also makes
    timestamps relative to ``t``), the ``fps`` filter repeats each frame until
    the next one, and ``select`` takes the frame that is current at 0 and drops
    the rest. If the video starts after ``t``, that is its first frame: 100 ticks
    per second is finer than the frame interval.
    """
    need("ffmpeg")
    run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-ss",
            f"{t:.3f}",
            "-noaccurate_seek",
            "-i",
            str(video),
            "-frames:v",
            "1",
            "-vf",
            f"fps=100:round=up,select='gte(t,0)*eq(selected_n,0)',scale={width}:-2",
            "-q:v",
            "2",
            str(path),
        ]
    )


def extract_full_frames(
    video: Path, keyframes: list[dict[str, Any]], out_dir: Path, width: int
) -> list[dict[str, Any]]:
    """Full-size frame per keyframe -> ``[{t, path}]``; existing files are reused."""
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = 0
    out = []
    for i, kf in enumerate(keyframes):
        path = out_dir / f"{i:05d}_{int(kf['t']):06d}.jpg"
        if not path.exists():
            grab_frame(video, kf["t"], path, width)
            missing += 1
        out.append({"t": kf["t"], "path": str(path)})
    log(f"{len(keyframes)} full frames ({missing} extracted, {len(keyframes) - missing} cached)")
    return out

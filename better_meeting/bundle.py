"""Stage 5: assemble the self-contained ``artifacts/`` folder."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .ocr import norm
from .render import render_screens_index, render_stats, render_timeline, render_transcript
from .shots import fallback_shots, select_shots, write_shot
from .utils import log, ts, ts_file


def write_bundle(
    art: Path,
    transcript: list[dict[str, Any]],
    screen: list[dict[str, Any]],
    frames: list[dict[str, Any]],
    duration: float,
    *,
    max_shots: int,
    shot_width: int,
    shot_quality: int,
    draw_label: bool,
    pause: float,
) -> None:
    """Rebuild ``art`` from scratch: transcript.md, timeline.md, screens/, screens_index.md."""
    if art.exists():
        shutil.rmtree(art)
    (art / "screens").mkdir(parents=True)

    shots = select_shots(screen, duration, max_shots)
    if not shots:
        shots = fallback_shots(frames, max_shots)
        if shots:
            log("no OCR text: picking frames evenly over time instead")
    log(f"selected {len(shots)} screenshots")

    shots_by_t: dict[float, str] = {}
    index_rows: list[tuple[str, str, str]] = []
    for i, ev in enumerate(shots, 1):
        rel = f"screens/{i:03d}_{ts_file(ev['t'])}.jpg"
        write_shot(ev["path"], art / rel, ts(ev["t"]), shot_width, shot_quality, draw_label)
        shots_by_t[round(ev["t"], 3)] = rel
        preview = " / ".join(norm(line) for line in ev["added"][:3])[:160]
        index_rows.append((ts(ev["t"]), rel, preview))

    stats = render_stats(transcript, screen, duration, len(shots))
    (art / "transcript.md").write_text(render_transcript(transcript), encoding="utf-8")
    (art / "timeline.md").write_text(
        render_timeline(transcript, screen, shots_by_t, pause, stats), encoding="utf-8"
    )
    (art / "screens_index.md").write_text(render_screens_index(index_rows), encoding="utf-8")
    log(f"done: {art}")

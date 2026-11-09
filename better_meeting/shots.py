"""Screenshots: pick the most informative frames within a budget, stamp the timecode."""

from __future__ import annotations

from pathlib import Path
from typing import Any

FONT_CANDIDATES = [
    "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
    "/System/Library/Fonts/Helvetica.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _text_amount(ev: dict[str, Any]) -> int:
    return sum(len(line) for line in ev["added"])


def select_shots(
    screen: list[dict[str, Any]], duration: float, max_shots: int
) -> list[dict[str, Any]]:
    """Pick at most ``max_shots`` screen events that added text.

    The recording is split into ``max_shots`` equal time buckets and the event
    with the most new text wins each bucket, so shots are spread over the whole
    meeting; leftover slots go to the most text-heavy remaining events.
    """
    candidates = [e for e in screen if e["added"]]
    if not candidates or max_shots <= 0 or len(candidates) <= max_shots:
        return candidates

    duration = duration or (candidates[-1]["t"] + 1)
    buckets: dict[int, tuple[int, dict[str, Any]]] = {}
    for ev in candidates:
        idx = min(int(ev["t"] / duration * max_shots), max_shots - 1)
        score = _text_amount(ev)
        if idx not in buckets or score > buckets[idx][0]:
            buckets[idx] = (score, ev)
    chosen = [v[1] for v in buckets.values()]

    if len(chosen) < max_shots:
        rest = sorted((e for e in candidates if e not in chosen), key=_text_amount, reverse=True)
        chosen += rest[: max_shots - len(chosen)]
    return sorted(chosen, key=lambda e: e["t"])


def fallback_shots(frames: list[dict[str, Any]], max_shots: int) -> list[dict[str, Any]]:
    """Evenly spaced keyframes, for when OCR is off or found no text (e.g. low-res
    recordings): the screen content still reaches the folder, and reading it is
    left to a multimodal model."""
    if max_shots <= 0 or not frames:
        return []
    if len(frames) <= max_shots:
        picked = frames
    else:
        step = len(frames) / max_shots
        picked = [frames[int(i * step)] for i in range(max_shots)]
    return [{"t": f["t"], "path": f["path"], "added": []} for f in picked]


def _font(size: int) -> Any:
    from PIL import ImageFont

    for p in FONT_CANDIDATES:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except OSError:
                continue
    return ImageFont.load_default()


def write_shot(src: str, dst: Path, label: str, width: int, quality: int, draw_label: bool) -> None:
    """Save ``src`` as a JPEG at most ``width`` wide, with ``label`` in a bar on top."""
    from PIL import Image, ImageDraw

    with Image.open(src) as opened:
        im = opened.convert("RGB")
    if im.width > width:
        im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS)
    if draw_label:
        bar = max(24, im.height // 26)
        canvas = Image.new("RGB", (im.width, im.height + bar), (18, 18, 18))
        canvas.paste(im, (0, bar))
        d = ImageDraw.Draw(canvas)
        d.text((8, max(2, bar // 6)), label, fill=(255, 220, 120), font=_font(int(bar * 0.7)))
        im = canvas
    im.save(dst, "JPEG", quality=quality, optimize=True)

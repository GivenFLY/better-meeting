"""Stage 4: OCR every keyframe and keep only the text that is new on screen."""

from __future__ import annotations

import difflib
import re
from typing import Any

from ..utils import log
from .registry import get_backend


def norm(text: str) -> str:
    """Collapse whitespace, so re-wrapped lines compare equal."""
    return re.sub(r"\s+", " ", text).strip()


def added_lines(new: str, old: str) -> list[str]:
    """Lines of ``new`` that did not appear in ``old`` (whitespace-insensitive, len > 1)."""
    old_set = {norm(line) for line in old.splitlines() if norm(line)}
    out = []
    for line in new.splitlines():
        n = norm(line)
        if n and n not in old_set and len(n) > 1:
            out.append(line.rstrip())
    return out


def ocr_frames(
    frames: list[dict[str, Any]], backend_name: str, langs: list[str], sim_threshold: float
) -> list[dict[str, Any]]:
    """``[{t, path}]`` -> screen events ``[{t, path, added}]``.

    A frame whose text is more similar than ``sim_threshold`` to the previous
    kept frame counts as the same screen and is skipped.
    """
    backend = get_backend(backend_name)
    log(f"OCR ({backend.name}) over {len(frames)} frames")
    events: list[dict[str, Any]] = []
    prev_text = ""
    for i, fr in enumerate(frames):
        text = backend.recognize(fr["path"], langs)
        if (
            prev_text
            and difflib.SequenceMatcher(None, norm(prev_text), norm(text)).ratio() > sim_threshold
        ):
            continue
        new = added_lines(text, prev_text)
        if new or not prev_text:
            events.append({"t": fr["t"], "path": fr["path"], "added": new})
        prev_text = text
        if (i + 1) % 25 == 0:
            log(f"  {i + 1}/{len(frames)}")
    log(f"{len(events)} screen events with new text")
    return events

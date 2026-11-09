"""Small helpers shared by every stage: timecodes, logging, subprocess, JSON."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger("better_meeting")


class BmError(Exception):
    """A user-facing failure: printed as ``[bm] error: ...`` and exit code 1."""


def setup_logging(level: int = logging.INFO) -> None:
    """Route the package logger to stderr with a ``[bm]`` prefix (idempotent)."""
    if not any(getattr(h, "_bm", False) for h in logger.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("[bm] %(message)s"))
        handler._bm = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)


def log(msg: str) -> None:
    logger.info(msg)


def ts(seconds: float) -> str:
    """Seconds -> ``MM:SS``, or ``HH:MM:SS`` from one hour on."""
    s = round(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def ts_file(seconds: float) -> str:
    """Seconds -> ``HH-MM-SS``, safe for file names."""
    s = round(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:02d}-{m:02d}-{sec:02d}"


def parse_ts(s: str) -> float:
    """``'75'``, ``'01:15'``, ``'1:01:15'``, ``'12.5'`` -> seconds."""
    try:
        nums = [float(p) for p in s.split(":")]
    except ValueError:
        nums = []
    if not nums or len(nums) > 3 or any(n < 0 for n in nums):
        raise BmError(f"cannot parse timecode {s!r} (expected SS, MM:SS or HH:MM:SS)")
    t = 0.0
    for n in nums:
        t = t * 60 + n
    return t


def run(cmd: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
    """Run a command, raise :class:`BmError` with the tail of stderr on failure."""
    res = subprocess.run(cmd, capture_output=True, text=True, **kw)
    if res.returncode != 0:
        tail = res.stderr.strip()[-500:] or "no output"
        raise BmError(f"`{cmd[0]}` failed (exit {res.returncode}): {tail}")
    return res


def need(binary: str) -> None:
    if shutil.which(binary) is None:
        raise BmError(f"`{binary}` not found on PATH")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def probe_duration(video: Path) -> float:
    """Container duration in seconds, 0.0 if ffprobe cannot tell."""
    need("ffprobe")
    res = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video),
        ]
    )
    try:
        return float(res.stdout.strip())
    except ValueError:
        return 0.0

"""Stage 1: extract the audio track as 16 kHz mono WAV for ASR."""

from __future__ import annotations

from pathlib import Path

from .utils import log, need, run


def extract_audio(
    video: Path, wav: Path, start: float | None = None, duration: float | None = None
) -> None:
    """Write ``wav`` from ``video``, optionally only ``[start, start + duration)``."""
    need("ffmpeg")
    log("extracting audio")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vn"]
    # The range is cut after decoding. Before -i, -ss seeks by one stream, the video
    # stream if there is one: a range that starts before the first frame of that
    # stream then loses its audio.
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    run(cmd)

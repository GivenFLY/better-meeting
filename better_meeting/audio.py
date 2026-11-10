"""Stage 1: extract the audio as 16 kHz mono WAV for ASR."""

from __future__ import annotations

from pathlib import Path

from .utils import log, need, run


def _audio_stream_count(video: Path) -> int:
    need("ffprobe")
    res = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(video),
        ]
    )
    # A set: some containers list a stream once more under its program.
    return len(set(res.stdout.split()))


def extract_audio(
    video: Path, wav: Path, start: float | None = None, duration: float | None = None
) -> None:
    """Write ``wav`` from ``video``, optionally only ``[start, start + duration)``.

    All audio streams are mixed. A recorder that captures the microphone and the
    system audio separately writes one stream for each, and ffmpeg alone keeps
    only one of them.
    """
    need("ffmpeg")
    log("extracting audio")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    if duration is not None:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-i", str(video), "-vn"]
    streams = _audio_stream_count(video)
    if streams > 1:
        log(f"mixing {streams} audio streams")
        inputs = "".join(f"[0:a:{i}]" for i in range(streams))
        cmd += ["-filter_complex", f"{inputs}amix=inputs={streams}[mix]", "-map", "[mix]"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    run(cmd)

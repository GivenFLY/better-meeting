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

    Each stream is in the mix at its own start time and at the level it has when
    it is extracted alone. Where the sum would pass full scale, a limiter turns it
    down. The limiter delays the mix by its look-ahead, 5 ms, and the last 5 ms of
    the audio are not written.
    """
    need("ffmpeg")
    log("extracting audio")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    span = ["-ss", f"{start:.3f}"] if start else []
    if duration is not None:
        span += ["-t", f"{duration:.3f}"]
    streams = _audio_stream_count(video)
    if streams > 1:
        log(f"mixing {streams} audio streams")
        # amix mixes the streams from their first samples, whatever their start times.
        # aresample pads a stream with silence up to its start time and across a gap
        # in its timestamps. aformat converts each stream as a single stream is
        # converted, so that a stereo and a mono stream meet at their own levels.
        # normalize=0: by default amix divides every stream by the number of streams,
        # the silent ones too, and changes the level when a stream ends.
        each = (
            "aresample=async=1:first_pts=0,"
            "aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono"
        )
        chains = ";".join(f"[0:a:{i}]{each}[a{i}]" for i in range(streams))
        inputs = "".join(f"[a{i}]" for i in range(streams))
        graph = f"{chains};{inputs}amix=inputs={streams}:normalize=0,alimiter[mix]"
        # The range is cut after the mix. Before -i, -ss seeks by one stream: if that
        # stream starts after the seek point, the other streams lose the audio between.
        # And without -ss, -t counts from the start of each stream.
        cmd += ["-i", str(video), "-vn", "-filter_complex", graph, "-map", "[mix]", *span]
    else:
        cmd += [*span, "-i", str(video), "-vn"]
    cmd += ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    run(cmd)

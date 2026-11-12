"""Stage 1: extract the audio as 16 kHz mono WAV for ASR."""

from __future__ import annotations

import json
from pathlib import Path

from .utils import BmError, log, need, run


def _seconds(value: object) -> float:
    try:
        return float(value)  # ffprobe writes "N/A" when it does not know
    except (TypeError, ValueError):
        return 0.0


def _audio_stream_starts(video: Path) -> list[float]:
    """For each audio stream, the seconds from the start of the file to its first sample."""
    need("ffprobe")
    res = run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=start_time:format=start_time",
            "-of",
            "json",
            str(video),
        ]
    )
    info = json.loads(res.stdout)
    file_start = _seconds(info.get("format", {}).get("start_time"))
    return [max(0.0, _seconds(s.get("start_time")) - file_start) for s in info.get("streams", [])]


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
    the audio are not written. If ffmpeg cannot make the mix, one stream is used.
    """
    need("ffmpeg")
    log("extracting audio")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    span = ["-ss", f"{start:.3f}"] if start else []
    if duration is not None:
        span += ["-t", f"{duration:.3f}"]
    out = ["-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le", str(wav)]
    starts = _audio_stream_starts(video)
    if len(starts) > 1:
        log(f"mixing {len(starts)} audio streams")
        # aformat converts each stream as a single stream is converted, so that a
        # stereo and a mono stream meet at their own levels. amix mixes the streams
        # from their first samples, whatever their start times, so adelay puts
        # silence before a stream that starts late. normalize=0: by default amix
        # divides every stream by the number of streams, the silent ones too, and
        # changes the level when a stream ends. asetpts counts the time of the mix
        # from 0, for -ss and -t.
        #
        # aresample with first_pts=0 also puts a late stream at its time, but it
        # drops samples where timestamps overlap, and it adds silence as long as
        # the elapsed time when the audio parameters change in the file.
        mono = "aformat=sample_fmts=s16:sample_rates=16000:channel_layouts=mono"
        chains = [
            f"[0:a:{i}]{mono}" + (f",adelay={late * 1000:.3f}" if late >= 0.001 else "") + f"[a{i}]"
            for i, late in enumerate(starts)
        ]
        inputs = "".join(f"[a{i}]" for i in range(len(starts)))
        mix = f"{inputs}amix=inputs={len(starts)}:normalize=0,alimiter,asetpts=N/SR/TB[mix]"
        graph = ";".join([*chains, mix])
        # The range is cut after the mix. Before -i, -ss seeks by one stream: if that
        # stream starts after the seek point, the other streams lose the audio between.
        # And without -ss, -t counts from the start of each stream.
        mixed = ["-i", str(video), "-vn", "-filter_complex", graph, "-map", "[mix]", *span]
        try:
            run([*cmd, *mixed, *out])
            return
        except BmError as e:
            # For example a stream that ffmpeg cannot decode, or an ffmpeg before 4.4,
            # which does not have normalize.
            log(f"could not mix the audio streams, one stream is used: {str(e).splitlines()[-1]}")
    run([*cmd, *span, "-i", str(video), "-vn", *out])

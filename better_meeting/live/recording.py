"""Finish a live session: one recording file, then the usual artifacts.

The recording holds the window's video and one audio stream per track, so
``bm extract`` and the point queries work on it like on any other recording.
The live transcript is stored as the transcription stage's cache: the pipeline
then only has to do the frames, the OCR and the bundle.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..asr import pipeline_meta
from ..bundle import write_bundle
from ..pipeline import run_pipeline
from ..utils import log, need, probe_duration, run, save

#: H.264 codes a frame in blocks of this many pixels and crops the last ones.
BLOCK = 16


def crop(video: Path, picture: tuple[int, int]) -> None:
    """Crop ``video`` in place to ``picture`` (width, height) from its top left corner.

    The frames are not encoded again: the stream is told to show less of them. The
    file is then written once more, so the container reports the new size too.
    """
    need("ffprobe")
    entries = "stream=codec_name,width,height"
    probe = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", entries]
    codec, width, height = run([*probe, "-of", "csv=p=0", str(video)]).stdout.strip().split(",")
    if codec != "h264" or picture[0] > int(width) or picture[1] > int(height):
        return
    if picture == (int(width), int(height)):
        return
    # The new cropping replaces the stream's own, so it counts from the coded size.
    right = -(-int(width) // BLOCK) * BLOCK - picture[0]
    bottom = -(-int(height) // BLOCK) * BLOCK - picture[1]
    ffmpeg = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i"]
    marked, cropped = video.with_name("video_marked.mp4"), video.with_name("video_cropped.mp4")
    bsf = f"h264_metadata=crop_right={right}:crop_bottom={bottom}"
    run([*ffmpeg, str(video), "-c", "copy", "-bsf:v", bsf, str(marked)])
    run([*ffmpeg, str(marked), "-c", "copy", str(cropped)])
    marked.unlink()
    cropped.replace(video)


def shrink(recording: Path, bitrate: str | None, scale: float) -> None:
    """Encode the video of ``recording`` again in place: scaled, at ``bitrate``.

    It runs after the artifacts are built, so the screenshots and the OCR come
    from the picture as it was captured. Without ``bitrate`` the encoder keeps a
    constant quality. The hardware encoder is used where ffmpeg has it: a long
    recording of a large screen takes minutes.
    """
    if recording.suffix != ".mp4" or not (bitrate or scale != 1.0):
        return
    encoders = run(["ffmpeg", "-hide_banner", "-encoders"]).stdout
    if "h264_videotoolbox" in encoders:
        codec = ["-c:v", "h264_videotoolbox", *(["-b:v", bitrate] if bitrate else ["-q:v", "60"])]
    else:
        codec = ["-c:v", "libx264", "-preset", "veryfast"]
        codec += ["-b:v", bitrate] if bitrate else ["-crf", "23"]
    filters = [f"scale=trunc(iw*{scale}/2)*2:trunc(ih*{scale}/2)*2"] if scale != 1.0 else []
    filters.append("format=yuv420p")
    encoded = recording.with_name(recording.stem + ".encoding.mp4")
    log("encoding the video again")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(recording)]
    cmd += ["-map", "0", "-c:a", "copy", "-vf", ",".join(filters)]
    # The capture skips frames while nothing changes: keep their times as they are.
    cmd += ["-fps_mode", "passthrough", *codec, "-movflags", "+faststart", str(encoded)]
    try:
        run(cmd)
    except BaseException:
        encoded.unlink(missing_ok=True)  # the recording as captured stays
        raise
    encoded.replace(recording)


def mux(
    work: Path, tracks: tuple[str, ...], dest: Path, picture: tuple[int, int] | None = None
) -> Path:
    """Write the session's video and track WAVs into one file and delete the parts.

    ``picture`` is the part of the video's frame that holds the window, if the
    frame is larger (:attr:`Capture.picture`). Returns the recording: ``dest``
    with ``.mp4``, or ``.m4a`` if there is no video.
    """
    need("ffmpeg")
    video = work / "video.mp4"
    wavs = [work / f"track_{t}.wav" for t in tracks]
    has_video = video.exists() and video.stat().st_size > 0
    if has_video and picture is not None:
        crop(video, picture)
    out = dest.with_suffix(".mp4" if has_video else ".m4a")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    inputs = [video, *wavs] if has_video else wavs
    for path in inputs:
        cmd += ["-i", str(path)]
    first = 1 if has_video else 0
    if has_video:
        cmd += ["-map", "0:v", "-c:v", "copy"]
    for i, track in enumerate(tracks):
        cmd += ["-map", f"{first + i}:a"]
        cmd += [
            f"-metadata:s:a:{i}",
            f"title={track}",
            f"-metadata:s:a:{i}",
            f"handler_name={track}",
        ]
    cmd += ["-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(out)]
    log(f"writing the recording: {out}")
    run(cmd)
    video.unlink(missing_ok=True)
    for wav in wavs:
        wav.unlink()
    return out


def finish(
    recording: Path, work: Path, transcript: list[dict[str, Any]], a: argparse.Namespace
) -> None:
    """Build ``work/artifacts`` for ``recording`` around the live ``transcript``.

    ``a`` is an ``extract`` namespace for ``recording`` whose ASR parameters are
    the ones the session ran with, so the stored transcript counts as up to date.
    """
    langs = [code.strip() for code in a.langs.split(",") if code.strip()]
    save(work / "transcript.json", transcript)
    save(
        work / "transcript.meta.json",
        pipeline_meta(a.lang, langs, a.asr_model, a.asr_backend, a.asr_opts),
    )
    if recording.suffix == ".mp4":
        run_pipeline(a)
        return
    # Microphone only: there is no picture, so no frames and no OCR.
    a.only = "transcribe"
    run_pipeline(a)
    write_bundle(
        work / "artifacts",
        transcript,
        [],
        [],
        probe_duration(recording),
        max_shots=a.max_shots,
        shot_width=a.shot_width,
        shot_quality=a.shot_quality,
        draw_label=not a.no_label,
        pause=a.pause,
    )

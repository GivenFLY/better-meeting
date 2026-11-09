"""Stage glue with caching in the per-video working folder.

Each stage writes its result next to a ``<stage>.meta.json`` holding the
parameters it was built with. A stage re-runs when ``--force`` is given, its
output is missing, its parameters changed, or an upstream stage re-ran.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

from .asr import pipeline_meta, resolve, transcribe
from .audio import extract_audio
from .bundle import write_bundle
from .frames import extract_full_frames, pick_keyframes, sample_thumbs
from .ocr import ocr_frames
from .utils import BmError, load, log, probe_duration, save


def _stale(out: Path, meta_file: Path, params: dict[str, Any], force: bool) -> bool:
    """True if ``out`` must be rebuilt for ``params``."""
    if force or not out.exists() or not meta_file.exists():
        return True
    try:
        return load(meta_file) != params
    except ValueError:
        return True


def run_pipeline(a: argparse.Namespace) -> None:
    if not a.video.exists():
        raise BmError(f"no such file: {a.video}")
    work: Path = a.out
    work.mkdir(parents=True, exist_ok=True)
    log(f"working folder: {work}")

    wav = work / "audio.wav"
    f_transcript = work / "transcript.json"
    f_keyframes = work / "keyframes.json"
    f_screen = work / "screen.json"
    d_thumbs = work / "thumbs"
    d_frames = work / "frames"

    # 1. audio
    if a.force or not wav.exists():
        extract_audio(a.video, wav)
    if a.only == "audio":
        return
    duration = probe_duration(a.video)

    # 2. transcribe
    langs = [code.strip() for code in a.langs.split(",") if code.strip()]
    backend, model = resolve(a.asr_backend, a.asr_model)
    meta = pipeline_meta(a.lang, langs, model, backend.name, a.asr_opts)
    if _stale(f_transcript, work / "transcript.meta.json", meta, a.force):
        save(f_transcript, transcribe(wav, a.lang, langs, model, backend.name, work, a.asr_opts))
        save(work / "transcript.meta.json", meta)
    transcript = load(f_transcript)
    log(f"transcript: {len(transcript)} segments")
    if a.only == "transcribe":
        return

    # 3. frames: thumbnails -> keyframes -> full-size frames
    kf_params = {
        "interval": a.frame_interval,
        "cell_delta": a.cell_delta,
        "min_cells": a.min_cells,
        "max_gap": a.max_gap,
    }
    frames_dirty = False
    if _stale(f_keyframes, work / "keyframes.meta.json", kf_params, a.force):
        save(
            f_keyframes,
            pick_keyframes(
                sample_thumbs(a.video, d_thumbs, a.frame_interval),
                a.cell_delta,
                a.min_cells,
                a.max_gap,
            ),
        )
        save(work / "keyframes.meta.json", kf_params)
        frames_dirty = True
    fr_params = {"width": a.frame_width}
    if frames_dirty or _stale(d_frames, work / "frames.meta.json", fr_params, a.force):
        if d_frames.exists():
            shutil.rmtree(d_frames)
        save(work / "frames.meta.json", fr_params)
        frames_dirty = True
    frames = extract_full_frames(a.video, load(f_keyframes), d_frames, a.frame_width)
    if a.only == "frames":
        return

    # 4. OCR
    if a.ocr == "none":
        screen: list[dict[str, Any]] = []
    else:
        ocr_params = {"backend": a.ocr, "langs": a.ocr_lang, "sim": a.ocr_sim}
        if frames_dirty or _stale(f_screen, work / "screen.meta.json", ocr_params, a.force):
            save(f_screen, ocr_frames(frames, a.ocr, a.ocr_lang.split(","), a.ocr_sim))
            save(work / "screen.meta.json", ocr_params)
        screen = load(f_screen)
    if a.only == "ocr":
        return

    # 5. bundle (cheap, always rebuilt)
    write_bundle(
        work / "artifacts",
        transcript,
        screen,
        frames,
        duration,
        max_shots=a.max_shots,
        shot_width=a.shot_width,
        shot_quality=a.shot_quality,
        draw_label=not a.no_label,
        pause=a.pause,
    )

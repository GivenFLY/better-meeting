"""Stage 2: timed transcription, provider-agnostic.

Whisper-style providers (a language hint per pass):
  ``--lang uk``    one pass in that language.
  ``--lang auto``  one full pass per language in ``--langs`` (default uk,ru,en),
                   then a merge: each stretch of the timeline keeps the segments
                   from the pass the provider was most confident about. Survives
                   any language mixing, down to alternating sentences.

Single-pass providers (multilingual, no language hint, e.g. Parakeet):
  one pass regardless of ``--lang``/``--langs``; a pinned ``--lang`` only labels
  the segments.

Every full pass is cached in the working folder (``pass_<lang>.json``, or
``pass_<backend>.json`` for single-pass providers) together with the model,
backend and options it was made with, so an interrupted run resumes and a
changed parameter re-runs only what it affects.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from ..utils import load, log, save
from .base import AsrBackend, RawSegment, Segment
from .registry import get_backend


def public(s: RawSegment | Segment) -> Segment:
    """Strip provider scores, keep the stable segment fields."""
    return {"start": s["start"], "end": s["end"], "text": s["text"], "lang": s["lang"]}


def pipeline_meta(
    lang: str, langs: list[str], model: str, backend: str, opts: dict[str, Any]
) -> dict[str, Any]:
    """Parameters ``transcript.json`` depends on, stored as ``transcript.meta.json``.

    ``model`` and ``backend`` must be resolved (no ``auto``, no ``None``).
    """
    return {
        "lang": lang,
        "langs": ",".join(langs),
        "model": model,
        "backend": backend,
        "opts": opts,
    }


def resolve(backend_name: str, model: str | None) -> tuple[AsrBackend, str]:
    """``(provider, concrete model id)`` for CLI values (``auto`` / ``None`` allowed)."""
    backend = get_backend(backend_name)
    return backend, backend.resolve_model(model)


def pass_path(work: Path, backend: AsrBackend, lang: str | None) -> Path:
    if backend.single_pass:
        return work / f"pass_{backend.name}.json"
    return work / f"pass_{lang}.json"


def _matches(data: Any, backend: AsrBackend, model: str, opts: dict[str, Any]) -> bool:
    return (
        isinstance(data, dict)
        and data.get("model") == model
        and data.get("backend") == backend.name
        and data.get("opts", {}) == opts
    )


def _cached_pass(
    wav: Path, work: Path, backend: AsrBackend, lang: str | None, model: str, opts: dict[str, Any]
) -> list[RawSegment]:
    cache = pass_path(work, backend, lang)
    label = backend.name if backend.single_pass else lang
    cached = load(cache) if cache.exists() else None
    if _matches(cached, backend, model, opts):
        segs: list[RawSegment] = cached["segments"]
        log(f"pass [{label}]: cached, {len(segs)} segments")
        return segs
    log(f"pass [{label}]")
    segs = backend.transcribe(wav, lang, model, opts)
    save(cache, {"model": model, "backend": backend.name, "opts": opts, "segments": segs})
    return segs


def _log_shares(prefix: str, segs: Iterable[Segment]) -> None:
    shares = Counter(s["lang"] for s in segs)
    log(prefix + ", ".join(f"{lang}: {n}" for lang, n in shares.most_common()))


def transcribe(
    wav: Path,
    lang: str,
    langs: list[str],
    model: str | None,
    backend_name: str,
    work: Path,
    opts: dict[str, Any] | None = None,
) -> list[Segment]:
    """Full-recording transcript, sorted by time. ``lang`` is a code or ``auto``."""
    opts = opts or {}
    backend, model = resolve(backend_name, model)
    backend.check_options(opts)

    if backend.single_pass:
        if lang == "auto" and len(langs) > 1:
            log(f"{backend.name}: one pass for all languages, --langs has no effect")
        segs = [public(s) for s in _cached_pass(wav, work, backend, None, model, opts)]
        if lang != "auto":
            segs = [{**s, "lang": lang} for s in segs]
        _log_shares("languages: ", segs)
        return segs

    if lang != "auto":
        return [public(s) for s in _cached_pass(wav, work, backend, lang, model, opts)]

    passes = [_cached_pass(wav, work, backend, code, model, opts) for code in langs]
    merged = [public(s) for s in merge_passes(passes)]
    _log_shares("merged: ", merged)
    return merged


def is_hallucination(s: RawSegment) -> bool:
    """Whisper's standard silence-artifact heuristic: likely no speech and low confidence."""
    return s["nospeech"] > 0.6 and s["score"] < -1.0


def _adds_speech(s: RawSegment, cursor: float) -> bool:
    """True if ``s`` still adds speech once the timeline is covered up to ``cursor``.

    A segment that starts at or after the cursor always does. One that crosses the
    cursor must reach more than 0.2 s past it and add at least half its length
    there. Less than that is another pass's wording of a stretch already covered.
    """
    if s["start"] >= cursor:
        return s["end"] > cursor
    return s["end"] > cursor + 0.2 and s["end"] - cursor >= (s["end"] - s["start"]) / 2


def merge_passes(passes: Iterable[list[RawSegment]]) -> list[RawSegment]:
    """Greedy timeline cover: at each point take the most confident segment covering it.

    Segments are first filtered with :func:`is_hallucination`. The cursor walks
    forward. Each pass offers its next segment that still adds speech past the
    cursor (:func:`_adds_speech`), provided it starts no later than 2 s past the
    cursor, so short gaps do not split the cover. The most confident offer wins.

    Only the next segment of a pass competes. If a later one could win, the
    segment before it would never be taken.
    """
    kept = [
        sorted((s for s in p if not is_hallucination(s)), key=lambda s: s["start"]) for p in passes
    ]
    starts = [s["start"] for p in kept for s in p]
    if not starts:
        return []

    out: list[RawSegment] = []
    cursor = min(starts)
    while True:
        offers = (next((s for s in p if _adds_speech(s, cursor)), None) for p in kept)
        cands = [s for s in offers if s is not None and s["start"] <= cursor + 2.0]
        if not cands:
            rest = [t for t in starts if t > cursor]
            if not rest:
                break
            cursor = min(rest)
            continue
        # On equal scores the earlier segment wins.
        best = max(cands, key=lambda s: (s["score"], -s["start"]))
        out.append(best)
        cursor = best["end"]
    return out


def transcribe_range(
    video: Path,
    start: float,
    end: float | None,
    lang: str | None,
    model: str | None,
    backend_name: str,
    work: Path,
    opts: dict[str, Any],
    force: bool,
) -> list[Segment]:
    """Point query: transcript of ``[start, end)`` with arbitrary provider options.

    Cached by the full parameter set under ``work/transcripts_at/``. If the video
    already went through ``extract`` with the same model, backend and options,
    the range is sliced from the pipeline's cache with no new ASR run.
    """
    backend, model = resolve(backend_name, model)
    backend.check_options(opts)  # before extracting audio, so no clip is left over
    key = {
        "from": round(start, 3),
        "to": round(end, 3) if end is not None else None,
        "lang": lang,
        "model": model,
        "backend": backend.name,
        "opts": opts,
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[
        :10
    ]
    cdir = work / "transcripts_at"
    span = f"{int(start):05d}-{'end' if end is None else f'{int(end):05d}'}"
    cache = cdir / f"{span}_{lang or 'auto'}_{digest}.json"

    if cache.exists() and not force:
        log(f"cached: {cache.name}")
        return load(cache)["segments"]

    if not force:
        reused = _slice_pipeline_cache(work, start, end, lang, model, backend, opts)
        if reused is not None:
            log("served from the full-pipeline cache (extract)")
            return reused

    from ..audio import extract_audio

    cdir.mkdir(parents=True, exist_ok=True)
    clip = cdir / f"{digest}.wav"
    extract_audio(video, clip, start=start, duration=None if end is None else end - start)
    try:
        segments = [public(s) for s in backend.transcribe(clip, lang, model, opts)]
    finally:
        clip.unlink(missing_ok=True)
    for s in segments:
        s["start"] += start
        s["end"] += start
    save(cache, {"params": key, "segments": segments})
    return segments


def _slice_pipeline_cache(
    work: Path,
    start: float,
    end: float | None,
    lang: str | None,
    model: str,
    backend: AsrBackend,
    opts: dict[str, Any],
) -> list[Segment] | None:
    """Slice a range out of ``extract``'s cache, or ``None`` if nothing usable exists.

    - single-pass provider: its one pass (a pinned ``lang`` relabels segments);
    - pinned ``lang``: that language's pass;
    - no ``lang``: the merged ``transcript.json``, if its meta says it was built
      in auto mode with the same model, backend and options.
    """

    def in_range(s: Segment) -> bool:
        return s["end"] > start and (end is None or s["start"] < end)

    if backend.single_pass or lang is not None:
        f = pass_path(work, backend, lang)
        if not f.exists():
            return None
        data = load(f)
        if not _matches(data, backend, model, opts):
            return None
        segs = [public(s) for s in data["segments"] if in_range(s)]
        if lang is not None:
            segs = [{**s, "lang": lang} for s in segs]
        return segs

    f_transcript, f_meta = work / "transcript.json", work / "transcript.meta.json"
    if not (f_transcript.exists() and f_meta.exists()):
        return None
    meta = load(f_meta)
    if (
        meta.get("lang") == "auto"
        and meta.get("model") == model
        and meta.get("backend") == backend.name
        and meta.get("opts", {}) == opts
    ):
        return [public(s) for s in load(f_transcript) if in_range(s)]
    return None

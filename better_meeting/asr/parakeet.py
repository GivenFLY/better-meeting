"""NVIDIA Parakeet TDT via parakeet-mlx (local, macOS).

One multilingual decode for the whole recording; the model takes no language
hint, so segments are labelled by a script heuristic unless ``--lang`` is pinned.
"""

from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from ..utils import BmError
from .base import AsrBackend, RawSegment
from .langid import label_segments
from .whisper_mlx import WHISPER_MODEL

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v3"


class ParakeetBackend(AsrBackend):
    name = "parakeet"
    default_model = PARAKEET_MODEL
    single_pass = True
    accepts_options = False

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def resolve_model(self, model: str | None) -> str:
        """Takes a Hugging Face repo id; the Whisper default maps to the Parakeet default."""
        if model is None or model == WHISPER_MODEL:
            return PARAKEET_MODEL
        if "/" in model:
            return model
        raise BmError(
            f"--asr-model {model!r} does not look like a Parakeet model; "
            f"pass a Hugging Face repo id, e.g. {PARAKEET_MODEL}"
        )

    def transcribe(
        self, wav: Path, lang: str | None, model: str, opts: dict[str, Any]
    ) -> list[RawSegment]:
        self.check_options(opts)
        if platform.system() != "Darwin":
            raise BmError(
                "the parakeet backend is macOS-only for now (parakeet-mlx); "
                "use --asr-backend faster"
            )
        try:
            from parakeet_mlx import from_pretrained  # type: ignore[import-not-found]
        except ImportError:
            raise BmError(
                'parakeet-mlx is not installed: `pip install "better-meeting[parakeet]"`'
            ) from None
        if model not in self._models:
            self._models[model] = from_pretrained(model)
        # Long audio is decoded in 2-minute chunks with overlap, like the parakeet-mlx CLI.
        res = self._models[model].transcribe(str(wav), chunk_duration=120.0, overlap_duration=15.0)
        segs: list[RawSegment] = [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "lang": lang,
                "score": float(s.confidence),
                "nospeech": 0.0,
            }
            for s in res.sentences
            if s.text.strip()
        ]
        if lang is None:
            label_segments(segs)
        return segs

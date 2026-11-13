"""Whisper via faster-whisper / CTranslate2 (local, any platform)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from better_meeting.utils import BmError

from ..base import AsrBackend, RawSegment
from .mlx_whisper import WHISPER_MODEL


class FasterWhisperBackend(AsrBackend):
    name = "faster"
    default_model = WHISPER_MODEL

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}

    def transcribe(
        self, wav: Path, lang: str | None, model: str, opts: dict[str, Any]
    ) -> list[RawSegment]:
        try:
            from faster_whisper import WhisperModel  # type: ignore[import-not-found]
        except ImportError:
            raise BmError(
                'faster-whisper is not installed: `pip install "better-meeting[faster]"`'
            ) from None
        if model not in self._models:
            self._models[model] = WhisperModel(model, device="auto", compute_type="int8")
        kwargs = {"language": lang, "vad_filter": True, "condition_on_previous_text": False, **opts}
        segs, info = self._models[model].transcribe(str(wav), **kwargs)
        detected = lang or info.language
        return [
            {
                "start": float(s.start),
                "end": float(s.end),
                "text": s.text.strip(),
                "lang": detected,
                "score": float(s.avg_logprob),
                "nospeech": float(s.no_speech_prob),
            }
            for s in segs
            if s.text.strip()
        ]

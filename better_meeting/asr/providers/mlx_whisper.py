"""Whisper on Apple Silicon via mlx-whisper (local, macOS)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from better_meeting.utils import BmError

from ..base import AsrBackend, RawSegment

WHISPER_MODEL = "large-v3-turbo"


class MlxWhisperBackend(AsrBackend):
    name = "mlx"
    default_model = WHISPER_MODEL

    def transcribe(
        self, wav: Path, lang: str | None, model: str, opts: dict[str, Any]
    ) -> list[RawSegment]:
        try:
            import mlx_whisper  # type: ignore[import-not-found]
        except ImportError:
            raise BmError(
                "mlx-whisper is not installed: `pip install mlx-whisper`, "
                "or use --asr-backend faster"
            ) from None
        repo = model if "/" in model else f"mlx-community/whisper-{model}"
        kwargs = {"language": lang, "condition_on_previous_text": False, **opts}
        res = mlx_whisper.transcribe(str(wav), path_or_hf_repo=repo, **kwargs)
        detected = lang or res.get("language")
        return [
            {
                "start": float(s["start"]),
                "end": float(s["end"]),
                "text": s["text"].strip(),
                "lang": detected,
                "score": float(s.get("avg_logprob", 0.0)),
                "nospeech": float(s.get("no_speech_prob", 0.0)),
            }
            for s in res["segments"]
            if s["text"].strip()
        ]

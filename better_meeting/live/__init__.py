"""Live sessions: transcribe a meeting while it runs, then build the usual artifacts.

capture/    pluggable window + audio capture providers
segmenter   cuts each audio track into utterances at pauses
echo        drops what the microphone only heard from the speakers
session     the live loop: utterances -> ASR provider -> transcript lines
recording   muxes the recording and hands it to the pipeline
"""

from __future__ import annotations

import importlib.util
import platform

from .capture import TRACKS, Capture, Window, capture_names, get_capture, register_capture
from .recording import finish, mux, shrink
from .session import LiveOptions, run_session

#: ``--source`` values -> the tracks they capture.
SOURCES: dict[str, tuple[str, ...]] = {
    "app": ("app",),
    "app+mic": ("app", "mic"),
    "mic": ("mic",),
}


def default_backend_name() -> str:
    """The ASR provider for ``--asr-backend auto`` in a live session.

    Parakeet if it is installed: it decodes an utterance several times faster than
    Whisper and keeps both languages of an utterance that mixes two. Whisper decodes
    an utterance in one language.
    """
    if platform.system() == "Darwin" and importlib.util.find_spec("parakeet_mlx"):
        return "parakeet"
    return "auto"


__all__ = [
    "SOURCES",
    "TRACKS",
    "Capture",
    "LiveOptions",
    "Window",
    "capture_names",
    "default_backend_name",
    "finish",
    "get_capture",
    "mux",
    "register_capture",
    "run_session",
    "shrink",
]

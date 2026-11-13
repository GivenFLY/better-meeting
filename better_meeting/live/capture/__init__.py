"""Window and audio capture for live sessions: provider contract plus registry."""

from .base import TRACKS, AudioHandler, Capture, Window
from .registry import capture_names, get_capture, register_capture

__all__ = [
    "TRACKS",
    "AudioHandler",
    "Capture",
    "Window",
    "capture_names",
    "get_capture",
    "register_capture",
]

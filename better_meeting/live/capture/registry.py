"""Capture provider registry."""

from __future__ import annotations

import platform
from collections.abc import Callable

from better_meeting.registry import Registry
from better_meeting.utils import BmError

from .base import Capture

_CAPTURES: Registry[Capture] = Registry(
    "capture provider",
    {
        "screencapturekit": "better_meeting.live.capture.providers.screencapturekit:"
        "ScreenCaptureKitCapture",
    },
)


def register_capture(name: str, factory: Callable[[], Capture] | str) -> None:
    """Add or replace a provider: a zero-argument callable, or ``"module:ClassName"``."""
    _CAPTURES.register(name, factory)


def capture_names() -> list[str]:
    """Names accepted by ``--capture`` (plus ``auto``)."""
    return _CAPTURES.names()


def get_capture(name: str) -> Capture:
    """A new provider instance for ``name``; ``auto`` is the platform default."""
    if name == "auto":
        if platform.system() != "Darwin":
            raise BmError("live capture is macOS-only for now (ScreenCaptureKit)")
        name = "screencapturekit"
    return _CAPTURES.create(name)

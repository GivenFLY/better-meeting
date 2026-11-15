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
        "windows": "better_meeting.live.capture.providers.windows:WindowsCapture",
    },
)

#: ``platform.system()`` -> the provider that captures there.
DEFAULTS = {"Darwin": "screencapturekit", "Windows": "windows"}


def register_capture(name: str, factory: Callable[[], Capture] | str) -> None:
    """Add or replace a provider: a zero-argument callable, or ``"module:ClassName"``."""
    _CAPTURES.register(name, factory)


def capture_names() -> list[str]:
    """Names accepted by ``--capture`` (plus ``auto``)."""
    return _CAPTURES.names()


def default_capture_name() -> str:
    """The provider for ``--capture auto`` on this platform."""
    name = DEFAULTS.get(platform.system())
    if name is None:
        raise BmError("live capture needs macOS 15+ or Windows 10 (version 2004) or later")
    return name


def get_capture(name: str) -> Capture:
    """A new provider instance for ``name``; ``auto`` is the platform default."""
    if name == "auto":
        name = default_capture_name()
    return _CAPTURES.create(name)

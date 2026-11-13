"""Capture provider registry, with the same lazy imports as the ASR and OCR registries."""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable

from ...utils import BmError
from .base import Capture

Factory = Callable[[], Capture]

_FACTORIES: dict[str, Factory | str] = {
    "screencapturekit": "better_meeting.live.capture.providers.screencapturekit:"
    "ScreenCaptureKitCapture",
}


def register_capture(name: str, factory: Factory | str) -> None:
    """Add or replace a provider: a zero-argument callable, or ``"module:ClassName"``."""
    _FACTORIES[name] = factory


def capture_names() -> list[str]:
    """Names accepted by ``--capture`` (plus ``auto``)."""
    return list(_FACTORIES)


def get_capture(name: str) -> Capture:
    """A new provider instance for ``name``; ``auto`` is the platform default."""
    if name == "auto":
        if platform.system() != "Darwin":
            raise BmError("live capture is macOS-only for now (ScreenCaptureKit)")
        name = "screencapturekit"
    if name not in _FACTORIES:
        raise BmError(f"unknown capture provider {name!r}; available: {', '.join(_FACTORIES)}")
    factory = _FACTORIES[name]
    if isinstance(factory, str):
        module, _, attr = factory.partition(":")
        factory = getattr(importlib.import_module(module), attr)
    return factory()

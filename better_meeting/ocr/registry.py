"""OCR provider registry."""

from __future__ import annotations

import platform
from collections.abc import Callable

from ..registry import Registry
from .base import OcrBackend

_BACKENDS: Registry[OcrBackend] = Registry(
    "OCR backend",
    {
        "vision": "better_meeting.ocr.providers.vision:VisionOcrBackend",
        "tesseract": "better_meeting.ocr.providers.tesseract:TesseractOcrBackend",
    },
)


def register_backend(name: str, factory: Callable[[], OcrBackend] | str) -> None:
    """Add or replace a provider (callable, or ``"module:Class"`` imported lazily)."""
    _BACKENDS.register(name, factory)


def backend_names() -> list[str]:
    """Names accepted by ``--ocr`` and ``bm ocr --backend``."""
    return _BACKENDS.names()


def default_backend_name() -> str:
    """macOS Vision on macOS, tesseract elsewhere."""
    return "vision" if platform.system() == "Darwin" else "tesseract"


def get_backend(name: str) -> OcrBackend:
    """The (cached) provider instance for ``name``."""
    return _BACKENDS.get(name)

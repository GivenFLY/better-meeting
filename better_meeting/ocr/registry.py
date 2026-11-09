"""OCR provider registry, with lazy imports like the ASR one."""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable

from ..utils import BmError
from .base import OcrBackend

Factory = Callable[[], OcrBackend]

_FACTORIES: dict[str, Factory | str] = {
    "vision": "better_meeting.ocr.providers.vision:VisionOcrBackend",
    "tesseract": "better_meeting.ocr.providers.tesseract:TesseractOcrBackend",
}
_INSTANCES: dict[str, OcrBackend] = {}


def register_backend(name: str, factory: Factory | str) -> None:
    """Add or replace a provider (callable, or ``"module:Class"`` imported lazily)."""
    _FACTORIES[name] = factory
    _INSTANCES.pop(name, None)


def backend_names() -> list[str]:
    return list(_FACTORIES)


def default_backend_name() -> str:
    """macOS Vision on macOS, tesseract elsewhere."""
    return "vision" if platform.system() == "Darwin" else "tesseract"


def get_backend(name: str) -> OcrBackend:
    if name not in _FACTORIES:
        raise BmError(f"unknown OCR backend {name!r}; available: {', '.join(_FACTORIES)}")
    if name not in _INSTANCES:
        factory = _FACTORIES[name]
        if isinstance(factory, str):
            module, _, attr = factory.partition(":")
            factory = getattr(importlib.import_module(module), attr)
        _INSTANCES[name] = factory()
    return _INSTANCES[name]

"""ASR provider registry.

Providers register under a name; the module behind a built-in provider is only
imported when that provider is first used, so optional dependencies stay optional.
"""

from __future__ import annotations

import importlib
import platform
from collections.abc import Callable

from ..utils import BmError
from .base import AsrBackend

Factory = Callable[[], AsrBackend]

_FACTORIES: dict[str, Factory | str] = {
    "mlx": "better_meeting.asr.whisper_mlx:MlxWhisperBackend",
    "faster": "better_meeting.asr.whisper_faster:FasterWhisperBackend",
    "parakeet": "better_meeting.asr.parakeet:ParakeetBackend",
}
_INSTANCES: dict[str, AsrBackend] = {}


def register_backend(name: str, factory: Factory | str) -> None:
    """Add or replace a provider.

    ``factory`` is a zero-argument callable returning an :class:`AsrBackend`, or a
    ``"package.module:ClassName"`` string imported lazily on first use.
    """
    _FACTORIES[name] = factory
    _INSTANCES.pop(name, None)


def backend_names() -> list[str]:
    """Names accepted by ``--asr-backend`` (plus ``auto``)."""
    return list(_FACTORIES)


def resolve_backend_name(name: str) -> str:
    """``auto`` -> the platform default: mlx-whisper on macOS, faster-whisper elsewhere."""
    if name == "auto":
        return "mlx" if platform.system() == "Darwin" else "faster"
    if name not in _FACTORIES:
        raise BmError(f"unknown ASR backend {name!r}; available: {', '.join(_FACTORIES)}")
    return name


def get_backend(name: str) -> AsrBackend:
    """The (cached) provider instance for ``name``; ``auto`` is resolved first."""
    name = resolve_backend_name(name)
    if name not in _INSTANCES:
        factory = _FACTORIES[name]
        if isinstance(factory, str):
            module, _, attr = factory.partition(":")
            factory = getattr(importlib.import_module(module), attr)
        _INSTANCES[name] = factory()
    return _INSTANCES[name]

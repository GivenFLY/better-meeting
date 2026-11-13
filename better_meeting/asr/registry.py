"""ASR provider registry."""

from __future__ import annotations

import platform
from collections.abc import Callable

from ..registry import Registry
from .base import AsrBackend

_BACKENDS: Registry[AsrBackend] = Registry(
    "ASR backend",
    {
        "mlx": "better_meeting.asr.providers.mlx_whisper:MlxWhisperBackend",
        "faster": "better_meeting.asr.providers.faster_whisper:FasterWhisperBackend",
        "parakeet": "better_meeting.asr.providers.parakeet:ParakeetBackend",
    },
)


def register_backend(name: str, factory: Callable[[], AsrBackend] | str) -> None:
    """Add or replace a provider.

    ``factory`` is a zero-argument callable returning an :class:`AsrBackend`, or a
    ``"package.module:ClassName"`` string imported lazily on first use.
    """
    _BACKENDS.register(name, factory)


def backend_names() -> list[str]:
    """Names accepted by ``--asr-backend`` (plus ``auto``)."""
    return _BACKENDS.names()


def resolve_backend_name(name: str) -> str:
    """``auto`` -> the platform default: mlx-whisper on macOS, faster-whisper elsewhere."""
    if name == "auto":
        return "mlx" if platform.system() == "Darwin" else "faster"
    _BACKENDS.check(name)
    return name


def get_backend(name: str) -> AsrBackend:
    """The (cached) provider instance for ``name``; ``auto`` is resolved first."""
    return _BACKENDS.get(resolve_backend_name(name))

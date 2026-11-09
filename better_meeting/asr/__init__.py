"""Speech recognition: provider registry plus the provider-agnostic transcription logic.

Adding a provider means subclassing :class:`AsrBackend` and calling
:func:`register_backend`; see ``CONTRIBUTING.md``.
"""

from .base import AsrBackend, RawSegment, Segment
from .core import merge_passes, pipeline_meta, resolve, transcribe, transcribe_range
from .registry import backend_names, get_backend, register_backend, resolve_backend_name

__all__ = [
    "AsrBackend",
    "RawSegment",
    "Segment",
    "backend_names",
    "get_backend",
    "merge_passes",
    "pipeline_meta",
    "register_backend",
    "resolve",
    "resolve_backend_name",
    "transcribe",
    "transcribe_range",
]

"""On-screen text: OCR provider registry plus the provider-agnostic diffing logic."""

from .base import OcrBackend
from .core import added_lines, norm, ocr_frames
from .registry import backend_names, default_backend_name, get_backend, register_backend

__all__ = [
    "OcrBackend",
    "added_lines",
    "backend_names",
    "default_backend_name",
    "get_backend",
    "norm",
    "ocr_frames",
    "register_backend",
]

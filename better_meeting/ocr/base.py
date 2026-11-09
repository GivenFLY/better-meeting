"""The OCR provider contract.

A provider reads the text on one image. Built-in providers are local (macOS
Vision, tesseract); a remote API provider implements the same class and reads
its credentials from the environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar


class OcrBackend(ABC):
    """Base class for every OCR provider."""

    #: Name used on the command line (``--ocr NAME`` / ``bm ocr --backend NAME``).
    name: ClassVar[str]

    @abstractmethod
    def recognize(self, path: str, langs: list[str]) -> str:
        """Text on the image at ``path``, one recognized line per output line.

        ``langs`` are BCP 47 tags such as ``en-US``; providers that use other
        codes convert them.
        """

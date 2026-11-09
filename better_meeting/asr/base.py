"""The ASR provider contract.

A provider turns a 16 kHz mono WAV into timed segments. Built-in providers run
locally (mlx-whisper, faster-whisper, parakeet-mlx); a remote API provider
implements the same class and reads its credentials from the environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, ClassVar, TypedDict


class Segment(TypedDict):
    """One transcript segment as stored in ``transcript.json`` and printed to users."""

    start: float
    end: float
    text: str
    lang: str | None


class RawSegment(Segment):
    """A segment straight from a provider, with the scores used for merging.

    ``score`` is a confidence where higher is better (Whisper's ``avg_logprob``);
    ``nospeech`` is the probability the segment is silence (0.0 if unknown).
    """

    score: float
    nospeech: float


class AsrBackend(ABC):
    """Base class for every speech-recognition provider."""

    #: Name used on the command line (``--asr-backend NAME``) and in cache files.
    name: ClassVar[str]
    #: Model used when ``--asr-model`` is not given.
    default_model: ClassVar[str]
    #: True if the provider decodes every language in one pass and takes no
    #: language hint. The pipeline then runs it once instead of once per
    #: candidate language.
    single_pass: ClassVar[bool] = False
    #: True if ``-O key=value`` decoding options are passed through.
    accepts_options: ClassVar[bool] = True

    def resolve_model(self, model: str | None) -> str:
        """Map the ``--asr-model`` value (``None`` = not given) to a concrete model id."""
        return model or self.default_model

    def check_options(self, opts: dict[str, Any]) -> None:
        """Reject ``-O`` options early, before any audio is extracted."""
        if opts and not self.accepts_options:
            from ..utils import BmError

            raise BmError(f"-O options are not supported by the {self.name!r} ASR backend")

    @abstractmethod
    def transcribe(
        self, wav: Path, lang: str | None, model: str, opts: dict[str, Any]
    ) -> list[RawSegment]:
        """Decode ``wav``.

        ``lang`` is an ISO 639-1 code, or ``None`` to let the provider detect it.
        Every returned segment must carry ``lang`` (the detected one if not given).
        """

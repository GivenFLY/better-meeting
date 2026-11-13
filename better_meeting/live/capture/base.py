"""The capture provider contract.

A provider records one window and delivers its audio while it records. Audio
comes in tracks: ``app`` is what the window's application plays, ``mic`` is the
microphone. The session transcribes each track on its own and writes each to its
own audio stream of the recording.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

#: Track names, in the order of the audio streams of the recording.
TRACKS = ("app", "mic")

#: ``on_audio(track, t, samples)``: mono float32 samples at 16 kHz, ``t`` is the
#: time of the first one in seconds from the start of the capture.
AudioHandler = Callable[[str, float, Any], None]


@dataclass(frozen=True)
class Window:
    """A window that can be captured, or a whole screen (``screen``)."""

    id: int
    app: str
    title: str
    screen: bool = False


class Capture(ABC):
    """Base class for every capture provider."""

    #: Name used on the command line (``--capture NAME``).
    name: ClassVar[str]

    def __init__(self) -> None:
        #: Set by the provider when the capture ends on its own, for example
        #: because the window was closed. The session then stops.
        self.finished = threading.Event()
        #: Width and height in pixels of the part of the video's frame that the
        #: window ever filled, from the top left corner. ``None`` if the whole
        #: frame is picture. The recording is cropped to it.
        self.picture: tuple[int, int] | None = None
        #: The most frames per second to record. Set it before :meth:`start`.
        self.fps = 10

    @abstractmethod
    def windows(self) -> list[Window]:
        """What can be captured: whole screens first, then the windows on screen."""

    @abstractmethod
    def start(
        self,
        window: Window | None,
        tracks: tuple[str, ...],
        video: Path,
        on_audio: AudioHandler,
    ) -> None:
        """Start capturing and return.

        The window's picture is written to ``video`` (no audio in it is needed:
        the session writes the tracks). A window can grow while it is recorded,
        so the frame may be larger than the window; see :attr:`picture`.
        ``window`` is ``None`` when only the microphone is captured; no video is
        written then. ``on_audio`` is called from any thread for every block of
        every track in ``tracks``.
        """

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing and finish ``video``. No ``on_audio`` call comes after it returns."""

"""Tell the user's own speech on the microphone from the application heard through speakers.

Without headphones the microphone also records what the application plays, and
every sentence of the meeting would appear a second time on the ``mic`` track.
The filter compares levels: sound from the speakers reaches the microphone at a
roughly constant fraction of the application's level, a moment later. A person
speaking next to the microphone is louder than that. The fraction is measured
while the session runs, as the median over the frames in which the application
plays; no delay has to be known, because a frame is compared with the loudest
application frame of the last :data:`WINDOW_S` seconds.

This is a decision about whole frames, not echo cancellation: where the user
talks over the application, both stay in the audio.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np

from .segmenter import FRAME_S, RATE, Utterance

#: The speakers' sound arrives and dies away within this many seconds.
WINDOW_S = 0.45
#: The application counts as playing above this level.
ACTIVE = 0.01
#: A frame this many times louder than the usual echo is the user.
FACTOR = 3.5
#: The echo level is trusted after this many seconds of the application playing.
CALIBRATION_S = 3.0
#: An utterance needs this many seconds of the user's speech to be kept.
MIN_USER_S = 0.3
#: Kept on both sides of the user's speech when an utterance is trimmed.
MARGIN_S = 0.3


def _frames(samples: Any, tail: bool = False) -> list[float]:
    """RMS of every FRAME_S of ``samples``; ``tail`` includes a last, shorter frame.

    A capture block can be shorter than a frame, so a block's tail must count.
    """
    n = round(FRAME_S * RATE)
    data = np.asarray(samples, dtype=np.float32)
    stop = len(data) if tail else len(data) // n * n
    return [float(np.sqrt(np.mean(np.square(data[i : i + n])))) for i in range(0, stop, n)]


class EchoFilter:
    """Feed it both tracks as they arrive; ask it which part of a mic utterance is the user."""

    def __init__(self) -> None:
        self._app: dict[int, float] = {}  # frame index on the session clock -> level
        self._ratios: deque[float] = deque(maxlen=round(600 / FRAME_S))
        self._span = round(WINDOW_S / FRAME_S)

    def _reference(self, frame: int) -> float:
        return max(self._app.get(k, 0.0) for k in range(frame - self._span, frame + 1))

    def played(self, t: float, samples: Any) -> None:
        """A block of the application's audio starting at ``t``."""
        first = round(t / FRAME_S)
        for i, level in enumerate(_frames(samples, tail=True)):
            self._app[first + i] = max(self._app.get(first + i, 0.0), level)
        old = first - round(60 / FRAME_S)  # an utterance is far shorter than a minute
        for k in [k for k in self._app if k < old]:
            del self._app[k]

    def heard(self, t: float, samples: Any) -> None:
        """A block of the microphone's audio starting at ``t``."""
        first = round(t / FRAME_S)
        for i, level in enumerate(_frames(samples, tail=True)):
            reference = self._reference(first + i)
            if reference > ACTIVE:
                self._ratios.append(level / reference)

    def user_part(self, utt: Utterance) -> Utterance | None:
        """The part of a microphone utterance the user spoke, ``None`` if it is all echo."""
        if len(self._ratios) * FRAME_S < CALIBRATION_S:
            return utt
        limit = FACTOR * sorted(self._ratios)[len(self._ratios) // 2]
        first = round(utt.start / FRAME_S)
        mine, echo = [], False
        for i, (level, speech) in enumerate(zip(_frames(utt.samples), utt.speech, strict=False)):
            reference = self._reference(first + i)
            if reference > ACTIVE:
                echo = True
            if speech and (reference <= ACTIVE or level > limit * reference):
                mine.append(i)
        if not echo:
            return utt
        if len(mine) * FRAME_S < MIN_USER_S:
            return None
        return utt.cut(
            utt.start + mine[0] * FRAME_S - MARGIN_S,
            utt.start + (mine[-1] + 1) * FRAME_S + MARGIN_S,
        )

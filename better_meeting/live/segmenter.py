"""Cut a live audio stream into utterances at pauses.

Live transcription decodes one utterance at a time with an ordinary
:class:`~better_meeting.asr.AsrBackend`, so the stream has to be cut somewhere.
The cut is made where the speaker pauses: a short chunk decodes fast, and a
provider is not asked to continue a sentence it only saw half of.

Speech is detected by frame energy against a noise floor that follows the
quiet frames. That is enough to find pauses; it does not have to tell speech
from other sounds, because the provider drops what is not speech.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

RATE = 16000
FRAME_S = 0.03
#: A forced cut looks for the quietest frame in this many trailing seconds.
CUT_WINDOW_S = 5.0


@dataclass(frozen=True)
class Utterance:
    """A stretch of audio between two pauses. Times are seconds from the session start."""

    start: float
    end: float
    samples: Any  # numpy float32, mono, RATE
    #: For every FRAME_S of ``samples``: was it speech. Empty if not known.
    speech: tuple[bool, ...] = ()

    def cut(self, start: float, end: float) -> Utterance:
        """The part between ``start`` and ``end``, seconds on the session clock."""
        a = max(0, round((start - self.start) / FRAME_S))
        b = min(len(self.speech), round((end - self.start) / FRAME_S))
        n = round(FRAME_S * RATE)
        return Utterance(
            self.start + a * FRAME_S,
            self.start + b * FRAME_S,
            self.samples[a * n : b * n],
            self.speech[a:b],
        )

    def speech_seconds(self, start: float, end: float) -> float:
        return sum(self.cut(start, end).speech) * FRAME_S


class Segmenter:
    """Feed mono float32 samples at :data:`RATE`, get closed utterances back.

    An utterance closes after ``silence`` seconds without speech. One that reaches
    ``max_len`` seconds is cut at its quietest recent frame and the rest starts the
    next utterance. ``pad`` seconds of audio are kept on both sides of the speech.
    An utterance with less than ``min_speech`` seconds of speech is dropped.
    """

    def __init__(
        self,
        silence: float = 0.6,
        max_len: float = 20.0,
        pad: float = 0.2,
        min_speech: float = 0.25,
        floor: float = 0.004,
    ) -> None:
        import numpy as np

        self._np = np
        self._n = round(FRAME_S * RATE)
        self._silence = max(1, round(silence / FRAME_S))
        self._max = max(self._silence + 1, round(max_len / FRAME_S))
        self._pad = max(1, round(pad / FRAME_S))
        self._min_speech = max(1, round(min_speech / FRAME_S))
        self._floor = floor
        self._noise = 0.0
        self._rest = np.zeros(0, dtype=np.float32)
        self._pos = 0  # frames consumed since the session start
        self._pre: deque[Any] = deque(maxlen=self._pad)
        # The open utterance: frames, their energy, their speech flags.
        self._buf: list[Any] = []
        self._rms: list[float] = []
        self._speech: list[bool] = []
        self._start = 0  # frame index of _buf[0]

    def feed(self, samples: Any, t: float | None = None) -> list[Utterance]:
        """Add samples. ``t`` is the time of the first one, if the source knows it.

        A source that skips silence leaves gaps between its blocks. A gap closes
        the open utterance and moves the clock forward.
        """
        np = self._np
        out: list[Utterance] = []
        if t is not None:
            now = (self._pos * self._n + len(self._rest)) / RATE
            if t - now > 0.1:
                out += self.flush()
                self._pos = round(t / FRAME_S)
                self._rest = np.zeros(0, dtype=np.float32)
                self._pre.clear()
        data = np.concatenate([self._rest, np.asarray(samples, dtype=np.float32)])
        whole = len(data) // self._n * self._n
        self._rest = data[whole:]
        for i in range(0, whole, self._n):
            done = self._frame(data[i : i + self._n])
            if done is not None:
                out.append(done)
        return out

    def flush(self) -> list[Utterance]:
        """Close the open utterance, for the end of the stream."""
        done = self._close(len(self._buf)) if self._buf else None
        return [done] if done is not None else []

    def open(self) -> Utterance | None:
        """The utterance still being spoken, for a draft. ``None`` if there is too little."""
        if sum(self._speech) < self._min_speech:
            return None
        return self._utterance(len(self._buf))

    def _frame(self, frame: Any) -> Utterance | None:
        rms = float(self._np.sqrt(self._np.mean(frame * frame)))
        speech = rms > max(self._floor, 3.0 * self._noise)
        if not speech:
            self._noise = 0.95 * self._noise + 0.05 * rms
        self._pos += 1
        if not self._buf:
            if not speech:
                self._pre.append(frame)
                return None
            self._buf = [*self._pre, frame]
            self._rms = [0.0] * len(self._pre) + [rms]
            self._speech = [False] * len(self._pre) + [True]
            self._start = self._pos - len(self._buf)
            self._pre.clear()
            return None
        self._buf.append(frame)
        self._rms.append(rms)
        self._speech.append(speech)
        last = len(self._speech) - 1 - self._speech[::-1].index(True)
        if len(self._buf) - 1 - last >= self._silence:
            return self._close(min(len(self._buf), last + 1 + self._pad))
        if len(self._buf) >= self._max:
            window = min(len(self._buf) - 1, round(CUT_WINDOW_S / FRAME_S))
            first = len(self._buf) - window
            cut = first + min(range(window), key=lambda k: self._rms[first + k])
            return self._close(max(cut, 1), keep_rest=True)
        return None

    def _utterance(self, n: int) -> Utterance:
        start = self._start * FRAME_S
        return Utterance(
            start,
            start + n * FRAME_S,
            self._np.concatenate(self._buf[:n]),
            tuple(self._speech[:n]),
        )

    def _close(self, n: int, keep_rest: bool = False) -> Utterance | None:
        """Close the open utterance after ``n`` frames. ``keep_rest`` opens the next one
        with the frames after the cut; otherwise they are trailing silence."""
        done = self._utterance(n) if sum(self._speech[:n]) >= self._min_speech else None
        if keep_rest and any(self._speech[n:]):
            self._buf, self._rms, self._speech = self._buf[n:], self._rms[n:], self._speech[n:]
            self._start += n
        else:
            self._pre.extend(self._buf[n:][-self._pad :])
            self._buf, self._rms, self._speech = [], [], []
        return done

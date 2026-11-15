"""Audio helpers shared by the capture providers: resampling and the track clock."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from better_meeting.live.segmenter import RATE

from .base import AudioHandler

#: A pause this long between two blocks of a track is silence the source skipped.
GAP_S = 0.1


class Resampler:
    """Brings one track's blocks to :data:`RATE`, low-pass filtered across block borders.

    The microphone arrives at the device's rate, usually 48 kHz. Taking every third
    sample without a filter folds everything above 8 kHz back into the speech band.
    """

    def __init__(self, rate: float) -> None:
        self._rate = rate
        factor = rate / RATE
        self._factor = round(factor) if factor > 1 and abs(factor - round(factor)) < 1e-6 else 0
        if self._factor:
            n = np.arange(-8 * self._factor, 8 * self._factor + 1)
            taps = np.sinc(n * 0.9 / self._factor) * np.hamming(len(n))
            self._taps = (taps / taps.sum()).astype(np.float32)
            self._tail = np.zeros(len(taps) - 1, dtype=np.float32)
            self._seen = 0  # input samples so far: output samples sit at multiples of the factor

    def __call__(self, data: Any) -> Any:
        if self._rate == RATE or not len(data):
            return data
        if not self._factor:  # an unusual rate: plain interpolation
            n = round(len(data) * RATE / self._rate)
            return np.interp(np.arange(n) * (self._rate / RATE), np.arange(len(data)), data)
        joined = np.concatenate([self._tail, data])
        self._tail = joined[-len(self._tail) :]
        first = -self._seen % self._factor
        self._seen += len(data)
        return np.convolve(joined, self._taps, mode="valid")[first :: self._factor]


def mono(raw: bytes, dtype: str, channels: int) -> Any:
    """Interleaved PCM bytes (``<f4`` or ``<i2``) as mono float32 in -1..1."""
    data = np.frombuffer(raw, dtype=dtype)
    if dtype != "<f4":
        data = data.astype(np.float32) / float(2 ** (8 * data.itemsize - 1))
    if channels > 1:
        frames = len(data) // channels
        data = data[: frames * channels].reshape(frames, channels).mean(axis=1)
    return data.astype(np.float32)


class TrackFeed:
    """One track of a source that has no timestamps of its own.

    Blocks are resampled and placed on the session clock. Blocks that follow each
    other are kept sample to sample; when the source delivered nothing for a while
    (a loopback device is quiet while nothing plays), the next block is placed at
    the time it arrived.
    """

    def __init__(self, track: str, rate: float, on_audio: AudioHandler, t0: float) -> None:
        self._track, self._on_audio, self._t0 = track, on_audio, t0
        self._resample = Resampler(rate)
        self._end: float | None = None  # where the next block goes if it follows the last

    def push(self, data: Any, now: float | None = None) -> None:
        """Deliver ``data`` (mono float32 at the source rate) that ended at ``now``."""
        samples = self._resample(data).astype("float32")
        if not len(samples):
            return
        now = time.monotonic() if now is None else now
        started = max(0.0, now - self._t0 - len(samples) / RATE)
        if self._end is None or started - self._end > GAP_S:
            self._end = started
        t, self._end = self._end, self._end + len(samples) / RATE
        self._on_audio(self._track, t, samples)

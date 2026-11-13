"""A live session: capture -> utterances -> transcript lines while the meeting runs.

Each audio track is cut at pauses (:mod:`.segmenter`) and every utterance is
decoded with an ordinary ASR provider, so any provider works live. While an
utterance is still being spoken it is decoded again about once a second and
shown as a draft; the line is final when the speaker pauses.

The session also writes every track to its own WAV, placed on the session
clock, so the tracks can be muxed into the recording as separate audio streams.
"""

from __future__ import annotations

import queue
import shutil
import signal
import sys
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import numpy as np

from ..asr import AsrBackend
from ..asr.core import is_hallucination, public
from ..utils import log, ts
from .capture import TRACKS, Capture, Window
from .echo import EchoFilter
from .segmenter import RATE, Segmenter, Utterance

#: An open utterance is decoded again for a draft once it grew by this much.
DRAFT_STEP_S = 1.0
#: Speech of this length that no segment covers is decoded a second time.
UNCOVERED_S = 1.5
#: An utterance is amplified to this RMS before decoding, by at most MAX_GAIN.
LEVEL, MAX_GAIN = 0.1, 20.0


def leveled(samples: Any) -> Any:
    """``samples`` at a level the ASR models are used to.

    A microphone is captured without the automatic gain that call applications
    add, and quiet speech makes a multilingual model guess the wrong language.
    """
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64)))) if len(samples) else 0.0
    gain = min(MAX_GAIN, LEVEL / rms) if rms > 0 else 1.0
    return samples if gain <= 1.0 else np.clip(samples * gain, -1.0, 1.0)


@dataclass
class LiveOptions:
    lang: str | None = None  # None = the provider detects it for every utterance
    opts: dict[str, Any] = field(default_factory=dict)
    silence: float = 0.6
    max_len: float = 20.0
    draft: bool = True


class TrackWriter:
    """One track as a 16 kHz mono WAV whose sample ``n`` is at ``n / RATE`` on the session clock."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._wav = wave.open(str(path), "wb")  # noqa: SIM115 (closed in close())
        self._wav.setnchannels(1)
        self._wav.setsampwidth(2)
        self._wav.setframerate(RATE)
        self._written = 0

    def write(self, t: float, samples: Any) -> None:
        gap = round(t * RATE) - self._written
        if gap > RATE // 10:  # the source skipped silence: keep the track on the clock
            self._wav.writeframes(bytes(2 * gap))
            self._written += gap
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
        self._wav.writeframes(pcm.tobytes())
        self._written += len(pcm)

    def close(self) -> None:
        self._wav.close()


def write_wav(path: Path, samples: Any) -> None:
    writer = TrackWriter(path)
    writer.write(0.0, samples)
    writer.close()


class Console:
    """Final lines go to ``out`` (data); the draft is redrawn on ``err`` if it is a terminal."""

    def __init__(self, out: TextIO, err: TextIO, show_track: bool) -> None:
        self._out, self._err = out, err
        self._show_track = show_track
        self._tty = err.isatty()
        self._drafting = False

    def _line(self, start: float, track: str, lang: str | None, text: str) -> str:
        tags = (f" [{track}]" if self._show_track else "") + (f" [{lang}]" if lang else "")
        return f"[{ts(start)}]{tags} {text}"

    def clear(self) -> None:
        if self._drafting:
            self._err.write("\r\x1b[2K")
            self._err.flush()
            self._drafting = False

    def draft(self, start: float, track: str, text: str) -> None:
        if not self._tty or not text:
            return
        width = shutil.get_terminal_size().columns - 1
        line = self._line(start, track, None, text + " …")
        if len(line) > width:  # the end of the utterance is the part that changes
            line = "…" + line[-(width - 1) :]
        self._err.write(f"\r\x1b[2K\x1b[2m{line}\x1b[0m")
        self._err.flush()
        self._drafting = True

    def final(self, seg: dict[str, Any]) -> None:
        self.clear()
        print(self._line(seg["start"], seg["track"], seg["lang"], seg["text"]), file=self._out)
        self._out.flush()


def _stop_on_signals() -> Any:
    """Make SIGINT and SIGTERM end the session like Ctrl+C; returns a function that undoes it.

    A process started in the background has SIGINT ignored, and SIGTERM would end it
    before the recording is written. Only the first signal stops the session: the
    ones after it are answered with a note, so that finishing is not interrupted.
    """
    if threading.current_thread() is not threading.main_thread():
        return lambda: None

    asked = False

    def stop(signum: int, frame: Any) -> None:
        nonlocal asked
        if asked:  # a second Ctrl+C while the last lines are decoded would lose the recording
            log("finishing the recording, one moment")
            return
        asked = True
        raise KeyboardInterrupt

    old = {sig: signal.signal(sig, stop) for sig in (signal.SIGINT, signal.SIGTERM)}
    return lambda: [signal.signal(sig, handler) for sig, handler in old.items()]


class _Session:
    """The state of one live session: queued audio, open utterances, the transcript."""

    def __init__(
        self,
        tracks: tuple[str, ...],
        work: Path,
        backend: AsrBackend,
        model: str,
        options: LiveOptions,
        console: Console,
    ) -> None:
        self._tracks = tracks
        self._work = work
        self._backend, self._model, self._options = backend, model, options
        self._console = console
        self._audio: queue.Queue[tuple[str, float, Any]] = queue.Queue()
        self._segmenters = {t: Segmenter(options.silence, options.max_len) for t in tracks}
        self._writers = {t: TrackWriter(work / f"track_{t}.wav") for t in tracks}
        self._clip = work / "live_utterance.wav"
        self._pending: list[tuple[str, Utterance]] = []  # closed utterances, not decoded yet
        self._drafted = dict.fromkeys(tracks, 0.0)  # length of the open utterance at its draft
        # With both tracks the microphone may hear the application through speakers.
        self._echo = EchoFilter() if len(tracks) > 1 else None
        self.transcript: list[dict[str, Any]] = []

    def run(self, capture: Capture, window: Window | None) -> None:
        """Capture until Ctrl+C or until the capture ends, then decode what is left."""
        try:
            log("loading the ASR model")  # before the capture starts: the first line is not late
            self._decode(Utterance(0.0, 1.0, np.zeros(RATE, dtype=np.float32)))
            capture.start(
                window, self._tracks, self._work / "video.mp4", lambda *b: self._audio.put(b)
            )
            log("listening; Ctrl+C stops the session")
            restore = _stop_on_signals()
            try:
                self._loop(capture)
            except KeyboardInterrupt:
                self._console.clear()
                log("stopping")
            finally:
                capture.stop()
                self._drain()
                restore()
        finally:
            self._console.clear()
            for writer in self._writers.values():
                writer.close()
            self._clip.unlink(missing_ok=True)

    def _loop(self, capture: Capture) -> None:
        while not (capture.finished.is_set() and self._audio.empty() and not self._pending):
            self._take(block=not self._pending)
            if self._pending:
                self._finish(*self._pending.pop(0))
            elif self._options.draft:
                self._draft()

    def _drain(self) -> None:
        """Decode everything that was captured before the capture stopped."""
        while self._take(block=False):
            pass
        for track in self._tracks:
            self._pending.extend((track, u) for u in self._segmenters[track].flush())
        for track, utt in sorted(self._pending, key=lambda p: p[1].start):
            self._finish(track, utt)
        self._pending.clear()

    def _take(self, block: bool) -> bool:
        """Move queued audio into the writers and segmenters. False if nothing was queued."""
        try:
            item = self._audio.get(timeout=0.2) if block else self._audio.get_nowait()
        except queue.Empty:
            return False
        while True:
            track, t, samples = item
            self._writers[track].write(t, samples)
            if self._echo:
                (self._echo.heard if track == "mic" else self._echo.played)(t, samples)
            self._pending.extend((track, u) for u in self._segmenters[track].feed(samples, t))
            try:
                item = self._audio.get_nowait()
            except queue.Empty:
                return True

    def _own(self, track: str, utt: Utterance) -> Utterance | None:
        """``utt`` without the application's echo; ``None`` if nothing else is in it."""
        return self._echo.user_part(utt) if self._echo and track == "mic" else utt

    def _decode(self, utt: Utterance, again: bool = True) -> list[dict[str, Any]]:
        write_wav(self._clip, leveled(utt.samples))
        raw = self._backend.transcribe(
            self._clip, self._options.lang, self._model, self._options.opts
        )
        segs: list[dict[str, Any]] = [
            {**public(s), "start": s["start"] + utt.start, "end": s["end"] + utt.start}
            for s in raw
            if not is_hallucination(s)
        ]
        if not again:
            return segs
        # A provider can skip a whole sentence, for example where the language
        # changes without a pause. A stretch of speech that no segment covers is
        # decoded once more on its own.
        edges: list[float] = [
            utt.start,
            *(t for s in segs for t in (s["start"], s["end"])),
            utt.end,
        ]
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if utt.speech_seconds(a, b) >= UNCOVERED_S:
                segs += self._decode(utt.cut(a - 0.2, b + 0.2), again=False)
        return sorted(segs, key=lambda s: s["start"])

    def _finish(self, track: str, utt: Utterance) -> None:
        """Decode a closed utterance and print its lines."""
        self._drafted[track] = 0.0
        mine = self._own(track, utt)
        for seg in self._decode(mine) if mine else []:
            seg["track"] = track
            self.transcript.append(seg)
            self._console.final(seg)

    def _draft(self) -> None:
        """Show the first open utterance that grew by DRAFT_STEP_S since its last draft."""
        for track in self._tracks:
            utt = self._segmenters[track].open()
            if utt is None:
                self._drafted[track] = 0.0
            elif utt.end - utt.start - self._drafted[track] >= DRAFT_STEP_S:
                self._drafted[track] = utt.end - utt.start
                mine = self._own(track, utt)
                if mine is None:
                    continue
                text = " ".join(s["text"] for s in self._decode(mine, again=False))
                self._console.draft(mine.start, track, text)
                return


def run_session(
    capture: Capture,
    window: Window | None,
    tracks: tuple[str, ...],
    work: Path,
    backend: AsrBackend,
    model: str,
    options: LiveOptions,
    out: TextIO | None = None,
    err: TextIO | None = None,
) -> list[dict[str, Any]]:
    """Capture until Ctrl+C or until the capture ends. Returns the transcript, by start time.

    Segments are the pipeline's segments plus ``track``. ``work`` gets ``video.mp4``
    (if a window is captured) and ``track_<name>.wav`` for every track.
    """
    tracks = tuple(t for t in TRACKS if t in tracks)
    console = Console(out or sys.stdout, err or sys.stderr, show_track=len(tracks) > 1)
    session = _Session(tracks, work, backend, model, options, console)
    session.run(capture, window)
    return sorted(session.transcript, key=lambda s: (s["start"], s["end"]))

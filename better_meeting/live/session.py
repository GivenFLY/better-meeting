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
    import numpy as np

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
        import numpy as np

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
    audio: queue.Queue[tuple[str, float, Any]] = queue.Queue()
    segmenters = {t: Segmenter(options.silence, options.max_len) for t in tracks}
    writers = {t: TrackWriter(work / f"track_{t}.wav") for t in tracks}
    console = Console(out or sys.stdout, err or sys.stderr, show_track=len(tracks) > 1)
    clip = work / "live_utterance.wav"
    pending: list[tuple[str, Utterance]] = []
    drafted = dict.fromkeys(tracks, 0.0)  # length of the open utterance at its last draft
    transcript: list[dict[str, Any]] = []
    # With both tracks the microphone may hear the application through speakers.
    echo = EchoFilter() if len(tracks) > 1 else None

    def own(track: str, utt: Utterance) -> Utterance | None:
        """``utt`` without the application's echo; ``None`` if nothing else is in it."""
        return echo.user_part(utt) if echo and track == "mic" else utt

    def decode(utt: Utterance, again: bool = True) -> list[dict[str, Any]]:
        write_wav(clip, leveled(utt.samples))
        raw = backend.transcribe(clip, options.lang, model, options.opts)
        segs = [
            {**public(s), "start": s["start"] + utt.start, "end": s["end"] + utt.start}
            for s in raw
            if not is_hallucination(s)
        ]
        if not again:
            return segs
        # A provider can skip a whole sentence, for example where the language
        # changes without a pause. A stretch of speech that no segment covers is
        # decoded once more on its own.
        edges = [utt.start, *(t for s in segs for t in (s["start"], s["end"])), utt.end]
        for a, b in zip(edges[::2], edges[1::2], strict=True):
            if utt.speech_seconds(a, b) >= UNCOVERED_S:
                segs += decode(utt.cut(a - 0.2, b + 0.2), again=False)
        return sorted(segs, key=lambda s: s["start"])

    def take(block: bool) -> bool:
        """Move queued audio into the writers and segmenters. False if nothing was queued."""
        try:
            item = audio.get(timeout=0.2) if block else audio.get_nowait()
        except queue.Empty:
            return False
        while True:
            track, t, samples = item
            writers[track].write(t, samples)
            if echo:
                (echo.heard if track == "mic" else echo.played)(t, samples)
            pending.extend((track, u) for u in segmenters[track].feed(samples, t))
            try:
                item = audio.get_nowait()
            except queue.Empty:
                return True

    def finish(track: str, utt: Utterance) -> None:
        drafted[track] = 0.0
        mine = own(track, utt)
        for seg in decode(mine) if mine else []:
            seg["track"] = track
            transcript.append(seg)
            console.final(seg)

    import numpy as np

    log("loading the ASR model")  # before the capture starts, so the first line is not late
    decode(Utterance(0.0, 1.0, np.zeros(RATE, dtype=np.float32)))
    capture.start(window, tracks, work / "video.mp4", lambda *block: audio.put(block))
    log("listening; Ctrl+C stops the session")
    restore = _stop_on_signals()
    try:
        while not (capture.finished.is_set() and audio.empty() and not pending):
            take(block=not pending)
            if pending:
                finish(*pending.pop(0))
                continue
            if not options.draft:
                continue
            for track in tracks:
                utt = segmenters[track].open()
                if utt is None:
                    drafted[track] = 0.0
                elif utt.end - utt.start - drafted[track] >= DRAFT_STEP_S:
                    drafted[track] = utt.end - utt.start
                    mine = own(track, utt)
                    if mine is None:
                        continue
                    text = " ".join(s["text"] for s in decode(mine, again=False))
                    console.draft(mine.start, track, text)
                    break
    except KeyboardInterrupt:
        console.clear()
        log("stopping")
    finally:
        capture.stop()
        while take(block=False):
            pass
        for track in tracks:
            pending.extend((track, u) for u in segmenters[track].flush())
        for track, utt in sorted(pending, key=lambda p: p[1].start):
            finish(track, utt)
        console.clear()
        for writer in writers.values():
            writer.close()
        clip.unlink(missing_ok=True)
        restore()
    return sorted(transcript, key=lambda s: (s["start"], s["end"]))

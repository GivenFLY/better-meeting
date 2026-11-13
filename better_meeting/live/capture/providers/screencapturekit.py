"""Window, application audio and microphone through ScreenCaptureKit (macOS 15+).

One ``SCStream`` does all of it. The window's picture goes to a recording
output that writes the video file. The application's audio and the microphone
arrive as sample buffers, each of its own output type, and are passed on as
separate tracks.

macOS captures the audio of the application that owns the window, not of the
one window. The process needs the Screen & System Audio Recording permission,
and the Microphone permission for the ``mic`` track.
"""

from __future__ import annotations

import ctypes
import threading
from pathlib import Path
from typing import Any, NamedTuple

from better_meeting.live.segmenter import RATE
from better_meeting.utils import BmError, log

from ..base import AudioHandler, Capture, Window

try:
    import CoreMedia as CM  # type: ignore[import-not-found]
    import objc  # type: ignore[import-not-found]
    import ScreenCaptureKit as SC  # type: ignore[import-not-found]
    from Foundation import NSURL, NSObject  # type: ignore[import-not-found]
except ImportError:
    raise BmError(
        'ScreenCaptureKit bindings are not installed: `pip install "better-meeting[live]"`'
    ) from None

WAIT_S = 30.0
START_WAIT_S = 180.0
CORE_GRAPHICS = "/System/Library/Frameworks/CoreGraphics.framework/CoreGraphics"
PERMISSION = (
    "screen capture is not permitted. Allow the terminal that runs `bm` in System Settings "
    "> Privacy & Security > Screen & System Audio Recording, then start it again"
)
FLAG_FLOAT, FLAG_NON_INTERLEAVED = 1, 32  # kAudioFormatFlagIsFloat, ...IsNonInterleaved


def _wait(start: Any, what: str, seconds: float = WAIT_S) -> Any:
    """Call ``start(handler)`` and wait for the handler's ``(value, error)`` or ``(error,)``."""
    done, box = threading.Event(), []

    def handler(*args: Any) -> None:
        box.extend(args)
        done.set()

    start(handler)
    if not done.wait(seconds):
        raise BmError(f"{what}: no answer from ScreenCaptureKit in {seconds:g} s")
    error = box[-1]
    if error is not None:
        if error.code() == -3801:  # SCStreamErrorUserDeclined
            raise BmError(PERMISSION)
        raise BmError(f"{what}: {error.localizedDescription()}")
    return box[0]


class _Format(NamedTuple):
    """AudioStreamBasicDescription, in the order of its fields."""

    mSampleRate: float
    mFormatID: int
    mFormatFlags: int
    mBytesPerPacket: int
    mFramesPerPacket: int
    mBytesPerFrame: int
    mChannelsPerFrame: int
    mBitsPerChannel: int
    mReserved: int


class _Resampler:
    """Brings one track's blocks to :data:`RATE`, low-pass filtered across block borders.

    The microphone arrives at the device's rate, usually 48 kHz. Taking every third
    sample without a filter folds everything above 8 kHz back into the speech band.
    """

    def __init__(self, rate: float) -> None:
        import numpy as np

        self._np = np
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
        np = self._np
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


def _samples(buffer: Any) -> tuple[Any, float]:
    """The audio of a sample buffer as mono float32, and its sample rate."""
    import numpy as np

    desc = CM.CMAudioFormatDescriptionGetStreamBasicDescription(
        CM.CMSampleBufferGetFormatDescription(buffer)
    )
    if not hasattr(desc, "mSampleRate"):  # some pyobjc versions return the plain fields
        desc = _Format(*desc)
    block = CM.CMSampleBufferGetDataBuffer(buffer)
    size = CM.CMBlockBufferGetDataLength(block)
    status, raw = CM.CMBlockBufferCopyDataBytes(block, 0, size, None)
    if status != 0:
        raise BmError(f"cannot read an audio buffer (status {status})")
    channels = int(desc.mChannelsPerFrame) or 1
    if desc.mFormatFlags & FLAG_FLOAT:
        data = np.frombuffer(raw, dtype="<f4")
    else:
        bits = int(desc.mBitsPerChannel)
        data = np.frombuffer(raw, dtype=f"<i{bits // 8}").astype(np.float32) / float(
            2 ** (bits - 1)
        )
    if channels > 1:
        frames = len(data) // channels
        if desc.mFormatFlags & FLAG_NON_INTERLEAVED:
            data = data[: frames * channels].reshape(channels, frames).mean(axis=0)
        else:
            data = data[: frames * channels].reshape(frames, channels).mean(axis=1)
    return data.astype(np.float32), float(desc.mSampleRate)


class _Output(NSObject):
    """Receives the stream's sample buffers and its end."""

    def initWithCapture_(self, capture: Any) -> Any:
        self = objc.super(_Output, self).init()
        if self is not None:
            self.capture = capture
        return self

    def stream_didOutputSampleBuffer_ofType_(self, stream: Any, buffer: Any, kind: int) -> None:
        self.capture._on_buffer(buffer, kind)

    def stream_didStopWithError_(self, stream: Any, error: Any) -> None:
        if error is not None and not self.capture._stopping:
            log(f"the capture ended: {error.localizedDescription()}")
        self.capture.finished.set()

    def recordingOutputDidFinishRecording_(self, output: Any) -> None:
        self.capture._recorded.set()

    def recordingOutput_didFailWithError_(self, output: Any, error: Any) -> None:
        log(f"the video recording failed: {error.localizedDescription()}")
        self.capture._recorded.set()


class ScreenCaptureKitCapture(Capture):
    name = "screencapturekit"

    def __init__(self) -> None:
        super().__init__()
        # A command-line process has no window server connection until something
        # asks for one, and ScreenCaptureKit aborts the process without it.
        ctypes.CDLL(CORE_GRAPHICS).CGMainDisplayID()
        self._stream: Any = None
        self._output: Any = None
        self._on_audio: AudioHandler | None = None
        self._t0: float | None = None
        self._lock = threading.Lock()
        self._recorded = threading.Event()
        self._recording = False
        self._stopping = False
        self._failed = False
        self._frame: tuple[int, int] | None = None  # set while a window is recorded
        self._resamplers: dict[str, _Resampler] = {}

    def _content(self) -> Any:
        return _wait(
            lambda h: (
                SC.SCShareableContent.getShareableContentExcludingDesktopWindows_onScreenWindowsOnly_completionHandler_(
                    True, True, h
                )
            ),
            "listing windows",
        )

    def _pixels(self, display: Any) -> tuple[int, int]:
        """The size of a display in pixels, rounded down to even numbers for the encoder."""
        whole = SC.SCContentFilter.alloc().initWithDisplay_excludingWindows_(display, [])
        scale, size = whole.pointPixelScale(), whole.contentRect().size
        return int(size.width * scale) // 2 * 2, int(size.height * scale) // 2 * 2

    def windows(self) -> list[Window]:
        content = self._content()
        screens = []
        for i, display in enumerate(content.displays(), 1):
            width, height = self._pixels(display)
            title = f"display {i} ({width}×{height})"
            screens.append(Window(int(display.displayID()), "Entire screen", title, screen=True))
        found = [
            Window(int(w.windowID()), str(w.owningApplication().applicationName()), str(w.title()))
            for w in content.windows()
            if w.windowLayer() == 0 and w.title() and w.owningApplication() is not None
        ]
        return screens + sorted(found, key=lambda w: (w.app.casefold(), w.title.casefold()))

    def start(
        self,
        window: Window | None,
        tracks: tuple[str, ...],
        video: Path,
        on_audio: AudioHandler,
    ) -> None:
        content = self._content()
        config = SC.SCStreamConfiguration.alloc().init()
        if window is None:
            # A stream needs something to show even when only the microphone is wanted.
            content_filter = SC.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                content.displays()[0], []
            )
            width, height, rate = 2, 2, 1
        elif window.screen:
            display = next((d for d in content.displays() if d.displayID() == window.id), None)
            if display is None:
                raise BmError(f"the screen is gone: {window.title}")
            content_filter = SC.SCContentFilter.alloc().initWithDisplay_excludingWindows_(
                display, []
            )
            (width, height), rate = self._pixels(display), self.fps
        else:
            target = next((w for w in content.windows() if w.windowID() == window.id), None)
            if target is None:
                raise BmError(f"the window is gone: {window.app} — {window.title}")
            content_filter = SC.SCContentFilter.alloc().initWithDesktopIndependentWindow_(target)
            # The frame cannot change while it is recorded and the window can grow, so
            # the frame is as large as the largest screen. The window sits in its top
            # left corner; `picture` follows how much of the frame it ever filled.
            sizes = [self._pixels(d) for d in content.displays()]
            width, height = max(w for w, _ in sizes), max(h for _, h in sizes)
            rate = self.fps
            self._frame = (width, height)
            self.picture = (2, 2)
        config.setWidth_(width)
        config.setHeight_(height)
        config.setMinimumFrameInterval_(CM.CMTimeMake(1, rate))
        config.setCapturesAudio_("app" in tracks)
        config.setExcludesCurrentProcessAudio_(True)
        config.setSampleRate_(RATE)
        config.setChannelCount_(1)
        config.setCaptureMicrophone_("mic" in tracks)

        self._on_audio = on_audio
        self._output = _Output.alloc().initWithCapture_(self)
        self._stream = SC.SCStream.alloc().initWithFilter_configuration_delegate_(
            content_filter, config, self._output
        )
        kinds = [SC.SCStreamOutputTypeScreen]
        kinds += [SC.SCStreamOutputTypeAudio] if "app" in tracks else []
        kinds += [SC.SCStreamOutputTypeMicrophone] if "mic" in tracks else []
        for kind in kinds:
            ok, error = self._stream.addStreamOutput_type_sampleHandlerQueue_error_(
                self._output, kind, None, None
            )
            if not ok:
                raise BmError(f"cannot add a stream output: {error.localizedDescription()}")
        if window is not None:
            recording = SC.SCRecordingOutputConfiguration.alloc().init()
            recording.setOutputURL_(NSURL.fileURLWithPath_(str(video.resolve())))
            output = SC.SCRecordingOutput.alloc().initWithConfiguration_delegate_(
                recording, self._output
            )
            ok, error = self._stream.addRecordingOutput_error_(output, None)
            if not ok:
                raise BmError(f"cannot record the window: {error.localizedDescription()}")
            self._recording = True
        # The first run waits here for the user to answer the microphone prompt.
        _wait(self._stream.startCaptureWithCompletionHandler_, "starting the capture", START_WAIT_S)

    def _on_buffer(self, buffer: Any, kind: int) -> None:
        """Called on a ScreenCaptureKit queue for every video frame and audio block."""
        if self._failed or not CM.CMSampleBufferIsValid(buffer):
            return
        t = CM.CMTimeGetSeconds(CM.CMSampleBufferGetPresentationTimeStamp(buffer))
        with self._lock:
            if self._t0 is None:
                self._t0 = t  # the first buffer of any kind starts the session clock
        if kind == SC.SCStreamOutputTypeScreen:
            if self._frame is not None:
                self._follow(buffer)
            return
        if self._on_audio is None:
            return
        track = "mic" if kind == SC.SCStreamOutputTypeMicrophone else "app"
        try:
            data, rate = _samples(buffer)
            if track not in self._resamplers:
                self._resamplers[track] = _Resampler(rate)
            samples = self._resamplers[track](data).astype("float32")
        except Exception as e:  # an exception must not escape into the Objective-C caller
            self._failed = True
            log(f"the {track} track stopped: {e}")
            self.finished.set()
            return
        if len(samples):
            self._on_audio(track, max(0.0, t - self._t0), samples)

    def _follow(self, buffer: Any) -> None:
        """Grow :attr:`picture` to the part of the frame this video frame fills."""
        try:
            info = CM.CMSampleBufferGetSampleAttachmentsArray(buffer, False)
            if not info:
                return
            rect = info[0].get(SC.SCStreamFrameInfoContentRect)
            scale = float(info[0].get(SC.SCStreamFrameInfoScaleFactor) or 1.0)
            if rect is None:
                return
            right = (float(rect["X"]) + float(rect["Width"])) * scale
            bottom = (float(rect["Y"]) + float(rect["Height"])) * scale
        except Exception:  # an exception must not escape into the Objective-C caller
            return
        assert self._frame is not None and self.picture is not None
        self.picture = (
            min(self._frame[0], max(self.picture[0], -int(-right // 2) * 2)),
            min(self._frame[1], max(self.picture[1], -int(-bottom // 2) * 2)),
        )

    def stop(self) -> None:
        if self._stream is None:
            return
        self._stopping = True
        try:
            _wait(self._stream.stopCaptureWithCompletionHandler_, "stopping the capture")
        except BmError as e:
            log(str(e))  # already stopped, for example because the window was closed
        if self._recording and not self._recorded.wait(WAIT_S):
            log("the video file may be incomplete: the recording did not finish in time")
        self._on_audio = None
        self._stream = None
        self.finished.set()

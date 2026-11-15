"""Window, application audio and microphone on Windows 10 (version 2004) and later.

Three system APIs do what one ``SCStream`` does on macOS:

- Windows Graphics Capture hands out the picture of a window or of a display
  (``windows-capture``); the frames go to ffmpeg through :class:`VideoPipe`.
- A process loopback gives the audio of the window's application and of its
  child processes, and of nothing else (``proc-tap``). For a whole screen the
  loopback of the default speakers gives everything the computer plays.
- WASAPI gives the microphone (``PyAudioWPatch``).

Windows captures the audio of the application that owns the window, not of the
one window. None of the audio sources has timestamps, so the blocks are placed
on the session clock by the time they arrive (:class:`TrackFeed`).

This provider was written from the documentation of these libraries and has not
run on a Windows machine yet: see "Testing `bm live` on Windows" in
CONTRIBUTING.md.
"""

from __future__ import annotations

import contextlib
import ctypes
import sys
import time
from ctypes import wintypes
from pathlib import Path
from typing import Any

from better_meeting.live.capture.audio import TrackFeed, mono
from better_meeting.live.capture.base import AudioHandler, Capture, Window
from better_meeting.live.capture.video import VideoPipe
from better_meeting.utils import BmError, log

try:
    import pyaudiowpatch as pyaudio  # type: ignore[import-not-found]
    from proctap import ProcessAudioCapture  # type: ignore[import-not-found]
    from windows_capture import WindowsCapture as GraphicsCapture  # type: ignore[import-not-found]
except ImportError as e:
    raise BmError(
        f"the Windows capture needs {e.name}, which is not installed: "
        '`pip install "better-meeting[live]"` (on Windows)'
    ) from None

if sys.platform != "win32":
    raise BmError("the windows capture provider runs on Windows only")

START_WAIT_S = 10.0
#: What proc-tap delivers: interleaved float32 at this rate, this many channels.
TAP_RATE, TAP_CHANNELS = 48000, 2
#: Windows of packaged (Store) applications belong to this process, not to the application.
FRAME_HOST = "applicationframehost"

GW_OWNER, GWL_EXSTYLE, WS_EX_TOOLWINDOW, DWMWA_CLOAKED = 4, -20, 0x80, 14
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STD_ERROR_HANDLE, ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0xFFFFFFF4, 0x0004
DPI_PER_MONITOR_AWARE_V2 = -4

_user32 = ctypes.WinDLL("user32", use_last_error=True)
_kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
_dwmapi = ctypes.WinDLL("dwmapi")
_WindowProc = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
_MonitorProc = ctypes.WINFUNCTYPE(
    wintypes.BOOL, wintypes.HMONITOR, wintypes.HDC, ctypes.POINTER(wintypes.RECT), wintypes.LPARAM
)


def _api(dll: Any, name: str, result: Any, *args: Any) -> Any:
    """A system function with its types declared: a handle is 64 bits wide, an int is not."""
    function = getattr(dll, name)
    function.restype, function.argtypes = result, args
    return function


_EnumWindows = _api(_user32, "EnumWindows", wintypes.BOOL, _WindowProc, wintypes.LPARAM)
_EnumChildWindows = _api(
    _user32, "EnumChildWindows", wintypes.BOOL, wintypes.HWND, _WindowProc, wintypes.LPARAM
)
_IsWindow = _api(_user32, "IsWindow", wintypes.BOOL, wintypes.HWND)
_IsWindowVisible = _api(_user32, "IsWindowVisible", wintypes.BOOL, wintypes.HWND)
_GetWindow = _api(_user32, "GetWindow", wintypes.HWND, wintypes.HWND, wintypes.UINT)
_GetWindowLong = _api(_user32, "GetWindowLongW", wintypes.LONG, wintypes.HWND, ctypes.c_int)
_GetWindowTextLength = _api(_user32, "GetWindowTextLengthW", ctypes.c_int, wintypes.HWND)
_GetWindowText = _api(
    _user32, "GetWindowTextW", ctypes.c_int, wintypes.HWND, wintypes.LPWSTR, ctypes.c_int
)
_GetWindowThreadProcessId = _api(
    _user32, "GetWindowThreadProcessId", wintypes.DWORD, wintypes.HWND, wintypes.LPDWORD
)
_EnumDisplayMonitors = _api(
    _user32,
    "EnumDisplayMonitors",
    wintypes.BOOL,
    wintypes.HDC,
    ctypes.POINTER(wintypes.RECT),
    _MonitorProc,
    wintypes.LPARAM,
)
_DwmGetWindowAttribute = _api(
    _dwmapi,
    "DwmGetWindowAttribute",
    ctypes.c_long,
    wintypes.HWND,
    wintypes.DWORD,
    ctypes.c_void_p,
    wintypes.DWORD,
)
_OpenProcess = _api(
    _kernel32, "OpenProcess", wintypes.HANDLE, wintypes.DWORD, wintypes.BOOL, wintypes.DWORD
)
_QueryFullProcessImageName = _api(
    _kernel32,
    "QueryFullProcessImageNameW",
    wintypes.BOOL,
    wintypes.HANDLE,
    wintypes.DWORD,
    wintypes.LPWSTR,
    wintypes.LPDWORD,
)
_CloseHandle = _api(_kernel32, "CloseHandle", wintypes.BOOL, wintypes.HANDLE)
_GetStdHandle = _api(_kernel32, "GetStdHandle", wintypes.HANDLE, wintypes.DWORD)
_GetConsoleMode = _api(
    _kernel32, "GetConsoleMode", wintypes.BOOL, wintypes.HANDLE, wintypes.LPDWORD
)
_SetConsoleMode = _api(_kernel32, "SetConsoleMode", wintypes.BOOL, wintypes.HANDLE, wintypes.DWORD)


def _prepare_process() -> None:
    """Sizes in real pixels on scaled displays, and a console that redraws the draft line."""
    with contextlib.suppress(AttributeError):  # the function came with Windows 10 version 1703
        _user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(DPI_PER_MONITOR_AWARE_V2))
    console, mode = _GetStdHandle(STD_ERROR_HANDLE), wintypes.DWORD()
    if _GetConsoleMode(console, ctypes.byref(mode)):
        _SetConsoleMode(console, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING)


def _pid(hwnd: int) -> int:
    pid = wintypes.DWORD()
    _GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(pid.value)


def _application(pid: int) -> str:
    """The name of a process's program without ``.exe``; empty if it may not be asked."""
    process = _OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
    if not process:
        return ""
    try:
        path = ctypes.create_unicode_buffer(32768)
        size = wintypes.DWORD(len(path))
        found = _QueryFullProcessImageName(process, 0, path, ctypes.byref(size))
        return Path(path.value).stem if found else ""
    finally:
        _CloseHandle(process)


def _top_windows() -> list[tuple[int, str]]:
    """Handle and title of every window a user would call a window."""
    found: list[tuple[int, str]] = []

    @_WindowProc
    def each(hwnd: int, _: int) -> bool:
        cloaked = wintypes.DWORD()  # set for windows of other virtual desktops and hidden apps
        _DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, ctypes.byref(cloaked), ctypes.sizeof(cloaked))
        length = _GetWindowTextLength(hwnd)
        if (
            _IsWindowVisible(hwnd)
            and not _GetWindow(hwnd, GW_OWNER)
            and not _GetWindowLong(hwnd, GWL_EXSTYLE) & WS_EX_TOOLWINDOW
            and not cloaked.value
            and length
        ):
            title = ctypes.create_unicode_buffer(length + 1)
            _GetWindowText(hwnd, title, length + 1)
            found.append((int(hwnd), title.value))
        return True

    _EnumWindows(each, 0)
    return found


def _displays() -> list[tuple[int, int]]:
    """The size in pixels of every display, rounded down to even numbers for the encoder.

    In the order of ``EnumDisplayMonitors``, which is also the order windows-capture
    counts its ``monitor_index`` in, starting at 1.
    """
    sizes: list[tuple[int, int]] = []

    @_MonitorProc
    def each(monitor: int, dc: int, rect: Any, _: int) -> bool:
        r = rect.contents
        sizes.append(((r.right - r.left) // 2 * 2, (r.bottom - r.top) // 2 * 2))
        return True

    _EnumDisplayMonitors(None, None, each, 0)
    return sizes


def _audio_pid(hwnd: int) -> int:
    """The process that plays the audio of a window: its own, except for packaged applications."""
    pid = _pid(hwnd)
    if _application(pid).casefold() != FRAME_HOST:
        return pid
    inner: list[int] = []

    @_WindowProc
    def each(child: int, _: int) -> bool:
        other = _pid(child)
        if other != pid:
            inner.append(other)
        return other == pid  # False stops the search

    _EnumChildWindows(hwnd, each, 0)
    return inner[0] if inner else pid


class WindowsCapture(Capture):
    name = "windows"

    def __init__(self) -> None:
        super().__init__()
        _prepare_process()
        self._on_audio: AudioHandler | None = None
        self._pipe: VideoPipe | None = None
        self._graphics: Any = None  # the running Windows Graphics Capture session
        self._wasapi: Any = None  # the PyAudio instance, once a stream is needed
        self._streams: list[Any] = []
        self._tap: Any = None

    def windows(self) -> list[Window]:
        screens = [
            Window(i, "Entire screen", f"display {i} ({width}×{height})", screen=True)
            for i, (width, height) in enumerate(_displays(), 1)
        ]
        found = [
            Window(hwnd, _application(_pid(hwnd)) or "unknown", title)
            for hwnd, title in _top_windows()
        ]
        return screens + sorted(found, key=lambda w: (w.app.casefold(), w.title.casefold()))

    def start(
        self,
        window: Window | None,
        tracks: tuple[str, ...],
        video: Path,
        on_audio: AudioHandler,
    ) -> None:
        self._on_audio = on_audio
        try:
            # The first frame of the video starts the session clock.
            t0 = self._record(window, video) if window is not None else time.monotonic()
            if window is not None and "app" in tracks:
                if window.screen:
                    self._listen("app", self._speakers(), t0)
                else:
                    self._tap_application(window, t0)
            if "mic" in tracks:
                self._listen("mic", self._microphone(), t0)
        except BaseException:
            self.stop()
            raise

    def _record(self, window: Window, video: Path) -> float:
        """Start recording ``window`` to ``video``; returns the time of the first frame."""
        displays = _displays()
        if window.screen:
            if not 1 <= window.id <= len(displays):
                raise BmError(f"the screen is gone: {window.title}")
            size, target = displays[window.id - 1], {"monitor_index": window.id}
        else:
            if not _IsWindow(window.id):
                raise BmError(f"the window is gone: {window.app} — {window.title}")
            # The frame cannot change while it is recorded and the window can grow, so
            # the frame is as large as the largest screen. The window sits in its top
            # left corner; `picture` follows how much of the frame it ever filled.
            size = max(w for w, _ in displays), max(h for _, h in displays)
            target = {"window_hwnd": window.id}
            self.picture = (2, 2)
        pipe = self._pipe = VideoPipe(video, size, self.fps)

        def on_frame_arrived(frame: Any, control: Any) -> None:
            """Called on the capture's thread whenever the picture changed."""
            try:
                width, height = pipe.show(frame.frame_buffer)
                if self.picture is not None:
                    self.picture = (
                        max(self.picture[0], -(-width // 2) * 2),
                        max(self.picture[1], -(-height // 2) * 2),
                    )
            except Exception as e:  # an exception must not escape into the native caller
                log(f"the video recording stopped: {e}")
                self.finished.set()

        def on_closed() -> None:
            self.finished.set()  # the window was closed

        try:
            graphics = GraphicsCapture(cursor_capture=True, draw_border=None, **target)
            graphics.event(on_frame_arrived)
            graphics.event(on_closed)
            self._graphics = graphics.start_free_threaded()
        except Exception as e:
            raise BmError(f"cannot capture {window.app} — {window.title}: {e}") from None
        if not pipe.started.wait(START_WAIT_S):
            raise BmError(
                f"no picture from {window.app} — {window.title} in {START_WAIT_S:g} s; "
                "a minimized window cannot be captured"
            )
        return pipe.t0

    def _devices(self) -> tuple[Any, dict[str, Any]]:
        """The PyAudio instance and the description of its WASAPI host."""
        if self._wasapi is None:
            self._wasapi = pyaudio.PyAudio()
        try:
            return self._wasapi, self._wasapi.get_host_api_info_by_type(pyaudio.paWASAPI)
        except OSError as e:
            raise BmError(f"WASAPI is not available: {e}") from None

    def _microphone(self) -> dict[str, Any]:
        audio, host = self._devices()
        if host["defaultInputDevice"] < 0:
            raise BmError("no microphone: Windows has no default recording device")
        return dict(audio.get_device_info_by_index(host["defaultInputDevice"]))

    def _speakers(self) -> dict[str, Any]:
        """The loopback device of the default speakers: everything the computer plays."""
        audio, host = self._devices()
        if host["defaultOutputDevice"] < 0:
            raise BmError("no speakers: Windows has no default playback device")
        speakers = audio.get_device_info_by_index(host["defaultOutputDevice"])
        if speakers.get("isLoopbackDevice"):
            return dict(speakers)
        for device in audio.get_loopback_device_info_generator():
            if speakers["name"] in device["name"]:
                return dict(device)
        raise BmError(f"no loopback device for {speakers['name']}")

    def _emit(self, track: str, t: float, samples: Any) -> None:
        handler = self._on_audio
        if handler is not None:
            handler(track, t, samples)

    def _failed(self, track: str, error: Exception) -> None:
        log(f"the {track} track stopped: {error}")
        self.finished.set()

    def _listen(self, track: str, device: dict[str, Any], t0: float) -> None:
        """Deliver what a WASAPI input (or loopback) device records as ``track``."""
        channels, rate = int(device["maxInputChannels"]), int(device["defaultSampleRate"])
        feed = TrackFeed(track, rate, self._emit, t0)

        def callback(data: bytes, frames: int, when: Any, status: int) -> tuple[None, int]:
            try:
                feed.push(mono(data, "<i2", channels))
            except Exception as e:  # an exception must not escape into the native caller
                self._failed(track, e)
                return None, pyaudio.paAbort
            return None, pyaudio.paContinue

        try:
            stream = self._wasapi.open(
                format=pyaudio.paInt16,
                channels=channels,
                rate=rate,
                frames_per_buffer=rate // 20,
                input=True,
                input_device_index=device["index"],
                stream_callback=callback,
            )
        except OSError as e:
            hint = (
                "; check Settings > Privacy & security > Microphone > "
                "Let desktop apps access your microphone"
                if track == "mic"
                else ""
            )
            raise BmError(f"cannot record from {device['name']}: {e}{hint}") from None
        self._streams.append(stream)

    def _tap_application(self, window: Window, t0: float) -> None:
        """Deliver the audio of the window's application, child processes included, as ``app``."""
        feed = TrackFeed("app", TAP_RATE, self._emit, t0)

        def on_data(pcm: bytes, frames: int) -> None:
            try:
                feed.push(mono(pcm, "<f4", TAP_CHANNELS))
            except Exception as e:  # an exception must not escape into the tap's thread
                self._failed("app", e)

        try:
            self._tap = ProcessAudioCapture(_audio_pid(window.id), on_data=on_data)
            self._tap.start()
        except Exception as e:
            raise BmError(f"cannot capture the audio of {window.app}: {e}") from None

    def stop(self) -> None:
        streams, self._streams = self._streams, []
        for stream in streams:
            try:
                stream.stop_stream()
                stream.close()
            except OSError as e:
                log(f"closing an audio stream: {e}")
        tap, self._tap = self._tap, None
        if tap is not None:
            tap.close()
        wasapi, self._wasapi = self._wasapi, None
        if wasapi is not None:
            wasapi.terminate()
        self._on_audio = None
        graphics, self._graphics = self._graphics, None
        if graphics is not None:
            try:
                graphics.stop()
                graphics.wait()
            except Exception as e:  # already stopped, for example because the window was closed
                log(f"stopping the capture: {e}")
        pipe, self._pipe = self._pipe, None
        if pipe is not None:
            pipe.close()
        self.finished.set()

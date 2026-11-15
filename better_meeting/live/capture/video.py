"""Write the frames a capture provider receives to a video file through ffmpeg.

For providers whose capture API hands out pictures and has no recorder of its
own. The frame of the video has a fixed size; a picture smaller than it sits in
the top left corner on black (see :attr:`Capture.picture`).
"""

from __future__ import annotations

import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from better_meeting.utils import FFMPEG, log, need

#: While the picture does not change, a frame is still written this often.
KEEP_S = 1.0
WAIT_S = 30.0


class VideoPipe:
    """An H.264 file that shows the last picture given to :meth:`show`.

    A frame is written when the picture changed, at most ``fps`` times a second.
    ffmpeg stamps every frame with the time it arrived, so the video keeps the
    session's clock whatever the frame rate turns out to be.
    """

    def __init__(self, path: Path, size: tuple[int, int], fps: int) -> None:
        need("ffmpeg")
        self.size = size
        #: Set once the first frame is written; :attr:`t0` is its ``time.monotonic()``.
        self.started = threading.Event()
        self.t0 = 0.0
        self._step = 1.0 / fps
        self._canvas = np.zeros((size[1], size[0], 4), dtype=np.uint8)
        self._shown = (0, 0)  # width and height of the picture on the canvas
        self._dirty = False
        self._lock = threading.Lock()
        self._stop = threading.Event()
        cmd = [*FFMPEG, "-f", "rawvideo", "-pix_fmt", "bgra", "-video_size", f"{size[0]}x{size[1]}"]
        # The demuxer counts time in 1/framerate: a millisecond is fine enough.
        cmd += ["-framerate", "1000", "-use_wallclock_as_timestamps", "1", "-i", "pipe:0"]
        cmd += ["-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "veryfast", "-crf", "23"]
        cmd += ["-pix_fmt", "yuv420p", str(path)]
        # Ctrl+C in the terminal goes to every process of its group. ffmpeg must not
        # get it: the session stops first and then closes the pipe.
        apart: dict[str, Any] = (
            {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
            if sys.platform == "win32"
            else {"start_new_session": True}
        )
        self._ffmpeg = subprocess.Popen(cmd, stdin=subprocess.PIPE, **apart)
        self._thread = threading.Thread(target=self._run, name="bm-video", daemon=True)
        self._thread.start()

    def show(self, picture: Any) -> tuple[int, int]:
        """Make ``picture`` (height × width × 4, BGRA) the picture of the next frame.

        Returns the width and height it takes on the frame: a picture larger than
        the frame is cut at the right and at the bottom.
        """
        height, width = min(picture.shape[0], self.size[1]), min(picture.shape[1], self.size[0])
        with self._lock:
            old_width, old_height = self._shown
            if height < old_height:  # the picture shrank: black where it was
                self._canvas[height:old_height, :old_width] = 0
            if width < old_width:
                self._canvas[:height, width:old_width] = 0
            self._canvas[:height, :width] = picture[:height, :width]
            self._shown = (width, height)
            self._dirty = True
        return width, height

    def _run(self) -> None:
        assert self._ffmpeg.stdin is not None
        last = 0.0
        while True:
            stopping = self._stop.wait(self._step)
            now = time.monotonic()
            with self._lock:
                seen = self._dirty or self.started.is_set()
                due = self._dirty or stopping or now - last >= KEEP_S
                data = self._canvas.tobytes() if seen and due else None
                self._dirty = False
            if data is not None:
                try:
                    self._ffmpeg.stdin.write(data)
                    self._ffmpeg.stdin.flush()
                except OSError as e:
                    log(f"the video recording failed: {e}")
                    return
                if not self.started.is_set():
                    # ffmpeg stamps a frame when it has read it, and the first write
                    # returns only then: a frame is larger than the pipe's buffer.
                    self.t0 = time.monotonic()
                    self.started.set()
                last = now
            if stopping:
                return

    def close(self) -> None:
        """Write the last picture once more, so the video ends when the session does."""
        self._stop.set()
        self._thread.join(WAIT_S)
        assert self._ffmpeg.stdin is not None
        try:
            self._ffmpeg.stdin.close()
            code = self._ffmpeg.wait(WAIT_S)
        except (OSError, subprocess.TimeoutExpired) as e:
            self._ffmpeg.kill()
            log(f"the video file may be incomplete: {e}")
            return
        if code != 0:
            log(f"the video file may be incomplete: ffmpeg exited with code {code}")

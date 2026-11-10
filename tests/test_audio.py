"""Audio extraction on a generated clip."""

import subprocess
import wave
from array import array

import pytest

from better_meeting.audio import extract_audio

pytestmark = pytest.mark.ffmpeg


def test_extract_audio_mixes_every_audio_stream(tmp_path, ffmpeg):
    """The tone is only in the second stream. ffmpeg alone keeps the silent first one."""
    clip = tmp_path / "two_streams.mov"
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            "anullsrc=sample_rate=16000:channel_layout=mono",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=440:sample_rate=16000",
            "-map",
            "0:a",
            "-map",
            "1:a",
            "-t",
            "1",
            "-c:a",
            "pcm_s16le",
            str(clip),
        ],
        check=True,
    )
    wav = tmp_path / "audio.wav"
    extract_audio(clip, wav)
    with wave.open(str(wav)) as w:
        samples = array("h", w.readframes(w.getnframes()))
    assert max(samples) > 1000  # the tone peaks near 4096 and the mix halves it

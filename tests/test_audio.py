"""Audio extraction on generated clips."""

import subprocess
import wave
from array import array

import pytest

from better_meeting.audio import extract_audio

pytestmark = pytest.mark.ffmpeg


def extract(tmp_path, video_starts=0.0, **span):
    """Samples of ``audio.wav`` from a clip with 2 s of video and a 2 s tone.

    The tone peaks at 1/8 of full scale: 4096. ``span`` goes to ``extract_audio``.
    """
    clip = tmp_path / "clip.mkv"  # Matroska takes PCM audio, so the samples are exact
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
            "sine=frequency=440:sample_rate=16000:duration=2",
            "-itsoffset",
            str(video_starts),
            "-f",
            "lavfi",
            "-i",
            "color=c=black:s=64x64:r=10:d=2",
            "-map",
            "0:a",
            "-map",
            "1:v",
            "-c:a",
            "pcm_s16le",
            "-pix_fmt",
            "yuv420p",
            str(clip),
        ],
        check=True,
    )
    wav = tmp_path / "audio.wav"
    extract_audio(clip, wav, **span)
    with wave.open(str(wav)) as w:
        return array("h", w.readframes(w.getnframes()))


def test_extract_audio_range_keeps_the_audio_before_the_first_video_frame(tmp_path, ffmpeg):
    """The video starts at 1 s and the range is 0.2-0.7 s. With -ss before -i,
    ffmpeg seeks to the first video frame and the range is empty."""
    samples = extract(tmp_path, video_starts=1.0, start=0.2, duration=0.5)
    assert len(samples) == 8000
    assert max(samples) > 3900

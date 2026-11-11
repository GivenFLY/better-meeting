"""Audio extraction on generated clips."""

import subprocess
import wave
from array import array

import pytest

from better_meeting.audio import extract_audio

pytestmark = pytest.mark.ffmpeg

SILENCE = "anullsrc=sample_rate=16000:channel_layout=stereo"
TONE = "sine=frequency=440:sample_rate=16000"  # mono, peaks at 1/8 of full scale: 4096


def extract(tmp_path, *sources, starts=(0.0, 0.0), **span):
    """Samples of ``audio.wav`` from a clip with one 1 s audio stream per lavfi source.

    ``starts`` is when each stream starts in the clip. ``span`` goes to ``extract_audio``.
    """
    clip = tmp_path / "clip.mov"
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for source, at in zip(sources, starts, strict=True):
        cmd += ["-itsoffset", str(at), "-t", "1", "-f", "lavfi", "-i", source]
    for i in range(len(sources)):
        cmd += ["-map", f"{i}:a"]
    subprocess.run([*cmd, "-c:a", "pcm_s16le", str(clip)], check=True)
    wav = tmp_path / "audio.wav"
    extract_audio(clip, wav, **span)
    with wave.open(str(wav)) as w:
        return array("h", w.readframes(w.getnframes()))


def test_extract_audio_mixes_every_audio_stream_at_its_own_level(tmp_path, ffmpeg):
    """The tone is only in the second stream. ffmpeg alone keeps the silent first
    stream, and a mix that divides by the number of streams halves the tone."""
    samples = extract(tmp_path, SILENCE, TONE)
    assert 3900 < max(samples) < 4300


def test_extract_audio_keeps_the_level_of_a_stereo_stream_next_to_a_mono_stream(tmp_path, ffmpeg):
    """The same tone in both channels, next to a silent mono stream. Without the
    conversion of each stream before the mix, the tone is 3 dB too loud."""
    tone = "aevalsrc=0.125*sin(2*PI*440*t)|0.125*sin(2*PI*440*t):sample_rate=16000"
    samples = extract(tmp_path, tone, "anullsrc=sample_rate=16000:channel_layout=mono")
    assert 3900 < max(samples) < 4300


def test_extract_audio_keeps_a_stream_that_starts_late_at_its_time(tmp_path, ffmpeg):
    """The second stream starts 0.5 s into the recording. amix alone mixes it from 0 s."""
    samples = extract(tmp_path, SILENCE, TONE, starts=(0, 0.5))
    first = next(i for i, s in enumerate(samples) if abs(s) > 1000)
    assert 0.5 <= first / 16000 < 0.52


def test_extract_audio_range_ends_on_time_when_a_stream_starts_late(tmp_path, ffmpeg):
    """A 1 s range from 0 s. With -t before -i, the stream that starts 0.5 s late
    runs to 1.5 s."""
    samples = extract(tmp_path, SILENCE, TONE, starts=(0, 0.5), duration=1.0)
    assert len(samples) == 16000


def test_extract_audio_range_keeps_the_audio_before_a_late_first_stream(tmp_path, ffmpeg):
    """The range is 0.2-0.4 s and the first stream starts at 0.5 s. With -ss before
    -i, ffmpeg seeks to the start of that stream and the tone before it is lost."""
    samples = extract(tmp_path, SILENCE, TONE, starts=(0.5, 0), start=0.2, duration=0.2)
    assert max(samples) > 3900


def test_extract_audio_limits_a_mix_that_is_too_loud(tmp_path, ffmpeg):
    """Two streams at 0.8 of full scale sum to 1.6. Without the limiter about a
    quarter of the samples are clipped at full scale."""
    samples = extract(
        tmp_path,
        "aevalsrc=0.8*sin(2*PI*440*t):sample_rate=16000",
        "aevalsrc=0.8*sin(2*PI*554*t):sample_rate=16000",
    )
    clipped = sum(abs(s) >= 32767 for s in samples)
    assert clipped < len(samples) / 20

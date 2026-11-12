"""Audio extraction on generated clips."""

import subprocess
import wave
from array import array

import pytest

from better_meeting.audio import extract_audio

pytestmark = pytest.mark.ffmpeg

SILENCE = "anullsrc=sample_rate=16000:channel_layout=stereo"
TONE = "sine=frequency=440:sample_rate=16000"  # mono, peaks at 1/8 of full scale: 4096


def make_clip(tmp_path, *sources, starts=None, name="clip.mov"):
    """A clip with one 1 s audio stream per lavfi source. ``starts`` is when each stream starts."""
    clip = tmp_path / name
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y"]
    for source, at in zip(sources, starts or [0] * len(sources), strict=True):
        cmd += ["-itsoffset", str(at), "-t", "1", "-f", "lavfi", "-i", source]
    for i in range(len(sources)):
        cmd += ["-map", f"{i}:a"]
    subprocess.run([*cmd, "-c:a", "pcm_s16le", str(clip)], check=True)
    return clip


def extract(clip, **span):
    """Samples of ``audio.wav`` for ``clip``. ``span`` goes to ``extract_audio``."""
    wav = clip.with_name("audio.wav")
    extract_audio(clip, wav, **span)
    with wave.open(str(wav)) as w:
        return array("h", w.readframes(w.getnframes()))


def first_sound(samples):
    """Seconds to the first sample of the tone."""
    return next(i for i, s in enumerate(samples) if abs(s) > 1000) / 16000


def test_extract_audio_mixes_every_audio_stream_at_its_own_level(tmp_path, ffmpeg):
    """The tone is only in the third stream. ffmpeg alone keeps the silent first
    stream, and a mix that divides by the number of streams gives a third of the tone."""
    samples = extract(make_clip(tmp_path, SILENCE, SILENCE, TONE))
    assert 3900 < max(samples) < 4300


def test_extract_audio_keeps_the_level_of_a_stereo_stream_next_to_a_mono_stream(tmp_path, ffmpeg):
    """The same tone in both channels, after a silent mono stream. Without the
    conversion of each stream before the mix, the tone is 3 dB too loud."""
    tone = "aevalsrc=0.125*sin(2*PI*440*t)|0.125*sin(2*PI*440*t):sample_rate=16000"
    clip = make_clip(tmp_path, "anullsrc=sample_rate=16000:channel_layout=mono", tone)
    assert 3900 < max(extract(clip)) < 4300


@pytest.mark.parametrize(
    ("sources", "starts"),
    [((SILENCE, TONE), (0, 0.5)), ((TONE, SILENCE), (0.5, 0))],
    ids=["last stream", "first stream"],
)
def test_extract_audio_keeps_a_stream_that_starts_late_at_its_time(
    tmp_path, ffmpeg, sources, starts
):
    """The tone starts 0.5 s into the recording. amix alone mixes it from 0 s."""
    samples = extract(make_clip(tmp_path, *sources, starts=starts))
    assert 0.5 <= first_sound(samples) < 0.52


def test_extract_audio_counts_start_times_from_the_start_of_the_file(tmp_path, ffmpeg):
    """Both streams start at 0.5 s, so the file starts there and nothing is late."""
    samples = extract(make_clip(tmp_path, SILENCE, TONE, starts=(0.5, 0.5)))
    assert first_sound(samples) < 0.02


def test_extract_audio_range_starts_and_ends_on_time(tmp_path, ffmpeg):
    """The tone starts at 0.5 s and the range is 0.4-0.6 s."""
    clip = make_clip(tmp_path, SILENCE, TONE, starts=(0, 0.5))
    samples = extract(clip, start=0.4, duration=0.2)
    assert len(samples) == 3200
    assert 0.1 <= first_sound(samples) < 0.12


def test_extract_audio_range_ends_on_time_when_a_stream_starts_late(tmp_path, ffmpeg):
    """A 1 s range from 0 s. With -t before -i, the stream that starts 0.5 s late
    runs to 1.5 s."""
    samples = extract(make_clip(tmp_path, SILENCE, TONE, starts=(0, 0.5)), duration=1.0)
    assert len(samples) == 16000


def test_extract_audio_range_keeps_the_audio_before_a_late_first_stream(tmp_path, ffmpeg):
    """The range is 0.2-0.4 s and the first stream starts at 0.5 s. With -ss before
    -i, ffmpeg seeks to the start of that stream and the tone before it is lost."""
    clip = make_clip(tmp_path, SILENCE, TONE, starts=(0.5, 0))
    assert max(extract(clip, start=0.2, duration=0.2)) > 3900


def test_extract_audio_limits_a_mix_that_is_too_loud(tmp_path, ffmpeg):
    """Two streams at 0.8 of full scale sum to 1.6. Without the limiter about a
    quarter of the samples are clipped at full scale."""
    clip = make_clip(
        tmp_path,
        "aevalsrc=0.8*sin(2*PI*440*t):sample_rate=16000",
        "aevalsrc=0.8*sin(2*PI*554*t):sample_rate=16000",
    )
    samples = extract(clip)
    clipped = sum(abs(s) >= 32767 for s in samples)
    assert clipped < len(samples) / 20


def test_extract_audio_does_not_delay_a_file_with_one_stream(tmp_path, ffmpeg):
    """The limiter of the mix delays the audio by 79 samples. A file with one
    stream does not go through the mix."""
    assert first_sound(extract(make_clip(tmp_path, TONE))) < 0.001


def test_extract_audio_uses_one_stream_when_ffmpeg_cannot_make_the_mix(tmp_path, ffmpeg):
    """ffmpeg has no decoder for the second stream: its codec name is changed in the file."""
    silence = "anullsrc=sample_rate=16000:channel_layout=mono"
    clip = make_clip(tmp_path, TONE, silence, name="clip.mkv")
    data = clip.read_bytes()
    codec = data.rindex(b"A_PCM/INT/LIT")
    clip.write_bytes(data[:codec] + b"A_XXX" + data[codec + 5 :])
    assert 3900 < max(extract(clip)) < 4300

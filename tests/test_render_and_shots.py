from PIL import Image

from better_meeting.render import (
    render_screens_index,
    render_stats,
    render_timeline,
    render_transcript,
)
from better_meeting.shots import fallback_shots, select_shots, write_shot

TRANSCRIPT = [
    {"start": 0.0, "end": 4.0, "text": "Привіт усім", "lang": "uk"},
    {"start": 20.0, "end": 24.0, "text": "Let's start", "lang": "en"},
]
SCREEN = [
    {"t": 10.0, "path": "/x.jpg", "added": ["kubectl get pods"]},
    {"t": 30.0, "path": "/y.jpg", "added": []},
]


def test_transcript_tags_languages_only_when_mixed():
    assert "[00:00] [uk] Привіт усім" in render_transcript(TRANSCRIPT)
    mono = render_transcript([{**TRANSCRIPT[0]}])
    assert "[00:00] Привіт усім" in mono and "[uk]" not in mono


def test_timeline_orders_speech_silence_and_screen():
    md = render_timeline(TRANSCRIPT, SCREEN, {10.0: "screens/001_00-00-10.jpg"}, 8.0)
    body = md.split("\n\n", 2)[2]
    expected = [
        "[00:00] SPEECH: [uk] Привіт усім",
        "[00:04] [silence 16s]",
        "[00:10] SCREEN (screenshot: screens/001_00-00-10.jpg):",
        "    kubectl get pods",
        "[00:20] SPEECH: [en] Let's start",
        "[00:30] SCREEN:",
        "    (screen changed, no text recognized)",
    ]
    lines = [line for line in body.splitlines() if line.startswith(("[", "    "))]
    assert lines == expected


def test_stats_language_shares():
    stats = render_stats(TRANSCRIPT, SCREEN, 61, 1)
    assert "- duration: 01:01" in stats
    assert "- speech languages: en 50%, uk 50%" in stats or "uk 50%, en 50%" in stats


def test_screens_index_escapes_pipes():
    md = render_screens_index([("00:10", "screens/001.jpg", "a | b")])
    assert "| 00:10 | `screens/001.jpg` | a \\| b |" in md


def test_select_shots_spreads_over_time():
    screen = [{"t": float(t), "path": "", "added": ["x" * (t % 7 + 1)]} for t in range(100)]
    shots = select_shots(screen, duration=100, max_shots=10)
    assert len(shots) == 10
    assert {int(s["t"] // 10) for s in shots} == set(range(10))  # one per tenth
    assert shots == sorted(shots, key=lambda s: s["t"])


def test_select_shots_under_budget_returns_all_with_text():
    assert select_shots(SCREEN, 60, 30) == [SCREEN[0]]


def test_fallback_shots_even_spacing():
    frames = [{"t": float(i), "path": f"/{i}.jpg"} for i in range(9)]
    assert [s["t"] for s in fallback_shots(frames, 3)] == [0.0, 3.0, 6.0]
    assert fallback_shots(frames, 0) == []


def test_write_shot_resizes_and_labels(tmp_path):
    src = tmp_path / "src.jpg"
    Image.new("RGB", (2000, 1000), (40, 80, 120)).save(src)
    dst = tmp_path / "dst.jpg"
    write_shot(str(src), dst, "00:10", width=1000, quality=70, draw_label=True)
    with Image.open(dst) as im:
        assert im.width == 1000
        assert im.height > 500  # label bar added on top

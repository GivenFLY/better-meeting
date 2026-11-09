import sys

import pytest

from better_meeting.utils import BmError, parse_ts, run, ts, ts_file


@pytest.mark.parametrize(
    ("seconds", "short", "file"),
    [
        (0, "00:00", "00-00-00"),
        (59.6, "01:00", "00-01-00"),
        (75, "01:15", "00-01-15"),
        (3600, "01:00:00", "01-00-00"),
        (3725.2, "01:02:05", "01-02-05"),
    ],
)
def test_timecode_formatting(seconds, short, file):
    assert ts(seconds) == short
    assert ts_file(seconds) == file


@pytest.mark.parametrize(
    ("text", "seconds"),
    [
        ("75", 75.0),
        ("12.5", 12.5),
        ("01:15", 75.0),
        ("1:01:15", 3675.0),
        ("0:00:00", 0.0),
    ],
)
def test_parse_ts(text, seconds):
    assert parse_ts(text) == seconds


@pytest.mark.parametrize("bad", ["", "abc", "1:2:3:4", "1::2", "-5", "12:-1"])
def test_parse_ts_rejects_garbage(bad):
    with pytest.raises(BmError, match="timecode"):
        parse_ts(bad)


def test_run_raises_with_stderr_tail():
    with pytest.raises(BmError, match="boom"):
        run([sys.executable, "-c", "import sys; sys.stderr.write('boom'); sys.exit(3)"])

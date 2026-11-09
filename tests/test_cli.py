import pytest

from better_meeting import __version__
from better_meeting.cli import build_parser, main, normalize_argv, parse_opts
from better_meeting.utils import BmError


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["video.mp4"], ["extract", "video.mp4"]),
        (["video.mp4", "--lang", "uk"], ["extract", "video.mp4", "--lang", "uk"]),
        (["-q", "video.mp4"], ["-q", "extract", "video.mp4"]),
        (["extract", "video.mp4"], ["extract", "video.mp4"]),
        (["ocr", "shot.png"], ["ocr", "shot.png"]),
        (["--version"], ["--version"]),
        ([], []),
    ],
)
def test_normalize_argv(argv, expected):
    assert normalize_argv(argv) == expected


def test_parse_opts_json_and_strings():
    opts = parse_opts(
        ["temperature=0.2", "initial_prompt=release, staging", "word_timestamps=true", "x=a=b"]
    )
    assert opts == {
        "temperature": 0.2,
        "initial_prompt": "release, staging",
        "word_timestamps": True,
        "x": "a=b",
    }


@pytest.mark.parametrize("bad", ["novalue", "=1"])
def test_parse_opts_rejects_malformed(bad):
    with pytest.raises(BmError):
        parse_opts([bad])


def test_extract_defaults():
    a = build_parser().parse_args(["extract", "v.mp4"])
    assert a.lang == "auto"
    assert a.langs == "uk,ru,en"
    assert a.asr_backend == "auto"
    assert a.asr_model is None
    assert a.max_shots == 30


def test_version(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_missing_file_is_a_clean_error(tmp_path, capsys):
    code = main(["frame", str(tmp_path / "nope.mp4"), "-t", "1"])
    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("[bm] error: no such file")


def test_agents_prints_guide(capsys):
    assert main(["agents"]) == 0
    assert "agent guide" in capsys.readouterr().out

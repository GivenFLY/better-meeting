"""Command-line interface: ``bm``.

Subcommands:
  extract     full pipeline: video -> runs/<name>/artifacts/ (``bm video.mp4`` is the same)
  frame       point query: one frame at a timecode
  ocr         point query: OCR a video frame at a timecode, or an existing image
  transcript  point query: transcript of a ``--from``..``--to`` range, with any
              Whisper option passed through (``-O key=value``); cached by parameters
  agents      print AGENTS.md, the guide for an LLM agent driving ``bm``

Point queries print data to stdout and logs to stderr, so an agent can capture
the result directly.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .asr import backend_names as asr_backend_names
from .ocr import backend_names as ocr_backend_names
from .ocr import default_backend_name as default_ocr_backend
from .utils import BmError, parse_ts, setup_logging, ts, ts_file

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tiff", ".bmp"}
COMMANDS = {"extract", "frame", "ocr", "transcript", "agents"}
STAGES = ["audio", "transcribe", "frames", "ocr", "bundle"]
DEFAULT_OCR_LANGS = "en-US,uk-UA"
ASR_MODEL_HELP = (
    "ASR model; default depends on the backend (Whisper: large-v3-turbo, "
    "Parakeet: mlx-community/parakeet-tdt-0.6b-v3). Parakeet takes a "
    "Hugging Face repo id"
)
OPT_HELP = (
    "any Whisper decoding option, repeatable, values parse as JSON "
    "(e.g. -O initial_prompt='project terms' -O temperature=0.2)"
)


def _add_asr_args(p: Any) -> None:
    """ASR options shared by `extract` and `transcript` (parser or argument group)."""
    p.add_argument(
        "--asr-backend",
        default="auto",
        choices=["auto", *asr_backend_names()],
        help="auto = Whisper (mlx on macOS, faster-whisper elsewhere); "
        "parakeet = one pass for all languages (macOS)",
    )
    p.add_argument("--asr-model", default=None, help=ASR_MODEL_HELP)
    p.add_argument("-O", "--opt", action="append", default=[], metavar="KEY=VALUE", help=OPT_HELP)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="bm",
        description="better-meeting: extract transcript, on-screen text and key "
        "screenshots from a meeting recording, as context for an LLM.",
    )
    ap.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    verbosity = ap.add_mutually_exclusive_group()
    verbosity.add_argument("-q", "--quiet", action="store_true", help="only print errors")
    verbosity.add_argument("-v", "--verbose", action="store_true", help="print debug logs")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="COMMAND")

    # extract
    px = sub.add_parser("extract", help="full pipeline: video -> artifacts/")
    px.add_argument("video", type=Path)
    g = px.add_argument_group("output")
    g.add_argument(
        "--out",
        type=Path,
        default=None,
        help="exact working folder (default: <out-root>/<video name>)",
    )
    g.add_argument(
        "--out-root",
        type=Path,
        default=Path("runs"),
        help="root that holds one working folder per video (default: runs)",
    )
    g.add_argument("--force", action="store_true", help="redo every stage, ignore the cache")
    g.add_argument("--only", choices=STAGES, help="stop after this stage")

    g = px.add_argument_group("transcription")
    g.add_argument(
        "--lang",
        default="auto",
        help="pin one language (uk/ru/en/...) for a single pass; auto = one "
        "pass per --langs entry, best segments merged (default: auto)",
    )
    g.add_argument(
        "--langs",
        default="uk,ru,en",
        help="candidate languages for --lang auto, comma-separated (default: uk,ru,en)",
    )
    _add_asr_args(g)
    g.add_argument(
        "--pause",
        type=float,
        default=8.0,
        help="silences at least this long (s) are marked in the timeline (default: 8)",
    )

    g = px.add_argument_group("frames")
    g.add_argument(
        "--frame-interval",
        type=float,
        default=2.0,
        help="thumbnail sampling interval, seconds (default: 2)",
    )
    g.add_argument(
        "--cell-delta",
        type=int,
        default=14,
        help="grayscale change (0-255) that marks a grid cell as changed (default: 14)",
    )
    g.add_argument(
        "--min-cells",
        type=int,
        default=3,
        help="changed cells (of 16x16) needed to keep a frame (default: 3)",
    )
    g.add_argument(
        "--max-gap",
        type=float,
        default=90.0,
        help="keep a frame at least every N seconds (default: 90)",
    )
    g.add_argument(
        "--frame-width",
        type=int,
        default=1920,
        help="width of extracted full frames, px (default: 1920)",
    )

    g = px.add_argument_group("OCR")
    g.add_argument(
        "--ocr",
        default=default_ocr_backend(),
        choices=[*ocr_backend_names(), "none"],
        help="OCR backend (default: vision on macOS, tesseract elsewhere)",
    )
    g.add_argument(
        "--ocr-lang",
        default=DEFAULT_OCR_LANGS,
        help=f"OCR languages as BCP 47 tags (default: {DEFAULT_OCR_LANGS})",
    )
    g.add_argument(
        "--ocr-sim",
        type=float,
        default=0.97,
        help="text similarity above which two frames count as the same screen (default: 0.97)",
    )

    g = px.add_argument_group("screenshots")
    g.add_argument(
        "--max-shots", type=int, default=30, help="screenshots to keep in artifacts/ (default: 30)"
    )
    g.add_argument(
        "--shot-width", type=int, default=1400, help="screenshot width, px (default: 1400)"
    )
    g.add_argument(
        "--shot-quality", type=int, default=72, help="screenshot JPEG quality (default: 72)"
    )
    g.add_argument(
        "--no-label", action="store_true", help="do not stamp the timecode on screenshots"
    )

    # agents
    sub.add_parser("agents", help="print the guide for LLM agents (AGENTS.md)")

    # frame
    pf = sub.add_parser("frame", help="extract one frame at a timecode, print its path")
    pf.add_argument("video", type=Path)
    pf.add_argument("-t", "--at", required=True, help="timecode: SS, MM:SS or HH:MM:SS")
    pf.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="output file (default: <out-root>/<video name>/frames_at/<time>.jpg)",
    )
    pf.add_argument("--out-root", type=Path, default=Path("runs"))
    pf.add_argument("--width", type=int, default=1920, help="frame width, px")

    # transcript
    pt = sub.add_parser("transcript", help="transcript of a time range (cached by parameters)")
    pt.add_argument("video", type=Path)
    pt.add_argument(
        "--from",
        dest="frm",
        default=None,
        help="range start: SS, MM:SS or HH:MM:SS (default: beginning)",
    )
    pt.add_argument("--to", default=None, help="range end (default: end of video)")
    pt.add_argument(
        "--lang",
        default=None,
        help="language (uk/ru/en/...); omitted = detected by the ASR backend",
    )
    _add_asr_args(pt)
    pt.add_argument(
        "--json", action="store_true", help="print raw JSON segments instead of timecoded lines"
    )
    pt.add_argument("--out-root", type=Path, default=Path("runs"))
    pt.add_argument("--force", action="store_true", help="ignore the cache and recompute")

    # ocr
    po = sub.add_parser("ocr", help="OCR a video frame at a timecode, or an existing image")
    po.add_argument("target", type=Path, help="video (needs --at) or image")
    po.add_argument("-t", "--at", default=None, help="timecode: SS, MM:SS or HH:MM:SS")
    po.add_argument("--backend", default=default_ocr_backend(), choices=ocr_backend_names())
    po.add_argument(
        "--lang",
        default=DEFAULT_OCR_LANGS,
        help=f"OCR languages as BCP 47 tags (default: {DEFAULT_OCR_LANGS})",
    )
    po.add_argument("--out-root", type=Path, default=Path("runs"))
    po.add_argument("--width", type=int, default=1920, help="frame width, px")
    return ap


def resolve_out(args: argparse.Namespace) -> Path:
    """Every video gets its own folder: ``--out`` if given, else ``<out-root>/<stem>``."""
    return args.out if args.out is not None else args.out_root / args.video.stem


def parse_opts(pairs: list[str]) -> dict[str, Any]:
    """Repeated ``KEY=VALUE`` -> dict; values parse as JSON, falling back to a string."""
    opts: dict[str, Any] = {}
    for kv in pairs:
        k, sep, v = kv.partition("=")
        if not sep or not k:
            raise BmError(f"expected KEY=VALUE, got {kv!r}")
        try:
            opts[k] = json.loads(v)
        except json.JSONDecodeError:
            opts[k] = v
    return opts


def _require_file(path: Path) -> None:
    if not path.is_file():
        raise BmError(f"no such file: {path}")


def _grab_at(video: Path, at: str, out: Path | None, out_root: Path, width: int) -> Path:
    from .frames import grab_frame

    _require_file(video)
    t = parse_ts(at)
    path = out or out_root / video.stem / "frames_at" / f"{ts_file(t)}.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    grab_frame(video, t, path, width)
    return path


def cmd_extract(a: argparse.Namespace) -> None:
    from .pipeline import run_pipeline

    a.out = resolve_out(a)
    a.asr_opts = parse_opts(a.opt)
    run_pipeline(a)


def cmd_frame(a: argparse.Namespace) -> None:
    print(_grab_at(a.video, a.at, a.out, a.out_root, a.width))


def cmd_agents(a: argparse.Namespace) -> None:
    """AGENTS.md ships inside the wheel; in a source checkout it is read from the repo root."""
    from importlib import resources

    packaged = resources.files(__package__) / "AGENTS.md"
    if packaged.is_file():
        print(packaged.read_text(encoding="utf-8"), end="")
        return
    dev = Path(__file__).resolve().parent.parent / "AGENTS.md"
    if dev.exists():
        print(dev.read_text(encoding="utf-8"), end="")
        return
    raise BmError("AGENTS.md not found in the package or next to the source")


def cmd_transcript(a: argparse.Namespace) -> None:
    from .asr import transcribe_range

    _require_file(a.video)
    start = parse_ts(a.frm) if a.frm else 0.0
    end = parse_ts(a.to) if a.to else None
    if end is not None and end <= start:
        raise BmError(f"--to ({a.to}) must be later than --from ({a.frm or '0'})")
    lang = None if a.lang in (None, "auto") else a.lang
    segments = transcribe_range(
        a.video,
        start,
        end,
        lang,
        a.asr_model,
        a.asr_backend,
        a.out_root / a.video.stem,
        parse_opts(a.opt),
        a.force,
    )
    if a.json:
        print(json.dumps(segments, ensure_ascii=False, indent=2))
    else:
        for s in segments:
            tag = f" [{s['lang']}]" if s.get("lang") else ""
            print(f"[{ts(s['start'])}]{tag} {s['text']}")


def cmd_ocr(a: argparse.Namespace) -> None:
    from .ocr import get_backend

    if a.target.suffix.lower() in IMAGE_EXTS:
        _require_file(a.target)
        path = a.target
    else:
        if a.at is None:
            raise BmError("a video needs a timecode: --at MM:SS")
        path = _grab_at(a.target, a.at, None, a.out_root, a.width)
    print(get_backend(a.backend).recognize(str(path), a.lang.split(",")))


HANDLERS = {
    "extract": cmd_extract,
    "agents": cmd_agents,
    "frame": cmd_frame,
    "transcript": cmd_transcript,
    "ocr": cmd_ocr,
}


def normalize_argv(argv: list[str]) -> list[str]:
    """``bm video.mp4 ...`` is shorthand for ``bm extract video.mp4 ...``.

    Leading global flags (``-q``/``-v``) are kept in front of the inserted command.
    """
    i = 0
    while i < len(argv) and argv[i] in ("-q", "--quiet", "-v", "--verbose"):
        i += 1
    if i < len(argv) and argv[i] not in COMMANDS and not argv[i].startswith("-"):
        return [*argv[:i], "extract", *argv[i:]]
    return argv


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(normalize_argv(sys.argv[1:] if argv is None else argv))
    level = logging.ERROR if args.quiet else logging.DEBUG if args.verbose else logging.INFO
    setup_logging(level)
    try:
        HANDLERS[args.cmd](args)
    except BmError as e:
        print(f"[bm] error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[bm] interrupted", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())

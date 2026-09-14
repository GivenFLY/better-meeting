# better-meeting — agent guide

> This guide is always available as `bm agents` wherever better-meeting is installed.

better-meeting is a **pure extractor**: it pulls everything out of a meeting recording (timestamped speech transcript, on-screen text, key screenshots) into a self-contained folder. It contains **no LLM and never interprets** — *you* are the model that reads the extracted data and answers questions about the meeting ("talks to the video").

## Setup

Requires Python 3.12+ and `ffmpeg`/`ffprobe` on PATH.

```bash
uv tool install git+https://github.com/GivenFLY/better-meeting.git   # global `bm` command
# in a clone: uv sync && uv run bm --help
# macOS: mlx-whisper + Vision OCR out of the box; add the [parakeet] extra for the optional Parakeet engine
# Linux/Windows: install the [faster] extra + tesseract-ocr, run with --ocr tesseract
```

## Extract (the main command)

```bash
bm extract meeting.mp4      # `bm meeting.mp4` works too
```

Output goes to `runs/<video-name>/` — **one folder per video, never shared**. Override the root with `--out-root DIR`, or pin an exact folder with `--out DIR`.

Stages are cached in that folder; re-running the same video is cheap and resumes where it stopped. `--force` redoes everything. `--only audio|transcribe|frames|ocr|bundle` stops early.

`extract` also accepts repeatable `-O key=value` — any Whisper option applied to every transcription pass (e.g. `-O initial_prompt="project terms"` to fix domain vocabulary). Changing `-O`, `--lang`, `--langs`, or the ASR model automatically invalidates the cached transcription — no `--force` needed.

What lands where:

```
runs/<name>/
├── audio.wav, transcript.json, keyframes.json, screen.json   # stage caches
├── pass_uk.json, pass_ru.json, pass_en.json                  # per-language ASR passes
├── thumbs/, frames/                                          # sampled + full frames
└── artifacts/                    <- THE DELIVERABLE, self-contained
    ├── timeline.md               # merged speech + new on-screen text + silences (main file);
    │                             #   header lists stats (duration, language shares)
    ├── transcript.md             # timestamped speech only
    └── screens/ + screens_index.md
```

**Your workflow after extract**: read `artifacts/timeline.md`, look at `screens/*.jpg` yourself if you are multimodal, and answer the user's questions with `[HH:MM:SS]` timecode references. Quote commands, paths, and identifiers exactly as they appear in the materials; if the answer is not in the materials, say so plainly instead of guessing.

## Languages

Meetings may be Ukrainian, Russian, English — or mixed within one recording.

- Default (`--lang auto`): transcribes the **whole recording once per language** in `--langs` (default `uk,ru,en`), then merges — each stretch of the timeline keeps the segments from the pass Whisper was most confident about. Survives per-sentence code-switching. Cost: one full pass per language; each pass cached separately.
- Known language: `--lang uk` → single pass, 3× faster.
- Narrow the candidates: `--langs uk,en`.
- Multilingual transcripts tag every line with `[uk]`/`[ru]`/`[en]`.
- Alternative engine (macOS, needs the `[parakeet]` extra): `--asr-backend parakeet` → NVIDIA Parakeet TDT v3, **one pass for all 25 supported languages** (uk/ru/en included), several times faster than three Whisper passes, punctuation built in. `--langs` is ignored, `-O` is rejected, `--lang` only labels segments; with `--lang auto` the `[uk]`/`[ru]`/`[en]` tags come from a script heuristic, not from the model — treat them as a hint, not ground truth. Works for `bm transcript` too.

## Point queries (drill into a moment)

All print **data to stdout, logs to stderr** — capture stdout programmatically. Timecodes accept `SS`, `MM:SS`, `HH:MM:SS` (same format as in `timeline.md`).

```bash
bm frame meeting.mp4 -t 12:40             # extract one frame; prints its path
bm ocr meeting.mp4 -t 12:40               # OCR a frame at a timecode; prints text
bm ocr path/to/screenshot.jpg             # OCR an existing image (--backend tesseract = second opinion)
bm transcript meeting.mp4 --from 10:00 --to 12:30 --lang en    # re-transcribe a range
bm transcript meeting.mp4 --from 10:00 --to 12:30 --lang uk \
    -O initial_prompt="project terms here" -O temperature=0.2 --json
```

`bm transcript` notes:
- `-O key=value` (repeatable) passes **any** option through to Whisper; values parse as JSON. `initial_prompt` with domain terms is the main lever against mangled names.
- Results cache by the full parameter set under `runs/<name>/transcripts_at/`; identical queries return instantly.
- If the video already went through `extract` with the same model/backend/`-O` options, the range is sliced from the pipeline's cache — no ASR run. Different options or `--force` give a genuinely fresh decode.
- `--json` gives raw segments; default is `[HH:MM:SS] [lang] text` lines.

## Pitfalls

- **Low-resolution recording (≤480p)**: OCR is hopeless — run with `--ocr none`. Screenshots are still selected (evenly across screen changes); read them visually instead.
- **OCR confuses `l/1/I` and `0/O`**: before relying on an exact command/identifier from `timeline.md`, verify against the screenshot (`bm frame` + look, or `bm ocr --backend tesseract` to cross-check).
- Silence markers `[тиша Nс]` in the timeline are neutral — do not assume what happened during them.
- A video with no audio track fails at the `audio` stage with a clear `[bm] ПОМИЛКА:` message — that is expected, not a bug to fix.
- ASR models download from HuggingFace on first use (Whisper large-v3-turbo ≈ 1.6 GB, Parakeet TDT v3 ≈ 2.5 GB) — the first run needs network and time.

## Working on the codebase

- Layout: `better_meeting/cli.py` (entry + subcommands) → `pipeline.py` (stage glue) → one module per stage (`audio`, `asr`, `frames`, `ocr`, `shots`, `render`, `bundle`); `utils.py` helpers.
- Hard rules: no LLM calls inside the tool (extraction only); one output folder per video; point commands keep data on stdout / logs on stderr; every expensive stage must cache and respect `--force`.
- Style: small modules, Ukrainian docstrings/log messages, English identifiers. No test suite — verify with targeted `python -c` checks and `bm --help` smoke runs, as there is no CI.

# better-meeting

[![CI](https://github.com/GivenFLY/better-meeting/actions/workflows/ci.yml/badge.svg)](https://github.com/GivenFLY/better-meeting/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

**Talk to your meeting recordings.** `bm` pulls everything out of a video (a timestamped transcript, the text that appeared on screen, and key screenshots) into one self-contained folder. Give that folder to any LLM or agent as context and ask it anything about the meeting.

```bash
bm meeting.mp4
# -> runs/meeting/artifacts/   timeline.md, transcript.md, screens/
```

It works for any recording: a standup that is all talk, a demo full of terminal commands, a workshop, a plain call.

**There is no LLM inside.** `bm` never summarizes or interprets. It only extracts, and the deterministic local pipeline caches every stage. Understanding the meeting is the job of whatever model *you* connect.

## What you get

`artifacts/timeline.md` interleaves speech, silences and new on-screen text:

```text
# Timeline

- duration: 42:10
- speech languages: uk 71%, en 29%
- speech segments: 612
- screen events: 48
- screenshots in screens/: 30

[12:31] SPEECH: [uk] Давайте подивимось, що з подами на стейджингу.
[12:40] SCREEN (screenshot: screens/007_00-12-40.jpg):
    $ kubectl get pods -n staging
    api-7d9f8c6b5-x2lq   0/1   CrashLoopBackOff   5   3m
[12:44] SPEECH: [en] Okay, so the API is crash-looping again.
[12:58] [silence 14s]
```

Next to it are `transcript.md` (speech only), `screens/*.jpg` (timecode-stamped) and `screens_index.md`.

## Install

You need **Python 3.12+** and **ffmpeg** on your `PATH`.

```bash
# macOS (Apple Silicon): Whisper via MLX and OCR via the Vision framework, out of the box
uv tool install better-meeting

# Linux / Windows: Whisper via faster-whisper; install tesseract-ocr from your package manager
uv tool install "better-meeting[faster]"
```

Add the `[parakeet]` extra for the optional Parakeet engine (macOS). Upgrade with `uv tool upgrade better-meeting`. `pipx install` works the same way.

On first use the ASR models download from Hugging Face: about 1.6 GB for Whisper large-v3-turbo, 2.5 GB for Parakeet TDT v3.

## How it works

The pipeline runs in stages. Each stage caches its result in the video's working folder, so a re-run only redoes what changed:

| # | Stage        | What happens |
|---|--------------|--------------|
| 1 | `audio`      | ffmpeg extracts 16 kHz mono audio |
| 2 | `transcribe` | timed speech recognition with the selected [ASR provider](#providers) |
| 3 | `frames`     | thumbnails are sampled, and full frames are kept only where the screen changed |
| 4 | `ocr`        | each keyframe is read with the selected [OCR provider](#providers); only *new* lines are kept |
| 5 | `bundle`     | `artifacts/` is assembled: timeline, transcript, screenshots |

Each stage records the parameters it was built with. If you change `--lang`, `-O`, `--frame-interval`, `--ocr-lang` and so on, the affected stages and everything downstream re-run automatically. `--force` redoes everything.

If OCR finds no readable text (a low-res recording, or `--ocr none`), screenshots are still picked evenly across the moments the screen changed, so a multimodal model can read them itself.

### Multilingual meetings

Whisper's built-in language detection only listens to the **first 30 seconds**. That is useless for meetings that mix languages, for example a Ukrainian standup with English terms and Russian asides.

With `--lang auto` (the default), `bm` transcribes the whole recording **once per candidate language** (`--langs`, default `uk,ru,en`). It then merges the passes: each stretch of the timeline keeps the segments from the pass Whisper was most confident about (`avg_logprob`). This holds up even when the language changes between sentences. Silence hallucinations are filtered with the standard `no_speech_prob` and `avg_logprob` heuristic.

The cost is one pass per language, and each pass is cached on its own. If you know the language, pin it with `--lang uk` for a single pass. If the meeting is in other languages, set `--langs de,en`.

Mixed-language transcripts tag every line (`[uk]`, `[en]`), and the timeline header shows each language's share.

### Parakeet: one pass for every language

`--asr-backend parakeet` uses NVIDIA's [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) via [parakeet-mlx](https://github.com/senstella/parakeet-mlx). It is a single multilingual model covering 25 European languages that decodes the recording **once**, with punctuation. It is several times faster than three Whisper passes.

- The model takes no language input. `--langs` is ignored and `-O` is rejected. A pinned `--lang` only labels segments. With `--lang auto`, a script heuristic labels each segment `uk`, `ru` or `en`, so treat those tags as a hint.
- macOS only for now. Whisper stays the default.

## Usage

```bash
bm extract VIDEO [options]     # `bm VIDEO` is the same
```

| Command               | What it does |
|-----------------------|--------------|
| `bm extract VIDEO`    | full pipeline → `runs/<name>/artifacts/` |
| `bm frame VIDEO -t T` | one frame at a timecode; prints its path |
| `bm ocr VIDEO -t T`   | OCR of a frame at a timecode (or of an existing image) |
| `bm transcript VIDEO` | transcript of a time range, with any Whisper option |
| `bm agents`           | print the agent guide ([AGENTS.md](AGENTS.md)) |

Common `extract` options (run `bm extract --help` for the full list):

| Option | Default | Meaning |
|--------|---------|---------|
| `--out-root DIR` | `runs` | root holding one folder per video |
| `--out DIR` | auto | exact working folder |
| `--force` | off | ignore every cache |
| `--only STAGE` | | stop after `audio` / `transcribe` / `frames` / `ocr` / `bundle` |
| `--lang CODE` | `auto` | pin one language, or `auto` = pass per `--langs` + merge |
| `--langs L1,L2` | `uk,ru,en` | candidate languages for `--lang auto` |
| `--asr-backend NAME` | `auto` | `mlx`, `faster` or `parakeet`; `auto` = mlx on macOS, faster elsewhere |
| `--asr-model NAME` | per backend | Whisper model name, or a Hugging Face repo id |
| `-O KEY=VALUE` | | any Whisper decoding option, repeatable |
| `--ocr NAME` | `vision` on macOS, else `tesseract` | OCR backend, or `none` |
| `--ocr-lang TAGS` | `en-US,uk-UA` | OCR languages (BCP 47) |
| `--frame-interval SEC` | `2` | thumbnail sampling interval |
| `--max-shots N` | `30` | screenshots kept in `artifacts/` |

Global flags: `-q` (errors only), `-v` (debug), `--version`.

### Examples

```bash
bm standup.mov --lang uk                                 # one known language, single pass
bm call-360p.mp4 --ocr none                              # OCR is hopeless at 360p; keep screenshots
bm demo.mkv -O initial_prompt="Kubernetes, Helm, Argo"   # bias Whisper toward domain terms
bm demo.mkv --asr-backend parakeet                       # fastest multilingual run (macOS)
for f in recordings/*.mp4; do bm "$f"; done              # batch: one folder per file
```

### One folder per video

Every recording gets its own working folder, `runs/<video-name>/` by default. Running the same file again reuses its cache, and a different file lands in a different folder. `--out-root` changes the root, and `--out` pins an exact folder.

```text
runs/standup-2026-08-20/
├── audio.wav, transcript.json, keyframes.json, screen.json, *.meta.json   # stage caches
├── pass_uk.json, pass_en.json, ...                                         # per-language ASR passes
├── thumbs/, frames/
└── artifacts/        <- hand this folder to your LLM
```

## Point queries for agents

Point commands let an agent drill into one moment without re-running the pipeline. Data goes to stdout and logs to stderr:

```bash
bm frame meeting.mp4 -t 12:40                 # -> runs/meeting/frames_at/00-12-40.jpg
bm ocr meeting.mp4 -t 12:40                   # recognized text
bm ocr runs/meeting/artifacts/screens/007_00-12-40.jpg --backend tesseract   # second opinion
bm transcript meeting.mp4 --from 10:00 --to 12:30 --lang en
bm transcript meeting.mp4 --from 10:00 --to 12:30 -O initial_prompt="Argo, Helm" --json
```

Timecodes accept `SS`, `MM:SS` or `HH:MM:SS`. `bm transcript` results are cached by their full parameter set. If `extract` already ran with the same model, backend and options, the range is sliced from its cache and no ASR runs at all.

## Talking to the video

Tell your agent: *"better-meeting is installed as `bm`; run `bm agents` and follow that guide to work with `<video>`"*. The guide covers extraction, what each artifact means, and the point commands. Then ask anything, for example "what did we decide about the release?", "what command did they run at 12:40?" or "summarize each person's status".

## Providers

Speech recognition and OCR are **pluggable providers** behind small interfaces ([`AsrBackend`](better_meeting/asr/base.py), [`OcrBackend`](better_meeting/ocr/base.py)):

| Kind | Provider | Runs | Notes |
|------|----------|------|-------|
| ASR | `mlx` | local, macOS | Whisper on Apple Silicon (default on macOS) |
| ASR | `faster` | local, any OS | Whisper via CTranslate2 (default elsewhere) |
| ASR | `parakeet` | local, macOS | one multilingual pass, no language hint |
| OCR | `vision` | local, macOS | Apple Vision framework (default on macOS) |
| OCR | `tesseract` | local, any OS | tesseract CLI |

Every built-in provider runs locally today, and the interfaces are designed for hosted providers as well. A cloud speech-to-text or vision API is a provider that reads its API key from the environment. The pipeline, caching, language merge and artifact format do not change. See [CONTRIBUTING.md](CONTRIBUTING.md#adding-a-provider) for how to add one.

## Roadmap

- Hosted ASR and OCR providers (API-key based), next to the local ones
- Parakeet on Linux
- Speaker labels (diarization)

Have a use case? [Open an issue](https://github.com/GivenFLY/better-meeting/issues).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Contributions are under Apache 2.0, with a DCO sign-off.

## License

[Apache 2.0](LICENSE). Third-party model attributions are in [NOTICE](NOTICE).

# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[Semantic Versioning](https://semver.org/) (pre-1.0: minor versions may break).

## [Unreleased]

### Added
- `bm live` (macOS 15+, `[live]` extra): pick a window and a source (the window's
  application, the microphone, or both) and read the transcript in the terminal
  while the meeting runs. Each source is cut at pauses and decoded with the regular
  ASR providers; Parakeet is the default if it is installed. After Ctrl+C the
  window's video and one audio stream per source are saved to `runs/<name>.mp4`
  and the usual `artifacts/` folder is built around the live transcript, with the
  frames and OCR of the recording. With two sources every line carries `[app]` or
  `[mic]`, in the terminal and in the artifacts. Quiet audio is amplified before
  decoding, and microphone speech that only repeats the application through the
  speakers is left out of the transcript. A whole screen can be picked instead
  of a window, and a window may be resized while it is recorded. `--fps`,
  `--video-bitrate` and `--scale` set the frame rate and the size of the video.

### Fixed
- `--lang auto` now returns the merged transcript in order of start time. A line
  that another pass's shorter lines interrupted used to come after them, so
  `transcript.md` had lines out of order and `timeline.md` measured some
  `[silence Ns]` markers between the wrong lines.

## [0.4.2] - 2026-11-12

### Fixed
- `--lang auto` no longer prints a sentence twice when another pass's copy of it
  ends less than 1 s later and most of that copy is already covered. It also no
  longer drops a short line that is followed by a more confident line of the same
  pass, or a line of 0.2 s or less. To re-merge an existing working folder, delete
  its `transcript.json`: the cached passes are reused and ASR does not run again.
- A keyframe's full frame now shows the screen that its thumbnail showed. Before,
  the thumbnail was taken up to half a `--frame-interval` after its timestamp and
  the full frame at the timestamp. A screen change between the two was detected,
  but the old screen was read: the new text was missing from the timeline until the
  next change, and the screenshot showed the old screen. Existing working folders
  keep their keyframes and frames: use `--force` to rebuild them.
- `bm transcript` no longer loses the audio of a range that starts before the first
  video frame. ffmpeg was told to seek before it opened the file, and it seeks by
  the video stream. In a recording whose video starts later than its audio, such a
  range came out empty or short.
- `bm frame`, `bm ocr -t` and the frames stage now return the frame that was on
  screen at the timecode. Before, they returned the first frame at or after it. In a
  recording that has no frames while the screen is still, that is the next change.
- A recording with more than one audio stream is now transcribed from all of
  them, mixed. Before, only the stream that ffmpeg selects by default was used,
  so a recording with the microphone and the system audio in separate streams
  lost one side of the conversation. Each stream keeps the level it has alone
  and its own start time, and a limiter turns the sum down where it would clip.
  An existing `audio.wav` is kept: use `--force` to rebuild it.

## [0.4.1] - 2026-11-11

### Added
- Published on PyPI: install with `uv tool install better-meeting`. Releases are
  built and uploaded by GitHub Actions through PyPI Trusted Publishing when a
  version tag is pushed.

## [0.4.0] - 2026-11-09

### Changed
- **Artifacts, logs and `--help` are now in English.** The timeline uses
  `SPEECH:` / `SCREEN` / `[silence 12s]`, and errors print as `[bm] error: ...`.
  Agents or scripts that parsed the old Ukrainian labels need updating.
- ASR and OCR are now pluggable providers (`AsrBackend` / `OcrBackend` with a
  registry) instead of hard-coded branches. Built-in providers are unchanged,
  and this is groundwork for hosted, API-key based providers.
- `--asr-model` defaults to the selected backend's own default model.
- A pinned `--lang` now caches its pass as `pass_<lang>.json`, so `bm transcript`
  with the same language can slice it instead of re-running ASR.
- The first run on an existing working folder rebuilds the frame and OCR caches
  once, because the new per-stage metadata files do not exist yet.

### Added
- `bm --version`, plus `-q/--quiet` and `-v/--verbose`.
- Help text for every `extract` option, grouped by stage.
- Test suite (unit tests plus an ffmpeg end-to-end test with fake providers),
  GitHub Actions CI on Linux and macOS, ruff configuration.
- `CHANGELOG.md`, `SECURITY.md`, issue and pull request templates, Dependabot.

### Fixed
- Changing `--frame-interval`, `--cell-delta`, `--min-cells` or `--max-gap` now
  re-picks keyframes. Previously the stale `keyframes.json` was reused.
- Stale thumbnails from a run with a different interval are removed. Before,
  they were picked up with wrong timestamps.
- Changing `--frame-width` re-extracts frames, and changing `--ocr`, `--ocr-lang`
  or `--ocr-sim` re-runs OCR.
- A tesseract failure is reported instead of silently producing empty text.
- `bm --version` reports the installed package version (it was out of sync with
  `pyproject.toml`).

## [0.3.0] - 2026-09-14

### Added
- Opt-in Parakeet TDT v3 backend (`--asr-backend parakeet`, macOS): one pass for
  all languages.
- `bm transcript`: range re-transcription with Whisper option passthrough and
  caching. It reuses the full pipeline's passes when parameters match.
- `-O key=value` on `extract`, with option-aware caching.
- `bm frame` and `bm ocr` point queries, and `bm agents`.
- Multilingual mode: a full pass per candidate language with a best-confidence merge.

### Removed
- The manual chat workflow (artifact prompts, screenshot sheets).

## [0.2.0] - 2026-08-20

- Initial public release.

[Unreleased]: https://github.com/GivenFLY/better-meeting/compare/v0.4.2...HEAD
[0.4.2]: https://github.com/GivenFLY/better-meeting/compare/v0.4.1...v0.4.2
[0.4.1]: https://github.com/GivenFLY/better-meeting/compare/v0.4.0...v0.4.1
[0.4.0]: https://github.com/GivenFLY/better-meeting/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/GivenFLY/better-meeting/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/GivenFLY/better-meeting/releases/tag/v0.2.0

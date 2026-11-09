"""better-meeting: extract everything from a meeting recording (speech + screen)
into a self-contained ``artifacts/`` folder that an LLM reads as full context.

There is no language model inside: the package only extracts.

Modules, in pipeline order:

  audio      1. ffmpeg -> 16 kHz mono WAV
  asr        2. pluggable speech-recognition providers -> timed segments
  frames     3. thumbnails -> grid-diff keyframes -> full-size frames
  ocr        4. pluggable OCR providers -> text + diff of new on-screen lines
  bundle     5. assemble artifacts/ (with render: markdown, shots: screenshots)
  pipeline   stage glue with per-stage caching in the video's working folder
  cli        argument parsing and ``main()``
  utils      timecodes, logging, subprocess, JSON helpers
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("better-meeting")
except PackageNotFoundError:  # running from a source tree that is not installed
    __version__ = "0.0.0+unknown"

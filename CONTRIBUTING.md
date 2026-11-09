# Contributing

Thanks for helping. Bug reports, provider integrations and fixes are all welcome.

## Development setup

```bash
git clone https://github.com/GivenFLY/better-meeting.git
cd better-meeting
uv sync                       # add --all-extras for faster-whisper and parakeet
uv run bm --help
```

You also need `ffmpeg` on `PATH`. The end-to-end tests are skipped without it.

## Checks

These are the same checks CI runs on every pull request:

```bash
uv run ruff check .
uv run ruff format --check .  # `uv run ruff format .` to fix
uv run pytest
```

The tests never download models: they use fake ASR/OCR providers registered in
[`tests/conftest.py`](tests/conftest.py). A change that only works with a real model
should still come with a test against a fake provider. Also describe how you checked
it with the real model in the pull request.

## Ground rules

- **No LLM inside the tool.** `bm` extracts. It does not summarize, classify or interpret.
- **One output folder per video.** Never share or overwrite another video's folder.
- **stdout is data, stderr is logs** for point commands (`frame`, `ocr`, `transcript`).
- **Every expensive stage caches** and invalidates itself when its parameters change. `--force` always recomputes.
- Code, comments, log messages and artifact text are in English. Transcripts and OCR keep the speaker's language, of course.

## Adding a provider

ASR and OCR providers are small classes registered by name. The pipeline, caching,
the multi-language merge and the artifact format all stay as they are.

**ASR**: subclass `better_meeting.asr.AsrBackend`:

```python
from better_meeting.asr import AsrBackend, RawSegment, register_backend

class AcmeSpeechBackend(AsrBackend):
    name = "acme"                  # --asr-backend acme
    default_model = "acme-large"
    single_pass = False            # True if one call handles every language
    accepts_options = True         # whether -O key=value is passed through

    def transcribe(self, wav, lang, model, opts) -> list[RawSegment]:
        api_key = os.environ["ACME_API_KEY"]   # credentials come from the environment
        ...
        return [{"start": 0.0, "end": 2.5, "text": "...", "lang": lang or detected,
                 "score": avg_logprob_or_confidence, "nospeech": 0.0}]
```

**OCR**: subclass `better_meeting.ocr.OcrBackend` and implement `recognize(path, langs) -> str`.

Register a built-in provider in the `_FACTORIES` dict of `asr/registry.py` or
`ocr/registry.py`, using a `"module:Class"` string. Its module, and any SDK it
imports, then loads only when someone selects it. Put the SDK in an optional extra in
`pyproject.toml`, not in the core dependencies.

For hosted providers:
- Read keys from environment variables (`ACME_API_KEY`). Never take them from flags,
  and never write them to cache files or logs.
- If the key is missing, raise `BmError` with a message that names the variable.
- `score` drives the multi-language merge (higher = more confident). If the API
  gives no confidence, return a constant and set `single_pass = True` when the
  service detects languages itself.
- Document in the README what data leaves the machine.

## Commit sign-off (DCO)

Every commit must be signed off:

```bash
git commit -s
```

This adds a `Signed-off-by: Your Name <you@example.com>` line, which certifies the
[Developer Certificate of Origin](https://developercertificate.org/): you wrote the
contribution yourself, or otherwise have the right to submit it under the project's
license. In particular, **do not submit code that belongs to your employer or a third
party** unless you are explicitly authorized to license it under Apache 2.0.

Per section 5 of the license, every contribution is licensed to the project under
Apache 2.0, and its author keeps the copyright. Pull requests without a sign-off
will not be merged.

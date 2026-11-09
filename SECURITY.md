# Security policy

## Reporting a vulnerability

Please do not open a public issue. Email the maintainer at the author address
listed in `pyproject.toml` instead. Expect a first reply within a week.

## Scope and data handling

- All built-in providers run locally, so recordings, transcripts and screenshots
  stay on your machine. The only network access is the one-time model download
  from Hugging Face.
- Hosted providers, once added, will read API keys from environment variables
  only. Keys must never be written to cache files, artifacts or logs. A provider
  that sends data off the machine must say so in its documentation.
- Working folders (`runs/` by default) contain the full transcript and frames of
  the meeting. Treat them as you would the recording itself.

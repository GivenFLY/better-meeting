"""OCR with the tesseract CLI (local, any platform)."""

from __future__ import annotations

from better_meeting.utils import need, run

from ..base import OcrBackend

# BCP 47 primary subtag -> tesseract traineddata name, for the common cases.
_TESS_LANG = {
    "en": "eng",
    "uk": "ukr",
    "ru": "rus",
    "de": "deu",
    "fr": "fra",
    "es": "spa",
    "it": "ita",
    "pl": "pol",
    "pt": "por",
}


def tesseract_langs(langs: list[str]) -> str:
    """``['en-US', 'uk-UA']`` -> ``'eng+ukr'``; unknown codes pass through unchanged."""
    codes = []
    for tag in langs:
        primary = tag.split("-")[0].lower()
        code = _TESS_LANG.get(primary, primary)
        if code not in codes:
            codes.append(code)
    return "+".join(codes)


class TesseractOcrBackend(OcrBackend):
    name = "tesseract"

    def recognize(self, path: str, langs: list[str]) -> str:
        need("tesseract")
        res = run(["tesseract", path, "stdout", "-l", tesseract_langs(langs), "--psm", "6"])
        return res.stdout

"""OCR with the macOS Vision framework (local, macOS)."""

from __future__ import annotations

from ..utils import BmError
from .base import OcrBackend


class VisionOcrBackend(OcrBackend):
    name = "vision"

    def recognize(self, path: str, langs: list[str]) -> str:
        try:
            import Vision  # type: ignore[import-not-found]
            from Foundation import NSURL  # type: ignore[import-not-found]
        except ImportError:
            raise BmError(
                "pyobjc Vision bindings are not installed: "
                "`pip install pyobjc-framework-Vision`, or use --ocr tesseract"
            ) from None
        url = NSURL.fileURLWithPath_(path)
        handler = Vision.VNImageRequestHandler.alloc().initWithURL_options_(url, None)
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(0)  # accurate
        req.setUsesLanguageCorrection_(False)  # never "correct" commands and identifiers
        req.setRecognitionLanguages_(langs)
        res = handler.performRequests_error_([req], None)
        ok, err = res if isinstance(res, tuple) else (res, None)
        if not ok:
            raise BmError(f"Vision OCR failed on {path}: {err}")
        lines = []
        for obs in req.results() or []:
            cand = obs.topCandidates_(1)
            if cand:
                lines.append(str(cand[0].string()))
        return "\n".join(lines)

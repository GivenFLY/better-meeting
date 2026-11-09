"""Shared fixtures: fake ASR/OCR providers registered through the public registry API."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, ClassVar

import pytest

from better_meeting import asr, ocr
from better_meeting.asr.base import AsrBackend, RawSegment
from better_meeting.ocr.base import OcrBackend


class FakeAsr(AsrBackend):
    """Per-language provider: the 'uk' pass is confident on [0, 5), 'en' on [5, 10)."""

    name = "fake"
    default_model = "fake-model"

    def __init__(self) -> None:
        self.calls: list[tuple[str | None, dict[str, Any]]] = []

    def transcribe(
        self, wav: Path, lang: str | None, model: str, opts: dict[str, Any]
    ) -> list[RawSegment]:
        self.calls.append((lang, dict(opts)))
        good = {"uk": (0.0, 5.0), "en": (5.0, 10.0)}
        out: list[RawSegment] = []
        for start, end in ((0.0, 5.0), (5.0, 10.0)):
            confident = good.get(lang or "", (-1, -1)) == (start, end)
            out.append(
                {
                    "start": start,
                    "end": end,
                    "text": f"{lang} {start:g}",
                    "lang": lang or "en",
                    "score": -0.1 if confident else -0.9,
                    "nospeech": 0.0,
                }
            )
        return out


class FakeSinglePass(AsrBackend):
    name = "fake1"
    default_model = "fake1-model"
    single_pass = True
    accepts_options = False

    def __init__(self) -> None:
        self.calls = 0

    def transcribe(
        self, wav: Path, lang: str | None, model: str, opts: dict[str, Any]
    ) -> list[RawSegment]:
        self.calls += 1
        return [
            {
                "start": 0.0,
                "end": 2.0,
                "text": "hello there",
                "lang": lang or "en",
                "score": 0.9,
                "nospeech": 0.0,
            },
            {
                "start": 2.0,
                "end": 4.0,
                "text": "це тест",
                "lang": lang or "uk",
                "score": 0.9,
                "nospeech": 0.0,
            },
        ]


class FakeOcr(OcrBackend):
    """Returns text keyed by frame index, so tests control what is 'on screen'."""

    name = "fakeocr"
    screens: ClassVar[list[str]] = []

    def __init__(self) -> None:
        self.seen: list[str] = []

    def recognize(self, path: str, langs: list[str]) -> str:
        self.seen.append(path)
        idx = len(self.seen) - 1
        return self.screens[idx] if idx < len(self.screens) else ""


@pytest.fixture
def fake_asr() -> FakeAsr:
    backend = FakeAsr()
    asr.register_backend("fake", lambda: backend)
    return backend


@pytest.fixture
def fake_single() -> FakeSinglePass:
    backend = FakeSinglePass()
    asr.register_backend("fake1", lambda: backend)
    return backend


@pytest.fixture
def fake_ocr() -> FakeOcr:
    backend = FakeOcr()
    ocr.register_backend("fakeocr", lambda: backend)
    return backend


@pytest.fixture
def ffmpeg() -> None:
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        pytest.skip("ffmpeg/ffprobe not on PATH")

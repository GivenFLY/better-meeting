from better_meeting.ocr import added_lines, ocr_frames
from better_meeting.ocr.providers.tesseract import tesseract_langs


def test_added_lines_ignores_whitespace_and_old_lines():
    old = "$ git status\nOn branch main"
    new = "$ git   status\nOn branch main\n$ git push\nx"
    assert added_lines(new, old) == ["$ git push"]  # 'x' is too short to count


def test_tesseract_lang_mapping():
    assert tesseract_langs(["en-US", "uk-UA", "en-GB"]) == "eng+ukr"
    assert tesseract_langs(["jpn"]) == "jpn"


def test_ocr_frames_skips_same_screen(fake_ocr):
    fake_ocr.screens = [
        "Agenda\nRelease plan",
        "Agenda\nRelease plan",  # identical: skipped
        "Agenda\nRelease plan\nRollback",  # one new line
    ]
    frames = [{"t": float(i), "path": f"/f{i}.jpg"} for i in range(3)]
    events = ocr_frames(frames, "fakeocr", ["en-US"], sim_threshold=0.97)
    assert [(e["t"], e["added"]) for e in events] == [
        (0.0, ["Agenda", "Release plan"]),
        (2.0, ["Rollback"]),
    ]

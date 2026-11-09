"""Render the text artifacts: ``transcript.md`` and ``timeline.md``."""

from __future__ import annotations

from typing import Any

from .utils import ts

MAX_SCREEN_LINES = 40  # new on-screen lines shown per screen event in the timeline


def is_multilang(transcript: list[dict[str, Any]]) -> bool:
    return len({s.get("lang") for s in transcript if s.get("lang")}) > 1


def render_transcript(transcript: list[dict[str, Any]]) -> str:
    multi = is_multilang(transcript)
    lines = ["# Transcript", ""]
    if multi:
        lines += ["Multilingual recording: every line carries its language code.", ""]
    for seg in transcript:
        tag = f" [{seg['lang']}]" if multi and seg.get("lang") else ""
        lines.append(f"[{ts(seg['start'])}]{tag} {seg['text']}")
    return "\n".join(lines) + "\n"


def render_timeline(
    transcript: list[dict[str, Any]],
    screen: list[dict[str, Any]],
    shots_by_t: dict[float, str],
    pause_threshold: float,
    stats: str = "",
) -> str:
    """Speech, silences of at least ``pause_threshold`` seconds and screen events, by time."""
    multi = is_multilang(transcript)
    items: list[tuple[float, str, Any]] = []
    prev_end = None
    for seg in transcript:
        if prev_end is not None and seg["start"] - prev_end >= pause_threshold:
            items.append((prev_end, "PAUSE", f"[silence {int(seg['start'] - prev_end)}s]"))
        tag = f"[{seg['lang']}] " if multi and seg.get("lang") else ""
        items.append((seg["start"], "SPEECH", tag + seg["text"]))
        prev_end = seg["end"]

    for ev in screen:
        if ev["added"]:
            body = "\n".join("    " + line for line in ev["added"][:MAX_SCREEN_LINES])
        else:
            body = "    (screen changed, no text recognized)"
        shot = shots_by_t.get(round(ev["t"], 3))
        items.append((ev["t"], "SCREEN", (body, shot)))

    items.sort(key=lambda x: x[0])
    out = ["# Timeline", ""]
    if stats:
        out += [stats, ""]
    out += [
        "SPEECH is the transcript. SCREEN lists only lines that newly appeared on screen (OCR).",
        "A screenshot link is given where that moment is saved in screens/.",
        "",
    ]
    for t, kind, body in items:
        if kind == "SPEECH":
            out.append(f"[{ts(t)}] SPEECH: {body}")
        elif kind == "PAUSE":
            out.append(f"[{ts(t)}] {body}")
        else:
            text, shot = body
            head = f"[{ts(t)}] SCREEN"
            if shot:
                head += f" (screenshot: {shot})"
            out.append(f"{head}:\n{text}")
    return "\n".join(out) + "\n"


def render_stats(
    transcript: list[dict[str, Any]], screen: list[dict[str, Any]], duration: float, n_shots: int
) -> str:
    """The stats block at the top of ``timeline.md``."""
    by_lang: dict[str, float] = {}
    for s in transcript:
        if s.get("lang"):
            by_lang[s["lang"]] = by_lang.get(s["lang"], 0.0) + (s["end"] - s["start"])
    total = sum(by_lang.values())
    langs = (
        ", ".join(
            f"{lang} {round(100 * v / total)}%"
            for lang, v in sorted(by_lang.items(), key=lambda kv: -kv[1])
        )
        if total
        else "unknown"
    )
    return "\n".join(
        [
            f"- duration: {ts(duration)}",
            f"- speech languages: {langs}",
            f"- speech segments: {len(transcript)}",
            f"- screen events: {len(screen)}",
            f"- screenshots in screens/: {n_shots}",
        ]
    )


def render_screens_index(rows: list[tuple[str, str, str]]) -> str:
    """``rows`` of ``(timecode, file, preview)`` -> ``screens_index.md``."""
    out = ["# Screenshots", "", "| Time | File | New on-screen text |", "|---|---|---|"]
    out += [f"| {t} | `{f}` | {p.replace('|', '\\|')} |" for t, f, p in rows]
    return "\n".join(out) + "\n"

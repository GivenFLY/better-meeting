"""Script-based language guess for providers that do not report a language.

Covers the uk/ru/en mix the tool was built for: Latin -> ``en``; Cyrillic is
split into ``uk``/``ru`` by marker letters and language-specific function words.
It is a heuristic: treat its tags as a hint, not ground truth.
"""

from __future__ import annotations

import re
from collections.abc import MutableSequence
from typing import Any

_UK_LETTERS = set("іїєґ")
_RU_LETTERS = set("ыэъё")
# Function words that exist in only one of the two languages
# (shared ones such as «не», «на», «так» are left out on purpose).
_UK_WORDS = set(
    (
        "і й що це як був була було ще вже або коли де хто вони ми ви він вона воно з із "
        "зі від під але цей ця його її їх дуже зараз потім треба потрібно добре дякую "
        "ласка гаразд згоден питання вчора"
    ).split()
)
_RU_WORDS = set(
    (
        "и что это как был была было есть нет ещё еще уже только если чтобы или когда "
        "где кто они мы вы он она оно из с к от под его её ее их очень сейчас потом надо "
        "нужно хорошо спасибо пожалуйста привет ладно согласен понятно вопрос сегодня "
        "вчера неделя встреча"
    ).split()
)
_CYRILLIC_WORD = re.compile("[\u0400-\u04ff']+")  # Cyrillic block


def guess_lang(text: str, prev: str | None) -> str | None:
    """Guess ``en``/``uk``/``ru`` for one segment.

    Short Cyrillic replies with no markers inherit ``prev`` (if it is uk/ru),
    otherwise default to ``uk``. Text with no letters at all returns ``prev``.
    """
    low = text.lower()
    if not any("Ѐ" <= c <= "ӿ" for c in low):
        return "en" if any(c.isalpha() for c in low) else prev
    words = _CYRILLIC_WORD.findall(low)
    uk = sum(c in _UK_LETTERS for c in low) + sum(w in _UK_WORDS for w in words)
    ru = sum(c in _RU_LETTERS for c in low) + sum(w in _RU_WORDS for w in words)
    if uk != ru:
        return "uk" if uk > ru else "ru"
    return prev if prev in ("uk", "ru") else "uk"


def label_segments(segments: MutableSequence[Any]) -> None:
    """Set ``lang`` on every segment in place, carrying context from the previous one."""
    prev = None
    for s in segments:
        s["lang"] = prev = guess_lang(s["text"], prev)

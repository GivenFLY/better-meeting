"""Print the CHANGELOG.md section of one version as GitHub release notes.

Usage: release_notes.py v0.5.0 [CHANGELOG.md]

GitHub renders a line break inside release notes as a line break on the page,
so the hard-wrapped lines of a changelog entry are joined back into one.
"""

import re
import sys
from pathlib import Path


def release_notes(changelog: str, version: str) -> str:
    """Return the body of the `## [version]` section, one line per paragraph or list item."""
    out: list[str] = []
    found = False
    link = ""
    for line in changelog.splitlines():
        if line.startswith("## ["):
            if found:
                found = False
                continue
            found = line.startswith(f"## [{version}]")
            continue
        if m := re.match(r"\[([^\]]+)\]: (\S+)", line):
            found = False
            if m.group(1) == version:
                link = m.group(2)
            continue
        if not found:
            continue
        text = line.strip()
        continues = out and out[-1] and text and not text.startswith(("- ", "#"))
        if continues and not out[-1].startswith("#"):
            out[-1] += " " + text
        else:
            out.append(line.rstrip())
    notes = "\n".join(out).strip()
    if not notes:
        raise SystemExit(f"CHANGELOG.md has no entry for {version}")
    if "/compare/" in link:
        notes += f"\n\n**Full changelog:** {link}"
    return notes


if __name__ == "__main__":
    path = Path(sys.argv[2] if len(sys.argv) > 2 else "CHANGELOG.md")
    print(release_notes(path.read_text(encoding="utf-8"), sys.argv[1].removeprefix("v")))

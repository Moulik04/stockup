"""Pre-commit hook: fail if any staged file contains a string from the local denylist.

The denylist is `.denylist.local` at the repo root — gitignored, because it holds the very strings
it protects. One entry per line, matched case-insensitively; blank lines and lines starting with
`#` are skipped. An entry is a plain substring, unless it starts with `re:`, in which case the rest
of the line is a regular expression (also case-insensitive) — for terms that are only a problem as
a whole word. If the file is absent or empty this is a no-op, so a fresh clone (or CI) is never
blocked by a list it cannot have.

It reads the *staged* blob (`git show :path`), not the working tree, so it checks what the commit
will actually contain. File paths are checked too. A hit is reported by file, line and the entry's
line number in the denylist — never the string itself, so the output can be pasted into an issue or
a log without re-leaking it.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

DENYLIST_NAME = ".denylist.local"
REGEX_PREFIX = "re:"


def load_denylist(path: Path) -> list[tuple[int, re.Pattern[str]]]:
    """(line number, compiled pattern) for each usable line; empty if the file does not exist.

    A bad `re:` pattern raises ValueError naming only its line number, never the pattern.
    """
    if not path.is_file():
        return []
    entries = []
    for number, line in enumerate(path.read_text().splitlines(), start=1):
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        source = entry[len(REGEX_PREFIX) :] if entry.startswith(REGEX_PREFIX) else re.escape(entry)
        try:
            entries.append((number, re.compile(source, re.IGNORECASE)))
        except re.error:
            raise ValueError(f"{DENYLIST_NAME} line {number}: invalid regular expression") from None
    return entries


def find_hits(text: str, entries: list[tuple[int, re.Pattern[str]]]) -> list[tuple[int, int]]:
    """(line in `text`, denylist line number) for every match."""
    hits = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        hits += [(line_number, entry_no) for entry_no, pattern in entries if pattern.search(line)]
    return hits


def _git(*args: str, cwd: Path) -> bytes:
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, check=True).stdout


def staged_files(cwd: Path) -> list[str]:
    out = _git("diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z", cwd=cwd)
    return [p for p in out.decode().split("\0") if p]


def check(repo: Path) -> list[str]:
    entries = load_denylist(repo / DENYLIST_NAME)
    if not entries:
        return []

    problems = []
    for path in staged_files(repo):
        for _, entry_no in find_hits(path, entries):
            problems.append(f"{path}: file name matches denylist entry #{entry_no}")
        blob = _git("show", f":{path}", cwd=repo)
        if b"\0" in blob:  # binary: no lines to report, and text matching on it is noise
            continue
        for line_no, entry_no in find_hits(blob.decode(errors="replace"), entries):
            problems.append(f"{path}:{line_no}: matches denylist entry #{entry_no}")
    return problems


def main() -> int:
    repo = Path(_git("rev-parse", "--show-toplevel", cwd=Path.cwd()).decode().strip())
    problems = check(repo)
    if not problems:
        return 0
    print(f"Commit blocked: staged content matches {DENYLIST_NAME}", file=sys.stderr)
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())

"""The pre-commit denylist hook, exercised against a throwaway git repo — nothing here touches the
real repo's index or denylist."""

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_denylist.py"
spec = importlib.util.spec_from_file_location("check_denylist", SCRIPT)
hook = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hook)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q")
    return tmp_path


def _stage(repo: Path, name: str, content: str | bytes) -> None:
    path = repo / name
    path.parent.mkdir(parents=True, exist_ok=True)
    (path.write_bytes if isinstance(content, bytes) else path.write_text)(content)
    _git(repo, "add", name)


def test_no_denylist_file_is_a_noop(repo):
    _stage(repo, "notes.md", "anything at all, including canary-zq9")
    assert hook.check(repo) == []


def test_comment_and_blank_only_denylist_is_a_noop(repo):
    (repo / hook.DENYLIST_NAME).write_text("# nothing yet\n\n   \n")
    _stage(repo, "notes.md", "canary-zq9")
    assert hook.check(repo) == []


def test_staged_content_match_is_reported_by_line_and_entry_not_by_string(repo):
    (repo / hook.DENYLIST_NAME).write_text("# header\nharmless\nCanary-ZQ9\n")
    _stage(repo, "docs/notes.md", "line one\nsee the CANARY-zq9 here\n")
    problems = hook.check(repo)
    assert problems == ["docs/notes.md:2: matches denylist entry #3"]
    assert all("canary" not in p.lower() for p in problems)


def test_checks_the_staged_blob_not_the_working_tree(repo):
    (repo / hook.DENYLIST_NAME).write_text("canary-zq9\n")
    _stage(repo, "notes.md", "clean")
    (repo / "notes.md").write_text("canary-zq9 added after staging")
    assert hook.check(repo) == []


def test_file_name_match_is_reported(repo):
    (repo / hook.DENYLIST_NAME).write_text("canary-zq9\n")
    _stage(repo, "canary-zq9.txt", "clean")
    assert hook.check(repo) == ["canary-zq9.txt: file name matches denylist entry #1"]


def test_binary_files_are_skipped(repo):
    (repo / hook.DENYLIST_NAME).write_text("canary-zq9\n")
    _stage(repo, "blob.bin", b"\x00\x01canary-zq9\x02")
    assert hook.check(repo) == []


WHOLE_WORD = r"re:(?<![\w-])gizmos(?![\w-])"


def test_regex_entry_matches_case_insensitively_by_line_and_entry(repo):
    (repo / hook.DENYLIST_NAME).write_text(f"# header\n{WHOLE_WORD}\n")
    _stage(repo, "notes.md", "clean\nBlue GIZMOS, and gizmos.\n(gizmos)\n")
    assert hook.check(repo) == [
        "notes.md:2: matches denylist entry #2",
        "notes.md:3: matches denylist entry #2",
    ]


@pytest.mark.parametrize(
    "text",
    [
        "gizmo",
        "gizmo chart",
        "gizmo-in and gizmo-out",
        "gizmosite",
        "gizmos-out",
        "pre-gizmos",
        "gizmo_s",
    ],
)
def test_whole_word_pattern_has_no_false_positives(repo, text):
    (repo / hook.DENYLIST_NAME).write_text(WHOLE_WORD + "\n")
    _stage(repo, "notes.md", text)
    assert hook.check(repo) == []


def test_regex_entry_also_checks_file_names(repo):
    (repo / hook.DENYLIST_NAME).write_text(WHOLE_WORD + "\n")
    _stage(repo, "blue gizmos.txt", "clean")
    assert hook.check(repo) == ["blue gizmos.txt: file name matches denylist entry #1"]


def test_plain_entries_stay_literal_not_regex(repo):
    (repo / hook.DENYLIST_NAME).write_text("a.c\n")
    _stage(repo, "notes.md", "abc\nA.C\n")
    assert hook.check(repo) == ["notes.md:2: matches denylist entry #1"]


def test_invalid_regex_fails_loudly_without_echoing_the_pattern(repo):
    (repo / hook.DENYLIST_NAME).write_text("ok\nre:canary-zq9(\n")
    with pytest.raises(ValueError) as excinfo:
        hook.load_denylist(repo / hook.DENYLIST_NAME)
    assert "line 2" in str(excinfo.value)
    assert "canary" not in str(excinfo.value)

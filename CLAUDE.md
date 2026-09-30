# Working rules for this repository

## CI is part of "pushed"

After every push to `main`, check the GitHub Actions run **for that commit** and wait for it to finish.
Do not report "pushed", "done" or "fixed" until it is green. If it is red, say so, read the failing job's
log (the whole comparison, not just its first failing line), and fix it before anything else, or report
it as red.

Why this is here: twice in this project a red CI went unnoticed because the report was about the push
and not its result: a Docker image that had never built (fixed in v1.2.1), and a regression test that
failed on the x86 runner for three commits (v1.3.0; `DECISIONS.md`, 2026-09-30). Tags are cut only from a
commit whose CI is green.

```bash
gh run list --limit 3 --json databaseId,headSha,conclusion,displayTitle
gh run watch <id> --exit-status
```

## Registered analyses

Track B's design, decision rules and economics were fixed in public commits before any model ran
(`docs/track_b.md`). Anything run afterwards is labelled post hoc, sits beside the registered verdicts,
and never replaces them. A registered document is corrected by a dated note, not edited in place.

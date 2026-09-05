"""Render an eval run as GitHub step-summary markdown.

Thin CLI over evals.summary_md, mirroring scripts/run_eval.py's relationship
to evals.cli: the logic lives in the package so the eval-suite unit-test job
covers it, and this file is only what the workflow invokes.

Usage:
    python scripts/render_eval_summary.py evals/runs/<run-id> \\
        --tag v2026.08.30 --commit abc123 --base-url http://dev.example \\
        >> "$GITHUB_STEP_SUMMARY"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Run THIS checkout's evals package, not whichever worktree last ran
# `pip install -e evals/`. See scripts/_local_evals.py.
from _local_evals import use_local_evals  # noqa: E402

use_local_evals()

from evals.summary_md import render_summary_markdown  # noqa: E402  -- follows use_local_evals()

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "evals" / "baselines" / "dev.json"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="evals/runs/<run-id> directory to render")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--tag", required=True, help="release tag, e.g. v2026.08.30")
    parser.add_argument("--commit", required=True, help="commit the tag points at")
    parser.add_argument("--base-url", required=True, help="dev base URL the suite ran against")
    parser.add_argument("--category", default="positive")
    args = parser.parse_args()

    # The report uses unicode delta markers (▲ / 🔻). $GITHUB_STEP_SUMMARY is
    # UTF-8, but stdout's default encoding on some platforms (e.g. a
    # non-UTF-8 codepage on Windows) can't represent them -- force it so the
    # CLI doesn't crash on the very run it's supposed to explain.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")

    summary_path = args.run_dir / "summary.json"
    if not summary_path.exists():
        # Rendered with `if: always()`, so this is reachable when the eval run
        # itself died. Say so in the summary rather than failing silently.
        print(f"# Release {args.tag}\n")
        print(f"**The eval run produced no summary.** `{summary_path}` is missing — "
              "the suite failed before it could report. Check the job log above.")
        return 1

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    baseline = (
        json.loads(args.baseline.read_text(encoding="utf-8"))
        if args.baseline.exists()
        else None
    )

    print(
        render_summary_markdown(
            summary,
            baseline,
            tag=args.tag,
            commit=args.commit,
            base_url=args.base_url,
            category=args.category,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

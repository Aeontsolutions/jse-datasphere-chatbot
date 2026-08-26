"""Gate a deploy on eval-suite results: compares a run's summary.json against
a checked-in baseline and fails (exit 1) on regression or hard-failure
conditions. Used by .github/workflows/deploy.yml between the dev deploy and
the prod deploy.

Judge dimensions are scored 1-5 (see evals/judge.py). `overall` mixes in the
negative/refusal personas, which score goal_completion=1.0 by design -- so
this gate reads `by_category.positive` only, per evals/README and
[[reference_eval_suite]].

Usage:
    # First time (or deliberately promoting a new baseline after a reviewed
    # improvement): run the suite, then record it as the baseline to diff
    # future runs against.
    python scripts/check_eval_gate.py evals/runs/<run-id> --update-baseline

    # Normal gate check (what CI runs):
    python scripts/check_eval_gate.py evals/runs/<run-id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "evals" / "baselines" / "dev.json"
DIMENSIONS = ("groundedness", "factfulness", "goal_completion", "persona_handling", "coherence")


def load_summary(run_dir: Path) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        sys.exit(f"ERROR: {summary_path} not found -- did the eval run finish and write a report?")
    return json.loads(summary_path.read_text(encoding="utf-8"))


def category_means(summary: dict, category: str) -> dict:
    cat = summary.get("by_category", {}).get(category)
    if cat is None:
        sys.exit(f"ERROR: summary has no by_category.{category} -- check personas were selected correctly")
    return {dim: cat.get(f"mean_{dim}") for dim in DIMENSIONS}


def write_baseline(path: Path, summary: dict, category: str) -> None:
    means = category_means(summary, category)
    fail_verdicts = summary["by_category"][category].get("verdict_counts", {}).get("fail", 0)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "source_run_id": summary.get("run_id"),
                "category": category,
                "means": means,
                "fail_verdicts": fail_verdicts,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote baseline to {path} from run {summary.get('run_id')}: {means}, fail_verdicts={fail_verdicts}")


def annotate(level: str, message: str) -> None:
    print(message)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path, help="evals/runs/<run-id> directory to check")
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--category", default="positive", help="by_category key to gate on (default: positive)")
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.4,
        help="Max allowed drop (1-5 scale) per dimension vs baseline before failing (default: 0.4)",
    )
    parser.add_argument(
        "--max-fail-increase",
        type=int,
        default=2,
        help="Max allowed increase in 'fail' verdicts vs baseline before failing (default: 2). Relative, "
        "not absolute -- many 'fail' verdicts reflect known data-coverage gaps (personas asking for years "
        "or fields the DB doesn't have), not app bugs, so an absolute cap of 0 would fail every run.",
    )
    parser.add_argument(
        "--max-judge-failed",
        type=int,
        default=0,
        help="Max allowed judge_failed_count (LLM-judge errors, not app failures) before failing (default: 0)",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write this run's means as the new baseline instead of gating. Use deliberately, after "
        "reviewing the run -- never wire this into an automatic pass/fail pipeline step, or a slow "
        "score decline just becomes the new normal every time it happens to pass.",
    )
    args = parser.parse_args()

    summary = load_summary(args.run_dir)

    if args.update_baseline:
        write_baseline(args.baseline, summary, args.category)
        return 0

    if not args.baseline.exists():
        annotate(
            "error",
            f"No baseline at {args.baseline}. Seed one first: "
            f"python scripts/check_eval_gate.py {args.run_dir} --update-baseline",
        )
        return 1

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_means = baseline["means"]
    current = category_means(summary, args.category)

    failures: list[str] = []
    for dim in DIMENSIONS:
        base_val = baseline_means.get(dim)
        cur_val = current.get(dim)
        if base_val is None:
            continue
        if cur_val is None:
            failures.append(f"{dim}: no score in current run (baseline was {base_val})")
            continue
        drop = base_val - cur_val
        flag = "REGRESSION" if drop > args.max_regression else "ok"
        print(f"  {dim:18s} baseline={base_val:.2f} current={cur_val:.2f} drop={drop:+.2f}  [{flag}]")
        if drop > args.max_regression:
            failures.append(f"{dim} dropped {drop:.2f} (baseline {base_val:.2f} -> {cur_val:.2f}, max allowed {args.max_regression})")

    cat_stats = summary["by_category"][args.category]
    fail_verdicts = cat_stats.get("verdict_counts", {}).get("fail", 0)
    baseline_fail_verdicts = baseline.get("fail_verdicts", 0)
    fail_increase = fail_verdicts - baseline_fail_verdicts
    print(f"  fail_verdicts      baseline={baseline_fail_verdicts} current={fail_verdicts} increase={fail_increase:+d}")
    if fail_increase > args.max_fail_increase:
        failures.append(
            f"fail_verdicts rose by {fail_increase} (baseline {baseline_fail_verdicts} -> {fail_verdicts}, "
            f"max allowed increase {args.max_fail_increase})"
        )

    judge_failed = summary.get("overall", {}).get("judge_failed_count", 0)
    if judge_failed > args.max_judge_failed:
        failures.append(f"{judge_failed} judge_failed_count (max allowed {args.max_judge_failed})")

    incomplete = summary.get("overall", {}).get("incomplete_count", 0)
    if incomplete:
        annotate("warning", f"{incomplete} incomplete conversation(s) (transport errors) -- not gating on this, but worth a look.")

    if failures:
        annotate("error", f"EVAL GATE FAILED ({args.run_dir}, baseline {baseline.get('source_run_id')}):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"EVAL GATE PASSED ({args.run_dir}, baseline {baseline.get('source_run_id')})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

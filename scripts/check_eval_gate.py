"""Gate a deploy on eval-suite results: compares a run's summary.json against
a checked-in baseline and fails (exit 1) on regression or hard-failure
conditions. Run by .github/workflows/deploy-dev.yml's eval-gate job (after the
dev deploy) and .github/workflows/release-prod.yml's release-eval job (before
the human approval on deploy-prod).

Judge dimensions are scored 1-5 (see evals/judge.py). `overall` mixes in the
negative/refusal personas, which score goal_completion=1.0 by design -- so
this gate reads `by_category.positive` only, per evals/README and
[[reference_eval_suite]].

The per-dimension regression threshold is derived from the spread the runs
actually measured, not asserted -- see evals/gate_stats.py for why a fixed 0.4
sat below the noise floor.

Usage:
    # Normal gate check (what CI runs):
    python scripts/check_eval_gate.py evals/runs/<run-id>

    # Seed or promote the baseline, after reviewing the runs. Pass several
    # runs: a baseline from one run enshrines that run's luck, and its own
    # standard error is what floors the gate's threshold.
    python scripts/check_eval_gate.py evals/runs/<a> evals/runs/<b> evals/runs/<c> --update-baseline
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# The `evals` package is pip-installed (`cd evals && pip install -e .`), which
# both CI jobs do before invoking this script. The threshold maths lives there
# rather than here so the eval-suite unit-test job covers it -- scripts/ has no
# test job of its own.
# Run THIS checkout's evals package, not whichever worktree last ran
# `pip install -e evals/`. See scripts/_local_evals.py.
from _local_evals import use_local_evals  # noqa: E402

use_local_evals()

from evals.gate_stats import (  # noqa: E402  -- must follow use_local_evals()
    DIMENSIONS,
    combined_standard_error,
    pool_runs,
    regression_threshold,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO_ROOT / "evals" / "baselines" / "dev.json"


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


def write_baseline(path: Path, summaries: list[dict], category: str) -> None:
    """Pool one or more reviewed runs into a baseline.

    Prefer several runs. A baseline from a single run enshrines that run's
    luck, and its own standard error is what floors the threshold the gate
    derives -- two consecutive runs of identical code have differed by 0.41 on
    a dimension here.
    """
    baseline = pool_runs(summaries, category=category)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(baseline, indent=2) + "\n", encoding="utf-8")

    runs = ", ".join(str(r) for r in baseline["source_run_ids"])
    print(f"Wrote baseline to {path} pooled from {len(summaries)} run(s) [{runs}], n={baseline['n']}")
    for dim in DIMENSIONS:
        mean, std = baseline["means"].get(dim), baseline["stds"].get(dim)
        if mean is None:
            continue
        print(f"  {dim:18s} mean={mean:.2f} std={std:.2f}")
    print(f"  fail_verdicts={baseline['fail_verdicts']}")
    if len(summaries) < 3:
        annotate(
            "warning",
            f"Baseline pooled from only {len(summaries)} run(s). Three or more gives a tighter "
            "gate, because the baseline's own standard error floors the threshold.",
        )


def _baseline_label(baseline: dict) -> str:
    """Pooled baselines record source_run_ids; ones written before pooling record source_run_id."""
    ids = baseline.get("source_run_ids")
    if ids:
        return ", ".join(str(i) for i in ids)
    return str(baseline.get("source_run_id"))


def annotate(level: str, message: str) -> None:
    print(message)
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level}::{message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "run_dirs",
        type=Path,
        nargs="+",
        metavar="RUN_DIR",
        help="evals/runs/<run-id> directory to check. Several may be given only with --update-baseline, which pools them.",
    )
    parser.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument("--category", default="positive", help="by_category key to gate on (default: positive)")
    parser.add_argument(
        "--regression-sigma",
        type=float,
        default=2.0,
        help="How many standard errors a dimension may drop before it counts as a regression "
        "(default: 2.0). The threshold is derived from the spread the runs actually measured "
        "rather than asserted -- see evals/gate_stats.py. At sigma 2.0 across five dimensions "
        "roughly one run in nine trips on noise alone; re-run before investigating a single "
        "tripped dimension.",
    )
    parser.add_argument(
        "--min-effect",
        type=float,
        default=0.15,
        help="Practical-significance floor on the regression threshold (default: 0.15). "
        "Statistical significance scales with sample size, so without this a large enough "
        "run would eventually block releases over a drop nobody can perceive.",
    )
    parser.add_argument(
        "--max-regression",
        type=float,
        default=0.4,
        help="Fixed fallback threshold, used only against a legacy baseline that records no "
        "`stds`/`n` (default: 0.4). Re-seed the baseline to get the variance-derived threshold.",
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
        "--max-judge-failed-pct",
        type=float,
        default=10.0,
        help="Max %% of conversations allowed to go unjudged (LLM-judge errors, not app "
        "failures) before failing (default: 10.0). Proportional rather than absolute for "
        "the same reason --max-fail-increase is relative: Gemini returns transient 503s "
        "under load, and an absolute cap of 0 failed a real run (2026-08-31) in which "
        "every quality dimension improved. evals/_genai_retry.py retries those errors "
        "first; this is the backstop for when the retries are also exhausted.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Write this run's means as the new baseline instead of gating. Use deliberately, after "
        "reviewing the run -- never wire this into an automatic pass/fail pipeline step, or a slow "
        "score decline just becomes the new normal every time it happens to pass.",
    )
    args = parser.parse_args()

    if args.update_baseline:
        write_baseline(args.baseline, [load_summary(d) for d in args.run_dirs], args.category)
        return 0

    if len(args.run_dirs) > 1:
        annotate("error", "Several run dirs are only accepted with --update-baseline; gate one run at a time.")
        return 1

    run_dir = args.run_dirs[0]
    summary = load_summary(run_dir)

    if not args.baseline.exists():
        annotate(
            "error",
            f"No baseline at {args.baseline}. Seed one first: "
            f"python scripts/check_eval_gate.py {run_dir} [<run-dir> ...] --update-baseline",
        )
        return 1

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    baseline_means = baseline["means"]
    current = category_means(summary, args.category)

    cat_stats = summary["by_category"][args.category]
    baseline_stds = baseline.get("stds")
    baseline_n = baseline.get("n")
    cur_n = cat_stats.get("judged_count") or 0

    # A baseline written before the variance-derived threshold records no
    # spread, so there is nothing to derive from. Fall back to the old fixed
    # number rather than inventing one, and say so loudly -- a gate quietly
    # running looser than the operator believes is the failure this whole
    # change exists to remove.
    variance_mode = bool(baseline_stds) and bool(baseline_n) and cur_n > 0
    if not variance_mode:
        annotate(
            "warning",
            f"Baseline {args.baseline} records no stds/n -- falling back to the fixed "
            f"--max-regression {args.max_regression}, which is below the measured noise floor. "
            f"Re-seed it: python scripts/check_eval_gate.py <run-dir> [<run-dir> ...] --update-baseline",
        )

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

        if variance_mode:
            se = combined_standard_error(
                cur_std=cat_stats.get(f"std_{dim}") or 0.0,
                cur_n=cur_n,
                base_std=baseline_stds.get(dim) or 0.0,
                base_n=baseline_n,
            )
            threshold = regression_threshold(se=se, sigma=args.regression_sigma, min_effect=args.min_effect)
        else:
            threshold = args.max_regression

        flag = "REGRESSION" if drop > threshold else "ok"
        print(
            f"  {dim:18s} baseline={base_val:.2f} current={cur_val:.2f} "
            f"drop={drop:+.2f} allowed={threshold:.2f}  [{flag}]"
        )
        if drop > threshold:
            failures.append(
                f"{dim} dropped {drop:.2f} (baseline {base_val:.2f} -> {cur_val:.2f}, "
                f"max allowed {threshold:.2f})"
            )

    # Counted from the deterministic rubric-weighted verdict. The judge's own
    # holistic verdict disagrees with the rubric's documented weights on ~1
    # conversation in 6 (measured 84% and 87% over two 93-conversation runs)
    # and is the harsher of the two, so gating on it put the judge's scoring
    # drift in the release path. Both are printed; only the computed one gates.
    computed_counts = cat_stats.get("computed_verdict_counts")
    judge_fail = cat_stats.get("verdict_counts", {}).get("fail", 0)
    if computed_counts is None:
        annotate(
            "error",
            "summary has no computed_verdict_counts -- it came from a build that gated on "
            "the judge's holistic verdict. Re-run the suite on a current build; silently "
            "comparing the two metrics would be worse than failing here.",
        )
        return 1
    if baseline.get("verdict_source") != "computed":
        annotate(
            "error",
            f"baseline {args.baseline} was seeded from judge verdicts (no "
            "verdict_source='computed'). Its fail_verdicts is not comparable with the "
            "computed count. Re-seed it with --update-baseline over three or more runs.",
        )
        return 1

    fail_verdicts = computed_counts.get("fail", 0)
    baseline_fail_verdicts = baseline.get("fail_verdicts", 0)
    fail_increase = fail_verdicts - baseline_fail_verdicts
    print(
        f"  fail_verdicts      baseline={baseline_fail_verdicts} current={fail_verdicts} "
        f"increase={fail_increase:+d}  [computed verdict]"
    )
    print(
        f"  (judge holistic fail count for reference: {judge_fail}; "
        f"agreement={cat_stats.get('verdict_agreement_rate')})"
    )
    if fail_increase > args.max_fail_increase:
        failures.append(
            f"fail_verdicts rose by {fail_increase} (baseline {baseline_fail_verdicts} -> {fail_verdicts}, "
            f"max allowed increase {args.max_fail_increase})"
        )

    # Proportional, not absolute: an unjudged conversation is dropped from the
    # scored sample, so what matters is how much of the evidence went missing,
    # not the raw count. Printed on every run, passing or not -- tolerating a
    # shrinking sample silently is the failure mode this threshold trades for.
    judge_failed = summary.get("overall", {}).get("judge_failed_count", 0)
    total_convos = summary.get("conversation_count", 0)
    if not total_convos:
        failures.append("summary reports conversation_count=0 -- the run produced nothing to gate on")
    else:
        judge_failed_pct = judge_failed / total_convos * 100
        print(
            f"  judge_failed       {judge_failed}/{total_convos} unjudged "
            f"({judge_failed_pct:.1f}%, max allowed {args.max_judge_failed_pct:.1f}%)"
        )
        if judge_failed_pct > args.max_judge_failed_pct:
            failures.append(
                f"{judge_failed} of {total_convos} conversations went unjudged "
                f"({judge_failed_pct:.1f}%, max allowed {args.max_judge_failed_pct:.1f}%) -- "
                "the scored means rest on a smaller sample than the suite intended"
            )

    incomplete = summary.get("overall", {}).get("incomplete_count", 0)
    if incomplete:
        annotate("warning", f"{incomplete} incomplete conversation(s) (transport errors) -- not gating on this, but worth a look.")

    if failures:
        annotate("error", f"EVAL GATE FAILED ({run_dir}, baseline {_baseline_label(baseline)}):")
        for f in failures:
            print(f"  - {f}")
        return 1

    print(f"EVAL GATE PASSED ({run_dir}, baseline {_baseline_label(baseline)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Variance-derived regression threshold for the eval gate.

The gate used to assert a fixed `--max-regression 0.4`. That number was below
the noise floor: per-conversation std is 1.2-1.8 on a 1-5 scale, so with 21
judged conversations a dimension's mean carries a standard error of ~0.34 and
the run-to-run difference has SE ~0.48. Two consecutive CI runs of identical
app code differed by +0.41 on groundedness. The gate looked stable only
because a stale baseline sat 0.5-1.0 below current scores and quietly absorbed
the noise -- accidental slack that ranged from 0.50 to 1.07 depending on which
dimension you looked at.

This module derives the threshold from the spread the run actually measured,
so it cannot be set below the noise by accident, and it tightens on its own as
replicates rise.

Logic lives here rather than in scripts/check_eval_gate.py because `scripts/`
has no test job -- the same reason evals/summary_md.py exists.
"""

from __future__ import annotations

import math
from typing import Any

DIMENSIONS = (
    "groundedness",
    "factfulness",
    "goal_completion",
    "persona_handling",
    "coherence",
)


def _category(summary: dict[str, Any], category: str) -> dict[str, Any]:
    cat = summary.get("by_category", {}).get(category)
    if cat is None:
        raise ValueError(f"summary {summary.get('run_id')!r} has no by_category.{category}")
    return cat


def pool_runs(summaries: list[dict[str, Any]], *, category: str) -> dict[str, Any]:
    """Combine several runs into one baseline, as if they were one sample.

    A baseline built from a single run enshrines that run's luck. Pooling
    reduces the baseline's own standard error, which is what floors the
    combined error the threshold is derived from.

    The pooled variance deliberately includes between-run spread as well as
    within-run spread: run-to-run drift is real noise the gate has to tolerate,
    and averaging the per-run stds would understate it badly (for two runs of
    [1,3] and [5,7] it reports 1.41 against a true 2.58).
    """
    if not summaries:
        raise ValueError("pool_runs needs at least one run summary")

    cats = [_category(s, category) for s in summaries]
    ns = [c.get("judged_count") or 0 for c in cats]
    total_n = sum(ns)
    if total_n <= 1:
        raise ValueError(f"pooled sample is too small to estimate spread (n={total_n})")

    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for dim in DIMENSIONS:
        values = [(n, c.get(f"mean_{dim}"), c.get(f"std_{dim}")) for n, c in zip(ns, cats)]
        usable = [(n, m, s) for n, m, s in values if n and m is not None]
        if not usable:
            means[dim] = None  # type: ignore[assignment]
            stds[dim] = None  # type: ignore[assignment]
            continue
        n_dim = sum(n for n, _, _ in usable)
        pooled_mean = sum(n * m for n, m, _ in usable) / n_dim
        # Total sum of squares = within-run + between-run.
        within = sum((n - 1) * (s or 0.0) ** 2 for n, _, s in usable)
        between = sum(n * (m - pooled_mean) ** 2 for n, m, _ in usable)
        variance = (within + between) / (n_dim - 1) if n_dim > 1 else 0.0
        means[dim] = pooled_mean
        stds[dim] = math.sqrt(variance)

    # Counted from the deterministic rubric-weighted verdict, not the judge's
    # holistic one. The two disagree on ~1 conversation in 6 and the judge is
    # the harsher of the pair, so gating on it imported the judge's scoring
    # drift into the release decision: across twelve CI-shaped slices the
    # judge's fail count had stdev 1.52 and swung by 5, against a
    # --max-fail-increase tolerance of 2. The computed count had stdev 0.83.
    #
    # A summary written before computed_verdict_counts existed cannot be
    # pooled: silently falling back to verdict_counts would produce a baseline
    # that looks current while mixing two different metrics.
    missing = [
        s.get("run_id") for s, c in zip(summaries, cats) if "computed_verdict_counts" not in c
    ]
    if missing:
        raise ValueError(
            "cannot pool a baseline from runs without computed_verdict_counts "
            f"(missing in: {', '.join(str(m) for m in missing)}). Re-run the suite "
            "on a build that emits it rather than mixing verdict sources."
        )

    # Averaged, not summed: the gate compares this against a single run's
    # fail count, so a pooled baseline must stay on that scale.
    fail_counts = [c["computed_verdict_counts"].get("fail", 0) for c in cats]
    fail_verdicts = round(sum(fail_counts) / len(fail_counts))

    return {
        "source_run_ids": [s.get("run_id") for s in summaries],
        "category": category,
        "means": means,
        "stds": stds,
        "n": total_n,
        "fail_verdicts": fail_verdicts,
        # Stamped so a baseline seeded under the old counting is detectable
        # rather than silently compared against the new metric.
        "verdict_source": "computed",
    }


def combined_standard_error(*, cur_std: float, cur_n: int, base_std: float, base_n: int) -> float:
    """Standard error of the difference between two independent means."""
    if cur_n <= 0 or base_n <= 0:
        raise ValueError(f"sample sizes must be positive (cur_n={cur_n}, base_n={base_n})")
    se_cur = (cur_std or 0.0) / math.sqrt(cur_n)
    se_base = (base_std or 0.0) / math.sqrt(base_n)
    return math.sqrt(se_cur**2 + se_base**2)


def regression_threshold(*, se: float, sigma: float, min_effect: float) -> float:
    """How far a dimension may drop before it counts as a regression.

    `min_effect` is a practical-significance floor. Statistical significance
    scales with sample size, so without it a large enough run would eventually
    block releases over a 0.02 drop nobody could perceive.
    """
    return max(min_effect, sigma * se)

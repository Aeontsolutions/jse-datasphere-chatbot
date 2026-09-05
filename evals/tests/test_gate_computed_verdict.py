"""The gate counts failures from the deterministic verdict, not the judge's.

Every conversation carries two verdicts: the judge LLM's holistic `verdict`,
and `computed_verdict` -- derived deterministically from the six dimension
scores using judge_rubric.yaml's documented `verdict_weights`. They disagree
on roughly one conversation in six (measured: 84% and 87% agreement across two
93-conversation runs), and the disagreement is directional -- the judge is
consistently harsher than its own rubric.

Gating on the holistic verdict therefore imported the judge's scoring drift
into the release decision. Measured over twelve CI-shaped slices (1 replicate
over all personas) drawn from four runs:

    judge    fail counts [8,9,4,4,5,5,6,6,4,6,7,6]  spread 5  stdev 1.52
    computed fail counts [5,3,3,3,3,3,2,3,3,3,3,5]  spread 3  stdev 0.83

`--max-fail-increase` defaults to 2, so the judge-verdict metric swung further
than the gate's whole tolerance. The computed metric fits inside it.

Both counts are still reported. The judge's verdict is not discarded -- it is
what `verdict_agreement_rate` is computed against, and a drop there is a real
signal about the judge or the rubric.
"""

from __future__ import annotations

import pytest

from evals.gate_stats import pool_runs


def _summary(run_id, *, judge_fails, computed_fails, n=25, mean=4.0, std=1.2):
    cat = {
        "count": n,
        "judged_count": n,
        "verdict_counts": {"pass": n - judge_fails, "partial": 0, "fail": judge_fails},
        "computed_verdict_counts": {
            "pass": n - computed_fails,
            "partial": 0,
            "fail": computed_fails,
        },
        "mean_coherence": mean,
        "std_coherence": std,
    }
    return {"run_id": run_id, "by_category": {"positive": cat}}


def test_baseline_fail_count_comes_from_the_computed_verdict():
    s = _summary("r1", judge_fails=8, computed_fails=3)
    assert pool_runs([s], category="positive")["fail_verdicts"] == 3


def test_baseline_records_which_verdict_it_counted():
    """A baseline seeded before this change counted judge verdicts. Comparing
    computed counts against it is apples to oranges, so the source is stamped
    and the gate refuses to compare across a mismatch."""
    assert (
        pool_runs([_summary("r1", judge_fails=8, computed_fails=3)], category="positive")[
            "verdict_source"
        ]
        == "computed"
    )


def test_pooling_averages_computed_fails_across_runs():
    """Averaged, not summed -- the gate compares a pooled baseline against a
    single run's count, so it must stay on that scale."""
    runs = [
        _summary("r1", judge_fails=9, computed_fails=3),
        _summary("r2", judge_fails=4, computed_fails=5),
    ]
    assert pool_runs(runs, category="positive")["fail_verdicts"] == 4


def test_legacy_summary_without_computed_counts_is_rejected():
    """Silently falling back to the judge's count would produce a baseline
    that looks current but mixes the two metrics."""
    legacy = _summary("old", judge_fails=8, computed_fails=0)
    del legacy["by_category"]["positive"]["computed_verdict_counts"]
    with pytest.raises(ValueError, match="computed_verdict_counts"):
        pool_runs([legacy], category="positive")

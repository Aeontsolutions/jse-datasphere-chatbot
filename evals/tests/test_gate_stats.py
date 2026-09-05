"""Tests for the eval gate's variance-derived regression threshold.

Why this exists: the gate's fixed `--max-regression 0.4` was below the noise
floor. Per-conversation std is 1.2-1.8 on a 1-5 scale, so with 21 judged
conversations the standard error of a dimension's mean is ~0.34 and the
run-to-run difference has SE ~0.48. Two consecutive CI runs of identical
app code differed by +0.41 on groundedness -- more than the threshold, purely
by chance. The gate only appeared stable because a stale baseline sat 0.5-1.0
below current scores and absorbed the noise.

The threshold is now derived from the measured spread instead of asserted.
"""

from __future__ import annotations

import math

import pytest

from evals.gate_stats import combined_standard_error, pool_runs, regression_threshold


def _summary(run_id: str, *, means: dict[str, float], stds: dict[str, float], n: int, fails: int):
    cat = {
        "count": n,
        "judged_count": n,
        # `fails` now describes the computed verdict, which is what pool_runs
        # counts. The judge's holistic counts are kept alongside and are
        # deliberately different so a read of the wrong field is visible.
        "verdict_counts": {"pass": 0, "partial": 0, "fail": fails + 5},
        "computed_verdict_counts": {"pass": 0, "partial": 0, "fail": fails},
    }
    for dim, value in means.items():
        cat[f"mean_{dim}"] = value
    for dim, value in stds.items():
        cat[f"std_{dim}"] = value
    return {"run_id": run_id, "by_category": {"positive": cat}}


# ---------------------------------------------------------------------------
# Pooling
# ---------------------------------------------------------------------------


def test_pooling_a_single_run_reproduces_it():
    s = _summary("r1", means={"coherence": 3.5}, stds={"coherence": 1.4}, n=21, fails=8)
    pooled = pool_runs([s], category="positive")

    assert pooled["means"]["coherence"] == pytest.approx(3.5)
    assert pooled["stds"]["coherence"] == pytest.approx(1.4)
    assert pooled["n"] == 21
    assert pooled["source_run_ids"] == ["r1"]


def test_pooled_mean_is_weighted_by_sample_size():
    a = _summary("r1", means={"coherence": 2.0}, stds={"coherence": 1.0}, n=10, fails=0)
    b = _summary("r2", means={"coherence": 5.0}, stds={"coherence": 1.0}, n=30, fails=0)
    pooled = pool_runs([a, b], category="positive")

    # (10*2 + 30*5) / 40 = 4.25, not the unweighted 3.5.
    assert pooled["means"]["coherence"] == pytest.approx(4.25)
    assert pooled["n"] == 40


def test_pooled_std_includes_between_run_spread():
    """Run-to-run drift is real noise and must widen the pooled spread.

    Two runs of [1,3] and [5,7]: each has std sqrt(2)~1.414, but the combined
    sample [1,3,5,7] has std sqrt(20/3)~2.582. A naive average of the two
    within-run stds would report 1.414 and understate the noise by 45%.
    """
    a = _summary("r1", means={"coherence": 2.0}, stds={"coherence": math.sqrt(2)}, n=2, fails=0)
    b = _summary("r2", means={"coherence": 6.0}, stds={"coherence": math.sqrt(2)}, n=2, fails=0)
    pooled = pool_runs([a, b], category="positive")

    assert pooled["means"]["coherence"] == pytest.approx(4.0)
    assert pooled["stds"]["coherence"] == pytest.approx(math.sqrt(20 / 3))


def test_pooled_fail_verdicts_is_the_mean_rate_scaled_to_one_run():
    # Baselines are compared against a single run's fail count, so pooling
    # must not accumulate them.
    a = _summary("r1", means={"coherence": 3.0}, stds={"coherence": 1.0}, n=20, fails=8)
    b = _summary("r2", means={"coherence": 3.0}, stds={"coherence": 1.0}, n=20, fails=10)
    pooled = pool_runs([a, b], category="positive")
    assert pooled["fail_verdicts"] == 9


def test_pooling_records_every_source_run():
    a = _summary("r1", means={"coherence": 3.0}, stds={"coherence": 1.0}, n=5, fails=0)
    b = _summary("r2", means={"coherence": 3.0}, stds={"coherence": 1.0}, n=5, fails=0)
    c = _summary("r3", means={"coherence": 3.0}, stds={"coherence": 1.0}, n=5, fails=0)
    assert pool_runs([a, b, c], category="positive")["source_run_ids"] == ["r1", "r2", "r3"]


def test_pooling_requires_at_least_one_run():
    with pytest.raises(ValueError):
        pool_runs([], category="positive")


# ---------------------------------------------------------------------------
# Threshold
# ---------------------------------------------------------------------------


def test_combined_standard_error_adds_in_quadrature():
    # SE_cur = 1.5/sqrt(25) = 0.30; SE_base = 2.0/sqrt(100) = 0.20.
    se = combined_standard_error(cur_std=1.5, cur_n=25, base_std=2.0, base_n=100)
    assert se == pytest.approx(math.sqrt(0.30**2 + 0.20**2))


def test_more_replicates_shrink_the_standard_error():
    few = combined_standard_error(cur_std=1.5, cur_n=21, base_std=1.5, base_n=63)
    many = combined_standard_error(cur_std=1.5, cur_n=63, base_std=1.5, base_n=63)
    assert many < few


def test_threshold_scales_with_sigma():
    assert regression_threshold(se=0.20, sigma=2.0, min_effect=0.0) == pytest.approx(0.40)
    assert regression_threshold(se=0.20, sigma=2.5, min_effect=0.0) == pytest.approx(0.50)


def test_min_effect_floors_the_threshold():
    """A huge n makes tiny differences statistically real but meaningless.

    Without a floor the gate would eventually block releases over a 0.02 drop
    nobody can perceive.
    """
    assert regression_threshold(se=0.01, sigma=2.0, min_effect=0.15) == pytest.approx(0.15)


def test_threshold_matches_the_agreed_operating_point():
    # 3 replicates (n=63) against a baseline pooled over 3 runs, std ~1.55,
    # at sigma 2.0 -- the configuration this change ships with.
    se = combined_standard_error(cur_std=1.55, cur_n=63, base_std=1.55, base_n=63)
    threshold = regression_threshold(se=se, sigma=2.0, min_effect=0.15)
    assert 0.50 < threshold < 0.60


def test_zero_sample_is_rejected_rather_than_dividing_by_zero():
    with pytest.raises(ValueError):
        combined_standard_error(cur_std=1.5, cur_n=0, base_std=1.5, base_n=63)

"""Pin the thresholds in scripts/check_eval_gate.py.

This is a hard gate on prod releases (.github/workflows/release-prod.yml, job
`release-eval`), and `scripts/` has no test job of its own -- so its behavior
was only ever exercised by real releases.

Two real failures motivated these tests, and both were the gate misjudging
noise rather than the app misbehaving:

- 2026-08-31: an absolute `--max-judge-failed 0` failed a run on two transient
  Gemini 503s while every quality dimension improved. That threshold is now
  proportional.
- The fixed `--max-regression 0.4` sat below the noise floor -- with ~21 judged
  conversations and a per-conversation std of ~1.55, a dimension's mean carries
  a standard error of ~0.34, and two consecutive runs of identical code
  differed by +0.41. That threshold is now derived from the measured spread.

These invoke the script exactly as the workflows do:

    python scripts/check_eval_gate.py evals/runs/<run-id>
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# evals/tests/ -> evals/ -> repo root. The workflows run the script from the
# repo root, so these do too.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts" / "check_eval_gate.py"

BASELINE = {
    "source_run_id": "baseline_seed",
    "category": "positive",
    "means": {
        "groundedness": 3.33,
        "factfulness": 3.38,
        "goal_completion": 3.24,
        "persona_handling": 3.43,
        "coherence": 3.29,
    },
    "fail_verdicts": 10,
}


DIMS = ("groundedness", "factfulness", "goal_completion", "persona_handling", "coherence")

# Roughly the spread seen in real runs: 1.2-1.8 per conversation on a 1-5 scale.
REAL_STD = 1.55


def _summary(*, judge_failed: int, conversation_count: int, **mean_overrides: float) -> dict:
    """A run that is healthy on every axis except the one under test."""
    means = {
        "groundedness": 3.74,
        "factfulness": 3.68,
        "goal_completion": 3.42,
        "persona_handling": 3.53,
        "coherence": 4.11,
    }
    means.update(mean_overrides)
    cat = {
        "count": 21,
        "judged_count": 21 - judge_failed,
        "verdict_counts": {"pass": 10, "partial": 3, "fail": 8},
    }
    for dim in DIMS:
        cat[f"mean_{dim}"] = means[dim]
        cat[f"std_{dim}"] = REAL_STD
    return {
        "run_id": "ci-999",
        "conversation_count": conversation_count,
        "by_category": {"positive": cat},
        "overall": {"judge_failed_count": judge_failed, "incomplete_count": 0},
    }


def _variance_baseline(**mean_overrides: float) -> dict:
    """A baseline recording spread and sample size, as pool_runs writes it."""
    means = {
        "groundedness": 3.74,
        "factfulness": 3.68,
        "goal_completion": 3.42,
        "persona_handling": 3.53,
        "coherence": 4.11,
    }
    means.update(mean_overrides)
    return {
        "source_run_ids": ["ci-1", "ci-2", "ci-3"],
        "category": "positive",
        "means": means,
        "stds": {d: REAL_STD for d in DIMS},
        "n": 63,
        "fail_verdicts": 8,
    }


@pytest.fixture
def run_gate(tmp_path, request):
    baseline = getattr(request, "param", None) or BASELINE
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline), encoding="utf-8")

    def _run(summary: dict, *extra_args: str) -> subprocess.CompletedProcess:
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(CLI), str(run_dir), "--baseline", str(baseline_path), *extra_args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

    return _run


def test_tolerates_a_transient_judge_failure_rate(run_gate):
    # The real 2026-08-31 run: 2 of 26 unjudged, 7.7%. Under the old absolute
    # cap of 0 this failed; it must now pass.
    result = run_gate(_summary(judge_failed=2, conversation_count=26))
    assert result.returncode == 0, result.stdout + result.stderr
    assert "EVAL GATE PASSED" in result.stdout


def test_fails_when_too_much_of_the_sample_went_unjudged(run_gate):
    # 5 of 26 is 19.2% -- the means would rest on four-fifths of the intended
    # sample, which is the thing the threshold exists to catch.
    result = run_gate(_summary(judge_failed=5, conversation_count=26))
    assert result.returncode == 1
    assert "went unjudged" in result.stdout
    assert "smaller sample" in result.stdout


def test_reports_the_rate_even_when_passing(run_gate):
    # Silently tolerating a shrinking sample is the failure mode this
    # threshold trades for, so the margin is printed on every run.
    result = run_gate(_summary(judge_failed=2, conversation_count=26))
    assert "2/26 unjudged" in result.stdout
    assert "7.7%" in result.stdout
    assert "max allowed 10.0%" in result.stdout


def test_threshold_is_configurable(run_gate):
    summary = _summary(judge_failed=2, conversation_count=26)
    result = run_gate(summary, "--max-judge-failed-pct", "5")
    assert result.returncode == 1
    assert "max allowed 5.0%" in result.stdout


def test_a_run_with_no_conversations_fails(run_gate):
    # Guards the zero denominator, and an empty run should never gate green.
    result = run_gate(_summary(judge_failed=0, conversation_count=0))
    assert result.returncode == 1
    assert "conversation_count=0" in result.stdout


def test_a_real_regression_still_fails(run_gate):
    # The judge-failure change must not have loosened the quality gate itself.
    summary = _summary(judge_failed=0, conversation_count=26)
    summary["by_category"]["positive"]["mean_goal_completion"] = 2.50  # -0.74 vs baseline
    result = run_gate(summary)
    assert result.returncode == 1
    assert "REGRESSION" in result.stdout


# ---------------------------------------------------------------------------
# Variance-derived threshold
#
# The fixed 0.4 sat below the noise floor: with 21 judged conversations and a
# per-conversation std of ~1.55, a dimension's mean carries a standard error of
# ~0.34. Two consecutive runs of identical code differed by +0.41. The gate now
# derives the threshold from the measured spread.
# ---------------------------------------------------------------------------


@pytest.fixture
def run_gate_variance(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_variance_baseline()), encoding="utf-8")

    def _run(summary: dict, *extra_args: str) -> subprocess.CompletedProcess:
        run_dir = tmp_path / "run"
        run_dir.mkdir(exist_ok=True)
        (run_dir / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(CLI), str(run_dir), "--baseline", str(baseline_path), *extra_args],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
        )

    return _run


def test_threshold_is_derived_not_the_fixed_number(run_gate_variance):
    result = run_gate_variance(_summary(judge_failed=0, conversation_count=26))
    assert result.returncode == 0, result.stdout + result.stderr
    # Reported per dimension so an operator can see the bar they cleared.
    assert "allowed=" in result.stdout
    assert "allowed=0.40" not in result.stdout


def test_a_drop_within_the_noise_floor_passes(run_gate_variance):
    # -0.35 on one dimension is inside normal run-to-run variation and used to
    # sit uncomfortably close to the old fixed 0.4.
    summary = _summary(judge_failed=0, conversation_count=26, goal_completion=3.42 - 0.35)
    result = run_gate_variance(summary)
    assert result.returncode == 0, result.stdout + result.stderr


def test_a_drop_beyond_the_noise_floor_fails(run_gate_variance):
    summary = _summary(judge_failed=0, conversation_count=26, goal_completion=3.42 - 1.2)
    result = run_gate_variance(summary)
    assert result.returncode == 1
    assert "REGRESSION" in result.stdout
    assert "goal_completion dropped" in result.stdout


def test_sigma_is_configurable(run_gate_variance):
    summary = _summary(judge_failed=0, conversation_count=26, goal_completion=3.42 - 0.55)
    assert run_gate_variance(summary).returncode == 0
    # A stricter sigma turns the same drop into a regression.
    tightened = run_gate_variance(summary, "--regression-sigma", "1.0")
    assert tightened.returncode == 1


def test_legacy_baseline_falls_back_and_says_so(run_gate):
    # BASELINE records no stds/n. Falling back silently would leave the gate
    # running looser than the operator believes -- exactly what this change
    # exists to remove.
    result = run_gate(_summary(judge_failed=0, conversation_count=26))
    assert "records no stds/n" in result.stdout
    assert "Re-seed it" in result.stdout


def test_update_baseline_pools_several_runs(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    dirs = []
    for i, coherence in enumerate((3.9, 4.1, 4.0)):
        d = tmp_path / f"run{i}"
        d.mkdir()
        summary = _summary(judge_failed=0, conversation_count=26, coherence=coherence)
        summary["run_id"] = f"ci-{i}"
        (d / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
        dirs.append(str(d))

    result = subprocess.run(
        [sys.executable, str(CLI), *dirs, "--baseline", str(baseline_path), "--update-baseline"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    written = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert written["source_run_ids"] == ["ci-0", "ci-1", "ci-2"]
    assert written["n"] == 63  # 21 judged x 3 runs
    assert set(written["stds"]) == set(DIMS)
    # Pooled fail verdicts stay on a single run's scale, not summed.
    assert written["fail_verdicts"] == 8


def test_several_run_dirs_are_rejected_when_gating(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(_variance_baseline()), encoding="utf-8")
    dirs = []
    for i in range(2):
        d = tmp_path / f"run{i}"
        d.mkdir()
        (d / "summary.json").write_text(
            json.dumps(_summary(judge_failed=0, conversation_count=26)), encoding="utf-8"
        )
        dirs.append(str(d))

    result = subprocess.run(
        [sys.executable, str(CLI), *dirs, "--baseline", str(baseline_path)],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 1
    assert "only accepted with --update-baseline" in result.stdout

"""Pin the judge-failure threshold in scripts/check_eval_gate.py.

This gate is a hard gate on prod releases (.github/workflows/release-prod.yml,
job `release-eval`), and `scripts/` has no test job of its own -- so its
behavior was only ever exercised by real releases.

The case that motivated these tests: on 2026-08-31 the threshold was an
absolute `--max-judge-failed 0`, and a CI run failed on two transient Gemini
503s while every quality dimension improved. The threshold is now proportional.

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


def _summary(*, judge_failed: int, conversation_count: int) -> dict:
    """A run that is healthy on every axis except the one under test."""
    return {
        "run_id": "ci-999",
        "conversation_count": conversation_count,
        "by_category": {
            "positive": {
                "count": 21,
                "judged_count": 21 - judge_failed,
                "mean_groundedness": 3.74,
                "mean_factfulness": 3.68,
                "mean_goal_completion": 3.42,
                "mean_persona_handling": 3.53,
                "mean_coherence": 4.11,
                "verdict_counts": {"pass": 10, "partial": 3, "fail": 8},
            }
        },
        "overall": {"judge_failed_count": judge_failed, "incomplete_count": 0},
    }


@pytest.fixture
def run_gate(tmp_path):
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(BASELINE), encoding="utf-8")

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

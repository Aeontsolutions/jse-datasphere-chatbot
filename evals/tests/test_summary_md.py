"""Tests for the release report the human approver reads.

This report is the whole substance of the human gate -- if it renders wrong,
someone approves a release on bad numbers.
"""

from evals.summary_md import render_summary_markdown

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


def _summary(**overrides):
    positive = {
        "count": 20,
        "judged_count": 20,
        "mean_groundedness": 3.50,
        "mean_factfulness": 3.38,
        "mean_goal_completion": 2.50,
        "mean_persona_handling": 3.43,
        "mean_coherence": 3.29,
        "verdict_counts": {"pass": 8, "partial": 5, "fail": 7},
    }
    positive.update(overrides)
    return {"run_id": "release-123", "by_category": {"positive": positive}}


def _render(summary, baseline=BASELINE):
    return render_summary_markdown(
        summary,
        baseline,
        tag="v2026.08.30",
        commit="abc123",
        base_url="http://dev.example",
    )


def test_shows_tag_and_commit():
    out = _render(_summary())
    assert "v2026.08.30" in out
    assert "abc123" in out


def test_regression_renders_negative_delta():
    # goal_completion 3.24 -> 2.50 is a 0.74 drop, past the gate's 0.4 limit.
    out = _render(_summary())
    assert "-0.74" in out


def test_improvement_renders_positive_delta():
    out = _render(_summary(mean_groundedness=3.83))
    assert "+0.50" in out


def test_fail_verdict_delta_against_baseline():
    out = _render(_summary())
    assert "-3" in out  # 7 fail verdicts vs baseline 10


def test_missing_baseline_is_stated_not_crashed():
    out = render_summary_markdown(
        _summary(),
        None,
        tag="v2026.08.30",
        commit="abc123",
        base_url="http://dev.example",
    )
    assert "No baseline" in out
    assert "n/a" in out


def test_verify_block_lists_endpoints_against_base_url():
    out = _render(_summary())
    assert "http://dev.example/docs" in out
    assert "http://dev.example/version" in out
    assert "/chat/stream" in out


def test_verify_block_tells_approver_to_check_version_first():
    # deploy-prod waits on a human, possibly for days, while every merge to
    # main redeploys dev. The approver must confirm /version still reports
    # this commit before trusting the URLs, or they hand-test another build.
    out = _render(_summary())
    check_line = next(line for line in out.splitlines() if "it must return" in line)
    assert "http://dev.example/version" in check_line
    assert "abc123" in check_line
    # The instruction has to come before the endpoint list, not after it.
    assert out.index("it must return") < out.index("Swagger UI")


def test_verify_block_says_what_a_mismatch_means():
    out = _render(_summary())
    assert "dev has moved on" in out
    assert "no longer exercise this release" in out


def test_missing_dimension_does_not_crash():
    summary = _summary()
    del summary["by_category"]["positive"]["mean_coherence"]
    out = _render(summary)
    assert "n/a" in out


def test_tool_use_appropriateness_is_not_reported():
    # _persona_stats emits it; the gate ignores it, so the report must too.
    out = _render(_summary(mean_tool_use_appropriateness=4.0))
    assert "tool_use_appropriateness" not in out

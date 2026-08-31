"""Pin the argument surface of scripts/render_eval_summary.py.

evals/tests/test_summary_md.py covers the rendering function. Nothing covered
the CLI wrapper itself -- its flag names, its exit codes, or its
missing-summary path -- so a rename would only surface at release time, in the
one job whose whole purpose is to explain a release to a human.

These invoke the script exactly as .github/workflows/release-prod.yml does:

    python scripts/render_eval_summary.py evals/runs/<run-id> \\
      --tag <tag> --commit <sha> --base-url <url> >> "$GITHUB_STEP_SUMMARY"

so a renamed flag or a changed exit code fails here instead of there.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

# evals/tests/ -> evals/ -> repo root. The workflow runs the script from the
# repo root, so these do too.
REPO_ROOT = Path(__file__).resolve().parents[2]
CLI = REPO_ROOT / "scripts" / "render_eval_summary.py"

SUMMARY = {
    "run_id": "release-999",
    "by_category": {
        "positive": {
            "count": 20,
            "judged_count": 20,
            "mean_groundedness": 3.50,
            "mean_factfulness": 3.38,
            "mean_goal_completion": 3.24,
            "mean_persona_handling": 3.43,
            "mean_coherence": 3.29,
            "verdict_counts": {"pass": 12, "partial": 5, "fail": 3},
        }
    },
}


def _run(run_dir: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(CLI),
            str(run_dir),
            "--tag",
            "v2026.08.30",
            "--commit",
            "abc123def456",
            "--base-url",
            "http://dev.example",
            *extra,
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        # The report carries unicode delta markers; decode explicitly rather
        # than inheriting a codepage that cannot represent them.
        encoding="utf-8",
        errors="replace",
    )


@pytest.fixture
def run_dir(tmp_path: Path) -> Path:
    d = tmp_path / "release-999"
    d.mkdir()
    (d / "summary.json").write_text(json.dumps(SUMMARY), encoding="utf-8")
    return d


def test_valid_run_dir_exits_zero_and_renders_a_table(run_dir: Path):
    result = _run(run_dir)
    assert result.returncode == 0, result.stderr
    assert "| Dimension | Baseline | This run | Delta |" in result.stdout
    assert "groundedness" in result.stdout
    assert "v2026.08.30" in result.stdout
    assert "abc123def456" in result.stdout


def test_missing_summary_exits_one_with_an_explanation(tmp_path: Path):
    empty = tmp_path / "release-missing"
    empty.mkdir()
    result = _run(empty)
    assert result.returncode == 1, result.stderr
    # Still writes something readable: the step renders with if: always(), so
    # this text is what the approver sees when the suite died mid-run.
    assert "The eval run produced no summary." in result.stdout
    assert "summary.json" in result.stdout
    assert "v2026.08.30" in result.stdout


def test_required_flags_are_actually_required(run_dir: Path):
    # Drop --commit: argparse must reject it (exit 2), not render a report
    # missing the commit the whole provenance story rests on.
    result = subprocess.run(
        [sys.executable, str(CLI), str(run_dir), "--tag", "v1", "--base-url", "http://x"],
        cwd=REPO_ROOT,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    assert result.returncode == 2
    assert "--commit" in result.stderr

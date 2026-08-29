# Trunk-Based Deploy Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split the single deploy workflow so every merge to `main` deploys to dev unattended, while prod ships only from a tag on `main` that passes a provenance check, a hard eval gate, and a human who has read the eval report.

**Architecture:** `deploy.yml` becomes `deploy-dev.yml` (dev only, no prod job). A new `release-prod.yml` triggers on `v*` tags and runs three jobs in sequence: `verify-tag` (dev is serving the tagged commit), `release-eval` (eval suite + hard regression gate + a rendered report), and `deploy-prod` (gated on the `prod` Environment's required reviewers). Provenance works by writing the commit into `fastapi_app/app/BUILD_SHA` at CI time so the Dockerfile bakes it into the image, and reading it back over a new dependency-free `GET /version`.

**Tech Stack:** GitHub Actions, AWS Copilot CLI v1.34.1, FastAPI, pytest, Python 3.10/3.11.

**Spec:** `docs/superpowers/specs/2026-08-30-trunk-based-deploy-pipeline-design.md`

## Global Constraints

- **Python compatibility:** `fastapi_app` code is tested on 3.10 **and** 3.11 (`test.yml` matrix). Use `from __future__ import annotations` for any `X | None` annotations. The `evals` package is tested on 3.11 only.
- **Test location:** CI runs only `fastapi_app/tests/unit/`, `fastapi_app/tests/integration/`, and `evals/` (via `cd evals && pytest -q`). Tests placed in `fastapi_app/tests/` root are **not executed** — never put new tests there.
- **Eval category:** always gate and report on `by_category.positive`, never `overall`. `overall` mixes in refusal personas that score `goal_completion=1.0` by design.
- **Gate dimensions:** exactly the five in `scripts/check_eval_gate.py` — `groundedness`, `factfulness`, `goal_completion`, `persona_handling`, `coherence`. `_persona_stats` also emits `tool_use_appropriateness`; the gate ignores it, so the report ignores it too.
- **No new runtime dependencies** in `fastapi_app`.
- **Dev base URL:** `vars.DEV_BASE_URL`, falling back to `http://jse-da-Publi-3w4oxbvTf5j0-374994583.us-east-1.elb.amazonaws.com`. Prod: `vars.PROD_BASE_URL`, falling back to `http://jse-da-Publi-2tSlV7zf7Ysl-685234288.us-east-1.elb.amazonaws.com`. Copy these verbatim.
- **Never pass a base URL as a job output.** GitHub Actions silently empties any job output whose value matches a registered secret; this URL tripped on that in practice. Recompute it per job from `env`.

---

### Task 1: Build provenance — `BUILD_SHA` and `GET /version`

Nothing today records which commit a deployed environment is running. `.dockerignore` excludes `.git/`, so the container cannot derive its own commit — CI must write it into the build context.

**Files:**
- Create: `fastapi_app/app/build_info.py`
- Create: `fastapi_app/tests/unit/test_build_info.py`
- Modify: `fastapi_app/app/main.py` (imports near top; new endpoint after `/metrics` at line 252-260; `health_status` dict at line 278-282)
- Modify: `fastapi_app/tests/integration/test_api_endpoints.py`
- Modify: `.gitignore` (repo root)

**Interfaces:**
- Consumes: nothing.
- Produces: `app.build_info.read_build_sha(path: Path | None = None) -> str`, the module constant `app.build_info.BUILD_SHA_PATH`, and `GET /version` returning `{"commit": str, "version": str}`. Task 4's `verify-tag` job reads the `commit` field.

- [ ] **Step 1: Write the failing test**

Create `fastapi_app/tests/unit/test_build_info.py`:

```python
"""Tests for build provenance reading.

CI writes the deploying commit to app/BUILD_SHA before the image is built.
Locally the file is absent, which must degrade to "unknown" rather than
raising -- every local run and every unit test hits that path.
"""

from app.build_info import UNKNOWN, read_build_sha


def test_reads_commit_from_file(tmp_path):
    sha_file = tmp_path / "BUILD_SHA"
    sha_file.write_text("abc123def456", encoding="utf-8")
    assert read_build_sha(sha_file) == "abc123def456"


def test_strips_trailing_newline(tmp_path):
    # `echo "$GITHUB_SHA" > BUILD_SHA` always leaves a trailing newline.
    sha_file = tmp_path / "BUILD_SHA"
    sha_file.write_text("abc123def456\n", encoding="utf-8")
    assert read_build_sha(sha_file) == "abc123def456"


def test_missing_file_is_unknown(tmp_path):
    assert read_build_sha(tmp_path / "does_not_exist") == UNKNOWN


def test_empty_file_is_unknown(tmp_path):
    sha_file = tmp_path / "BUILD_SHA"
    sha_file.write_text("   \n", encoding="utf-8")
    assert read_build_sha(sha_file) == UNKNOWN
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd fastapi_app && pytest tests/unit/test_build_info.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'app.build_info'`

- [ ] **Step 3: Write minimal implementation**

Create `fastapi_app/app/build_info.py`:

```python
"""Commit provenance for the running build.

`.dockerignore` excludes `.git/`, so a container cannot derive its own commit.
CI writes the deploying commit to `app/BUILD_SHA` immediately before
`copilot svc deploy`, and the Dockerfile's `COPY . .` bakes it into the image.
Locally the file is absent and the commit reads "unknown".

The release pipeline reads this back over `GET /version` to prove dev is
serving the commit a release tag points at.
"""

from __future__ import annotations

from pathlib import Path

BUILD_SHA_PATH = Path(__file__).resolve().parent / "BUILD_SHA"
UNKNOWN = "unknown"


def read_build_sha(path: Path | None = None) -> str:
    """Return the commit this build was made from, or "unknown" if unstamped."""
    target = BUILD_SHA_PATH if path is None else path
    try:
        sha = target.read_text(encoding="utf-8").strip()
    except OSError:
        return UNKNOWN
    return sha or UNKNOWN
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd fastapi_app && pytest tests/unit/test_build_info.py -v
```

Expected: PASS, 4 tests.

- [ ] **Step 5: Add the `/version` endpoint test**

In `fastapi_app/tests/integration/test_api_endpoints.py`, add to the existing `TestAPIEndpoints` class (it already has a `client` fixture using the shared `test_client`):

```python
    def test_version_endpoint(self, client):
        """/version reports build provenance and never depends on S3/BigQuery/Gemini."""
        response = client.get("/version")
        assert response.status_code == 200
        data = response.json()
        assert "commit" in data
        assert "version" in data
        # Unstamped in the test environment -- CI writes BUILD_SHA only at deploy.
        assert data["commit"] == "unknown"

    def test_health_reports_commit(self, client):
        """/health carries the same commit field, where operators look first."""
        response = client.get("/health")
        assert "commit" in response.json()
```

- [ ] **Step 6: Run to verify it fails**

```bash
cd fastapi_app && pytest tests/integration/test_api_endpoints.py -k "version or commit" -v
```

Expected: FAIL — `/version` returns 404, and `/health` has no `commit` key.

- [ ] **Step 7: Add the endpoint and the health field**

In `fastapi_app/app/main.py`, add the import alongside the other `from app.` imports near the top of the file:

```python
from app.build_info import read_build_sha
```

Add a module-level constant next to the other module constants, after the `app = FastAPI(...)` block (around line 141). Reading once at import keeps `/version` free of filesystem I/O per request:

```python
# Read once at import -- the file is baked into the image and never changes
# for the life of the process.
BUILD_SHA = read_build_sha()
```

Insert the new endpoint immediately after the `/metrics` endpoint (which ends at line 260) and before `@app.get("/health")`:

```python
@app.get("/version")
async def version():
    """Commit and app version of the running build.

    Deliberately dependency-free. `/health` probes S3, BigQuery, and Gemini
    and returns 503 when any is unhealthy, so it cannot be relied on to report
    provenance at the moment provenance matters most -- during a release. The
    release pipeline reads this endpoint to prove dev is serving the tagged
    commit before it will ship anything to prod.
    """
    return {"commit": BUILD_SHA, "version": app.version}
```

In `health_check`, add the commit to the status dict (currently lines 278-282):

```python
        health_status = {
            "status": "healthy",
            "commit": BUILD_SHA,
            "timestamp": time.time(),
            "components": {},
        }
```

- [ ] **Step 8: Run to verify it passes**

```bash
cd fastapi_app && pytest tests/integration/test_api_endpoints.py -k "version or commit" -v
```

Expected: PASS, 2 tests.

- [ ] **Step 9: Ignore the generated file**

Append to `.gitignore` at the repo root:

```
# Written by CI immediately before `copilot svc deploy` so the Dockerfile's
# COPY . . bakes the commit into the image. Never committed.
fastapi_app/app/BUILD_SHA
```

- [ ] **Step 10: Run the full affected suites**

```bash
cd fastapi_app && pytest tests/unit/ tests/integration/ -q
```

Expected: PASS, no new failures versus the pre-change baseline.

- [ ] **Step 11: Commit**

```bash
git add fastapi_app/app/build_info.py fastapi_app/tests/unit/test_build_info.py fastapi_app/app/main.py fastapi_app/tests/integration/test_api_endpoints.py .gitignore
git commit -m "feat(api): record build commit and expose it on /version"
```

---

### Task 2: Render an eval run as a report for the approver

The logic lives in the `evals` package so the existing `eval-suite-unit-tests` CI job covers it; `scripts/render_eval_summary.py` is a thin CLI over it. This mirrors `scripts/run_eval.py`, which is a wrapper over `evals.cli` (verified — see its module docstring).

**Files:**
- Create: `evals/summary_md.py`
- Create: `evals/tests/test_summary_md.py`
- Create: `scripts/render_eval_summary.py`

**Interfaces:**
- Consumes: nothing from Task 1.
- Produces: `evals.summary_md.render_summary_markdown(summary: dict, baseline: dict | None, *, tag: str, commit: str, base_url: str, category: str = "positive") -> str`, and the CLI `python scripts/render_eval_summary.py <run_dir> --tag T --commit C --base-url U` which prints markdown to stdout. Task 4 pipes that into `$GITHUB_STEP_SUMMARY`.

Reference shapes, taken from the live code — a run's `summary.json` has `by_category.<cat>` built by `_persona_stats` (`evals/report.py:299`), giving `count`, `judged_count`, `mean_<dim>`, `std_<dim>`, and `verdict_counts` with `pass`/`partial`/`fail` keys. The baseline at `evals/baselines/dev.json` is `{"source_run_id", "category", "means": {<dim>: float}, "fail_verdicts": int}`.

- [ ] **Step 1: Write the failing test**

Create `evals/tests/test_summary_md.py`:

```python
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


def test_missing_dimension_does_not_crash():
    summary = _summary()
    del summary["by_category"]["positive"]["mean_coherence"]
    out = _render(summary)
    assert "n/a" in out


def test_tool_use_appropriateness_is_not_reported():
    # _persona_stats emits it; the gate ignores it, so the report must too.
    out = _render(_summary(mean_tool_use_appropriateness=4.0))
    assert "tool_use_appropriateness" not in out
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd evals && pytest tests/test_summary_md.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'evals.summary_md'`

- [ ] **Step 3: Write the implementation**

Create `evals/summary_md.py`:

```python
"""Render an eval run's summary.json as GitHub step-summary markdown.

The release pipeline puts this in front of the human approver -- it is the
report they read before deciding whether a tag ships to prod, and it is what
makes that approval a real review rather than a click on a green check.

Scores come from `by_category.positive`, matching scripts/check_eval_gate.py:
`overall` mixes in refusal personas that score goal_completion=1.0 by design.
"""

from __future__ import annotations

from typing import Any

# Exactly the dimensions scripts/check_eval_gate.py gates on. _persona_stats
# also emits tool_use_appropriateness, which the gate ignores.
DIMENSIONS = (
    "groundedness",
    "factfulness",
    "goal_completion",
    "persona_handling",
    "coherence",
)

GET_ENDPOINTS = (
    ("/docs", "Swagger UI -- interactive, drive the API by hand from here"),
    ("/openapi.json", "OpenAPI schema, for pointing your own client at it"),
    ("/version", "commit this build was made from"),
    ("/health", "component status (S3, BigQuery, Gemini)"),
    ("/financial/metadata", "available companies and statement coverage"),
)

POST_ENDPOINTS = (
    ("/chat/stream", "main agent endpoint -- what the eval personas exercise"),
    ("/fast_chat_v2", "financial-data fast path"),
    ("/chat", "basic non-streaming chat"),
)


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _delta(current: float | None, base: float | None) -> str:
    if current is None or base is None:
        return "n/a"
    diff = current - base
    marker = "🔻" if diff < 0 else "▲" if diff > 0 else "—"
    return f"{diff:+.2f} {marker}"


def render_summary_markdown(
    summary: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    tag: str,
    commit: str,
    base_url: str,
    category: str = "positive",
) -> str:
    """Build the release report. Never raises on a partial summary -- a run
    that failed halfway must still render something the approver can read."""
    stats = summary.get("by_category", {}).get(category, {})
    base_means = (baseline or {}).get("means", {})

    lines: list[str] = [
        f"# Release {tag}",
        "",
        f"Commit `{commit}`, confirmed as the build dev is serving.",
        "",
        f"Eval run `{summary.get('run_id', 'unknown')}` — category `{category}`, "
        f"{stats.get('judged_count', 0)} of {stats.get('count', 0)} conversations judged.",
        "",
    ]

    if baseline is None:
        lines += ["> **No baseline found.** Scores are shown without comparison.", ""]

    lines += [
        "| Dimension | Baseline | This run | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dim in DIMENSIONS:
        current = stats.get(f"mean_{dim}")
        base = base_means.get(dim)
        lines.append(f"| {dim} | {_fmt(base)} | {_fmt(current)} | {_delta(current, base)} |")

    verdicts = stats.get("verdict_counts", {})
    fail_now = verdicts.get("fail", 0)
    fail_base = (baseline or {}).get("fail_verdicts")
    fail_base_cell = "n/a" if fail_base is None else str(fail_base)
    fail_delta = "n/a" if fail_base is None else f"{fail_now - fail_base:+d}"
    lines += [
        f"| fail verdicts | {fail_base_cell} | {fail_now} | {fail_delta} |",
        "",
        f"Verdicts: **{verdicts.get('pass', 0)} pass**, "
        f"{verdicts.get('partial', 0)} partial, {verdicts.get('fail', 0)} fail.",
        "",
        "## Verify it yourself",
        "",
        f"This release is running on dev at `{base_url}`. Exercise it with your own "
        "client before approving — the scores above are a floor, not the whole picture.",
        "",
    ]
    for path, note in GET_ENDPOINTS:
        lines.append(f"- [`GET {path}`]({base_url}{path}) — {note}")
    for path, note in POST_ENDPOINTS:
        lines.append(f"- `POST {base_url}{path}` — {note}")

    lines += [
        "",
        "Full transcripts are in the run artifact attached to this workflow run. "
        "Download it and browse with `python evals/serve.py`.",
        "",
    ]
    return "\n".join(lines)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd evals && pytest tests/test_summary_md.py -v
```

Expected: PASS, 8 tests.

- [ ] **Step 5: Add the CLI wrapper**

Create `scripts/render_eval_summary.py`:

```python
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

from evals.summary_md import render_summary_markdown

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
```

- [ ] **Step 6: Verify the CLI runs end to end**

The `evals` package must be installed (`cd evals && pip install -e .`). Render an existing run if one is present, otherwise a synthetic one:

```bash
python -c "import json,pathlib; d=pathlib.Path('/tmp/r'); d.mkdir(exist_ok=True); (d/'summary.json').write_text(json.dumps({'run_id':'x','by_category':{'positive':{'count':1,'judged_count':1,'mean_groundedness':3.5,'verdict_counts':{'pass':1,'partial':0,'fail':0}}}}))" && python scripts/render_eval_summary.py /tmp/r --tag v2026.08.30 --commit abc123 --base-url http://dev.example
```

Expected: markdown printed to stdout containing the table, `http://dev.example/docs`, and exit code 0.

- [ ] **Step 7: Commit**

```bash
git add evals/summary_md.py evals/tests/test_summary_md.py scripts/render_eval_summary.py
git commit -m "feat(evals): render a release report for the human approver"
```

---

### Task 3: Reduce `deploy.yml` to a dev-only workflow

**Files:**
- Rename: `.github/workflows/deploy.yml` → `.github/workflows/deploy-dev.yml`
- Modify: the renamed file — delete the `deploy-prod` job (lines 136-181), rewrite the header comment, add the BUILD_SHA step

**Interfaces:**
- Consumes: `fastapi_app/app/BUILD_SHA` path from Task 1.
- Produces: a workflow whose successful run means both the dev deploy and the eval gate passed. Task 4 depends on dev being deployed by it.

- [ ] **Step 1: Rename the file**

```bash
git mv .github/workflows/deploy.yml .github/workflows/deploy-dev.yml
```

- [ ] **Step 2: Replace the header comment**

Replace lines 1-18 (from `name: Deploy` through the `workflow_dispatch: {}` block) with:

```yaml
name: Deploy to dev

# Trunk-based development: every merge to main lands on dev immediately, with
# no approval, so the team can keep merging. Prod is a separate pipeline
# triggered by a release tag -- see .github/workflows/release-prod.yml and
# docs/adr/0004-trunk-based-development.md.
#
# The eval gate stays here as a fast regression tripwire: it catches a bad
# merge on the day it lands rather than at release time a week later.
#
# One-time setup required before this workflow can run -- see
# docs/CI_CD_DEPLOY_PIPELINE.md for exact commands:
#   - An IAM role for GitHub OIDC (repo variable AWS_DEPLOY_ROLE_ARN)
#   - Secrets: COPILOT_API_MANIFEST, COPILOT_DEV_ENV_MANIFEST, GEMINI_API_KEY
#   - A seeded baseline at evals/baselines/dev.json

on:
  push:
    branches: [main]
  workflow_dispatch: {}
```

Note `COPILOT_PROD_ENV_MANIFEST` is dropped from this list — it moves to the release workflow.

- [ ] **Step 3: Add the build-stamp step**

In the `deploy-dev` job, insert immediately before the `copilot svc deploy --env dev` step:

```yaml
      # .dockerignore excludes .git/, so the container cannot derive its own
      # commit. Write it into the build context instead -- the Dockerfile's
      # COPY . . bakes it in, and GET /version reads it back. The release
      # pipeline uses that to prove dev is serving a given commit.
      - name: Stamp build commit
        run: echo "${{ github.sha }}" > fastapi_app/app/BUILD_SHA
```

- [ ] **Step 4: Delete the prod job**

Remove the entire `deploy-prod:` job (everything from `  deploy-prod:` to the end of the file). The file now ends after the `eval-gate` job's "Upload eval run artifacts" step.

- [ ] **Step 5: Validate the workflow parses**

```bash
python -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/deploy-dev.yml')); print(sorted(d['jobs'])); sys.exit(0 if sorted(d['jobs'])==['deploy-dev','eval-gate'] else 1)"
```

Expected: `['deploy-dev', 'eval-gate']` and exit code 0.

- [ ] **Step 6: Confirm no lingering prod references**

```bash
grep -n "prod" .github/workflows/deploy-dev.yml
```

Expected: no output.

- [ ] **Step 7: Commit**

```bash
git add .github/workflows/
git commit -m "refactor(cicd): reduce deploy.yml to a dev-only workflow"
```

---

### Task 4: `release-prod.yml` — tag-triggered release with three gates

**Files:**
- Create: `.github/workflows/release-prod.yml`

**Interfaces:**
- Consumes: `GET /version` from Task 1, `scripts/render_eval_summary.py` from Task 2, the dev deploy from Task 3.
- Produces: the release pipeline itself. Nothing depends on it.

- [ ] **Step 1: Create the workflow**

Create `.github/workflows/release-prod.yml`:

```yaml
name: Release to prod

# Trunk-based development: merges to main deploy to dev unattended
# (deploy-dev.yml). A release is a tag on main. Pushing a v* tag runs three
# gates in order -- dev is provably serving the tagged commit, the eval suite
# passes its regression gate, then a human approves after reading the report.
# Only the third gate exercises discretion; the first two are preconditions.
#
# See docs/adr/0004-trunk-based-development.md and
# docs/CI_CD_DEPLOY_PIPELINE.md.

on:
  push:
    tags: ['v*']

env:
  AWS_REGION: us-east-1
  COPILOT_APP: jse-datasphere-chatbot
  COPILOT_SVC: api
  COPILOT_VERSION: v1.34.1
  DEV_BASE_URL: ${{ vars.DEV_BASE_URL || 'http://jse-da-Publi-3w4oxbvTf5j0-374994583.us-east-1.elb.amazonaws.com' }}
  PROD_BASE_URL: ${{ vars.PROD_BASE_URL || 'http://jse-da-Publi-2tSlV7zf7Ysl-685234288.us-east-1.elb.amazonaws.com' }}

permissions:
  contents: read

# A constant group, not github.ref -- two releases must serialise rather than
# race on prod.
concurrency:
  group: release-prod
  cancel-in-progress: false

jobs:
  verify-tag:
    name: Verify dev is serving the tagged commit
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      # The eval suite below runs against dev. If dev is serving anything
      # other than the tagged commit, those results describe code that isn't
      # being released, and the human would approve on the wrong evidence.
      - name: Compare tag commit against dev /version
        run: |
          TAG_SHA="${{ github.sha }}"
          STATUS=$(curl -s -o /tmp/version.json -w '%{http_code}' "${{ env.DEV_BASE_URL }}/version" || echo 000)
          if [ "$STATUS" = "404" ]; then
            echo "::error::dev has no /version endpoint -- it predates this release pipeline. Merge to main, let dev redeploy, then re-tag."
            exit 1
          fi
          if [ "$STATUS" != "200" ]; then
            echo "::error::dev /version returned HTTP $STATUS -- cannot confirm what dev is running."
            exit 1
          fi
          DEV_SHA=$(python3 -c 'import json; print(json.load(open("/tmp/version.json")).get("commit",""))')
          echo "tag commit: $TAG_SHA"
          echo "dev commit: $DEV_SHA"
          if [ "$DEV_SHA" != "$TAG_SHA" ]; then
            echo "::error::dev is serving $DEV_SHA but tag ${{ github.ref_name }} points at $TAG_SHA. Tag HEAD of main, or wait for its dev deploy to finish."
            exit 1
          fi
          echo "dev is serving the tagged commit."

  release-eval:
    name: Release eval suite (hard gate)
    runs-on: ubuntu-latest
    needs: verify-tag
    timeout-minutes: 30
    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install eval suite
        run: |
          cd evals
          python -m pip install --upgrade pip
          pip install -e .
          # python-dotenv: a dependency of scripts/run_eval.py, not of the
          # evals package itself -- absent on a clean CI runner.
          pip install python-dotenv

      - name: Run eval suite against dev
        env:
          GEMINI_API_KEY: ${{ secrets.GEMINI_API_KEY }}
        run: |
          python scripts/run_eval.py \
            --base-url "${{ env.DEV_BASE_URL }}" \
            --replicates 1 \
            --concurrency 3 \
            --max-cost-usd 5 \
            --run-id "release-${{ github.run_id }}"

      # Rendered before the gate and with if: always(), so a FAILED gate is as
      # readable as a passing one. The human needs to see why a release stopped.
      - name: Render report for the approver
        if: always()
        run: |
          python scripts/render_eval_summary.py "evals/runs/release-${{ github.run_id }}" \
            --tag "${{ github.ref_name }}" \
            --commit "${{ github.sha }}" \
            --base-url "${{ env.DEV_BASE_URL }}" >> "$GITHUB_STEP_SUMMARY"

      # Hard gate: a regression stops the release before any human is asked.
      - name: Regression gate
        run: python scripts/check_eval_gate.py "evals/runs/release-${{ github.run_id }}"

      - name: Upload eval run artifacts
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: release-eval-${{ github.ref_name }}
          path: evals/runs/release-${{ github.run_id }}
          retention-days: 90

  deploy-prod:
    name: Deploy to prod
    runs-on: ubuntu-latest
    needs: release-eval
    timeout-minutes: 25
    # Required reviewers on this Environment are the human gate -- the only
    # gate in this pipeline that exercises discretion. The job does not start
    # until release-eval succeeds; it then waits for approval.
    environment: prod
    permissions:
      id-token: write
      contents: read
    steps:
      - uses: actions/checkout@v4

      - name: Configure AWS credentials (OIDC)
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ vars.AWS_DEPLOY_ROLE_ARN }}
          aws-region: ${{ env.AWS_REGION }}

      - name: Install AWS Copilot CLI
        run: |
          curl -sSL -o /usr/local/bin/copilot "https://github.com/aws/copilot-cli/releases/download/${{ env.COPILOT_VERSION }}/copilot-linux"
          chmod +x /usr/local/bin/copilot

      - name: Write gitignored Copilot manifests
        run: |
          mkdir -p fastapi_app/copilot/api fastapi_app/copilot/environments/prod
          cat > fastapi_app/copilot/api/manifest.yml <<'COPILOT_API_MANIFEST_EOF'
          ${{ secrets.COPILOT_API_MANIFEST }}
          COPILOT_API_MANIFEST_EOF
          cat > fastapi_app/copilot/environments/prod/manifest.yml <<'COPILOT_PROD_ENV_MANIFEST_EOF'
          ${{ secrets.COPILOT_PROD_ENV_MANIFEST }}
          COPILOT_PROD_ENV_MANIFEST_EOF

      - name: Stamp build commit
        run: echo "${{ github.sha }}" > fastapi_app/app/BUILD_SHA

      - name: copilot svc deploy --env prod
        working-directory: fastapi_app
        run: copilot svc deploy --name "${{ env.COPILOT_SVC }}" --env prod

      - name: Verify prod is serving the tagged commit
        run: |
          HEALTHY=0
          for i in $(seq 1 20); do
            if curl -fsS "${{ env.PROD_BASE_URL }}/health" >/dev/null; then HEALTHY=1; break; fi
            sleep 15
          done
          if [ "$HEALTHY" != "1" ]; then
            echo "::error::prod /health did not turn healthy in time"
            exit 1
          fi
          PROD_SHA=$(curl -fsS "${{ env.PROD_BASE_URL }}/version" | python3 -c 'import json,sys; print(json.load(sys.stdin).get("commit",""))')
          if [ "$PROD_SHA" != "${{ github.sha }}" ]; then
            echo "::error::prod is serving $PROD_SHA, expected ${{ github.sha }}"
            exit 1
          fi
          echo "prod is serving ${{ github.sha }} (${{ github.ref_name }})"
```

- [ ] **Step 2: Validate the workflow parses and has the right job graph**

```bash
python -c "import yaml,sys; d=yaml.safe_load(open('.github/workflows/release-prod.yml')); j=d['jobs']; assert sorted(j)==['deploy-prod','release-eval','verify-tag'], sorted(j); assert j['release-eval']['needs']=='verify-tag'; assert j['deploy-prod']['needs']=='release-eval'; assert j['deploy-prod']['environment']=='prod'; assert d['concurrency']['group']=='release-prod'; print('ok')"
```

Expected: `ok`

- [ ] **Step 3: Confirm the tag trigger**

```bash
python -c "import yaml; d=yaml.safe_load(open('.github/workflows/release-prod.yml')); print(d[True]['push'])"
```

Expected: `{'tags': ['v*']}` — note PyYAML parses the bare key `on` as boolean `True`, which is why the lookup uses `d[True]`.

- [ ] **Step 4: Commit**

```bash
git add .github/workflows/release-prod.yml
git commit -m "feat(cicd): release prod from a tag behind eval and human gates"
```

---

### Task 5: Document the strategy

**Files:**
- Modify: `docs/CI_CD_DEPLOY_PIPELINE.md` (header lines 1-6, section 2 at line 105, new release section)
- Create: `docs/adr/0004-trunk-based-development.md`

**Interfaces:**
- Consumes: everything above. Produces: nothing code depends on.

- [ ] **Step 1: Write the ADR**

Create `docs/adr/0004-trunk-based-development.md`, following the format of `docs/adr/0001-fix-event-loop-blocking-llm-calls.md` (`# ADR NNNN: <title>`, then `## Status`, `## Context`, `## Decision`, `## Consequences`):

```markdown
# ADR 0004: Trunk-based development with a tag-triggered, human-gated prod release

## Status

Accepted (2026-08-30). Implemented in `.github/workflows/deploy-dev.yml` and
`.github/workflows/release-prod.yml`.

## Context

Prod deploys ran in the same workflow run as the dev deploy: a push to `main`
deployed dev, ran the eval gate, then waited for a reviewer on the `prod`
GitHub Environment.

Two problems followed. Prod was always pinned to the newest commit on `main`,
so there was no way to ship a deliberate, reviewed cut. And the reviewer
approved a green check mark — no eval report was ever put in front of them,
because the viewer at `evals/viewer/` is local-only and no workflow published
anything from it.

The team wanted to merge freely and have that reach dev at once, then release
to prod weekly or on demand behind a human who had actually read the results.

## Decision

Trunk-based development, releasing from trunk by tag.

`main` is the trunk; short-lived branches merge into it via PR. Every merge
deploys to dev with no approval. A release is a CalVer tag (`vYYYY.MM.DD`) on
`main` HEAD, and pushing it runs three gates in order:

1. **Provenance** — dev's `GET /version` must report the tagged commit.
2. **Quality** — the eval suite runs against dev and `check_eval_gate.py` must
   pass. This is a hard gate; a regression stops the release before any human
   is asked.
3. **Judgement** — a required reviewer on the `prod` Environment approves,
   having read a rendered report of dimension scores, baseline deltas, fail
   counts, and a list of live dev endpoints to exercise by hand.

There are **no release branches and no back-merges**. A release branch that
merges back into trunk is GitFlow, and that merge-back is the specific practice
this strategy exists to remove. A hotfix is an ordinary PR to `main` followed
by a new tag; a rollback is a tag on the previous good commit.

## Consequences

Prod is decoupled from merge velocity: the team merges as fast as review
allows, and release cadence becomes an independent choice.

The eval suite runs twice for a released commit — once on merge, once at
release. This is deliberate. The merge run gives fast feedback; the release run
is generated against the exact build being shipped and is the artifact the
approver reads. At weekly cadence the extra cost is roughly one suite run per
week.

Build provenance became a requirement rather than a nicety. Because
`.dockerignore` excludes `.git/`, the container cannot derive its own commit, so
CI writes it to `app/BUILD_SHA` before the image is built and `GET /version`
reads it back. This also answers "what is prod actually running?" during
incidents, which was previously unanswerable.

A genuine regression blocks the release outright. The escape hatch is the
existing one: fix forward on trunk, or promote a reviewed new baseline with
`check_eval_gate.py --update-baseline`.
```

- [ ] **Step 2: Update the pipeline doc header**

In `docs/CI_CD_DEPLOY_PIPELINE.md`, replace lines 1-5 (the title, the one-line flow description, and the "None of the steps below" paragraph) with:

```markdown
# CI/CD Deploy Pipeline Setup

Covers one-time setup for the two deploy workflows. The project uses
trunk-based development — see [ADR 0004](adr/0004-trunk-based-development.md).

- [`.github/workflows/deploy-dev.yml`](../.github/workflows/deploy-dev.yml) —
  push to `main` → `copilot svc deploy --env dev` → eval suite against dev.
  No approval. Every merge lands on dev.
- [`.github/workflows/release-prod.yml`](../.github/workflows/release-prod.yml) —
  push a `v*` tag → verify dev is serving the tagged commit → eval suite +
  hard regression gate → human approval → `copilot svc deploy --env prod`.

See [Cutting a release](#cutting-a-release) below for the day-to-day procedure.

None of the steps below can be done by an agent — they touch IAM and repo/security settings. Run them yourself with the `ats-jse-elroy` SSO profile.
```

- [ ] **Step 3: Update section 2 (GitHub Environments)**

Replace the body of `## 2. GitHub Environments (approval gate + per-env config)` (the three lines from `Repo → **Settings → Environments**:` through the `**prod**` bullet) with:

```markdown
Repo → **Settings → Environments**:
- **`dev`** — no protection rules needed (auto-deploys on push to `main`).
- **`prod`** — add **Required reviewers** (yourself, at minimum). This gates the
  `deploy-prod` job of `release-prod.yml`, and it is the **only human gate in
  the pipeline**. It is reached only after `verify-tag` and `release-eval` have
  both passed, so an approver is never asked about a build that failed the
  regression gate.

There is deliberately **no ruleset restricting who may push `v*` tags** — the
reviewer requirement is the control. If repo write access ever widens beyond
people trusted to release, revisit that.

The OIDC trust policy in step 1 matches on `environment:dev` / `environment:prod`
and carries no branch or ref condition, so moving the prod deploy onto a tag
trigger needs **no IAM change**. Jobs that declare `environment: prod` get the
same `sub` claim whether they were started by a branch push or a tag push.
```

- [ ] **Step 4: Add a "Cutting a release" section**

Append a new section to `docs/CI_CD_DEPLOY_PIPELINE.md`:

````markdown
## Cutting a release

Releases come off `main` HEAD. Tagging an older commit is rejected, because dev
would no longer be serving that build and the eval results would describe code
you are not shipping.

```bash
git checkout main && git pull
git tag v2026.08.30
git push origin v2026.08.30
```

Use `-2`, `-3` suffixes for multiple releases in one day (`v2026.08.30-2`).

Pushing the tag starts `release-prod.yml`. Watch for:

1. **verify-tag** — fails if dev is not serving the tagged commit. Fix by
   waiting for the dev deploy to finish, or by tagging HEAD.
2. **release-eval** — runs the suite against dev. Read the **step summary** on
   the run page: dimension scores against baseline, fail-verdict delta, and
   live dev URLs to exercise by hand before approving. A regression fails this
   job and the release stops here.
3. **deploy-prod** — waits for your approval, then deploys and verifies prod's
   `/version` matches the tag.

**Hotfix:** an ordinary PR into `main`, then a new tag. No hotfix branch.

**Rollback:** tag the previous good commit. It ships through the same pipeline.
````

- [ ] **Step 5: Verify the docs reference real paths**

```bash
grep -c "release-prod.yml" docs/CI_CD_DEPLOY_PIPELINE.md && ls .github/workflows/release-prod.yml docs/adr/0004-trunk-based-development.md
```

Expected: a non-zero count and both files listed.

- [ ] **Step 6: Commit**

```bash
git add docs/CI_CD_DEPLOY_PIPELINE.md docs/adr/0004-trunk-based-development.md
git commit -m "docs(cicd): document trunk-based development and the release procedure"
```

---

## Post-merge validation

These cannot be run before merge — the pipeline only exists once `main` has it.

- [ ] Merge to `main`. Confirm `deploy-dev.yml` runs, dev redeploys, and
      `curl $DEV_BASE_URL/version` returns the merge commit rather than `unknown`.
- [ ] **Negative case first:** push a tag on a deliberately older commit and
      confirm `verify-tag` fails with the "dev is serving X but tag points at Y"
      message and nothing deploys. Delete the tag afterwards
      (`git push origin :refs/tags/<tag>`).
- [ ] Tag `main` HEAD. Confirm `verify-tag` passes, read the step summary,
      exercise `/docs` on dev by hand, approve, and confirm prod `/version`
      returns the tagged commit.

## Notes for the executor

- **Task order matters.** Task 4's workflow calls `/version` (Task 1) and
  `scripts/render_eval_summary.py` (Task 2). Do not reorder.
- **`main.py` is exempt from ruff and pre-commit** (`pyproject.toml`
  per-file-ignores, line 108). Match the surrounding style rather than
  reformatting it.
- **Do not put tests in `fastapi_app/tests/` root** — CI does not run them.
- **No IAM change is needed.** The OIDC trust policy
  (`docs/CI_CD_DEPLOY_PIPELINE.md` step 1) matches on
  `repo:Aeontsolutions/jse-datasphere-chatbot:environment:{dev,prod}` with no
  branch or ref condition, so a `deploy-prod` job started by a tag push gets the
  same `sub` claim it does today. Verified against the policy in that doc.
- **One deviation from the spec's file table:** the spec listed only
  `scripts/render_eval_summary.py`. The plan splits it into
  `evals/summary_md.py` (logic) plus a thin CLI, so the existing
  `eval-suite-unit-tests` job covers the rendering — `scripts/` has no test
  job of its own, and `scripts/check_eval_gate.py` is untested for exactly that
  reason. This mirrors `scripts/run_eval.py` over `evals.cli`.
- The `refactor` branch entry in `test.yml` and `lint.yml`, and the public
  `/docs` on prod, are both out of scope per the spec. Leave them alone.

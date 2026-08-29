# Trunk-Based Development and a Human-Gated Prod Release

Date: 2026-08-30
Status: Approved for planning

## Problem

Prod deploys today run in the same workflow run as the dev deploy.
`.github/workflows/deploy.yml` triggers on every push to `main`, deploys dev,
runs the eval gate, then waits for a reviewer on the `prod` GitHub Environment.

Two consequences follow. Prod is always tied to the newest commit on `main`, so
there is no way to ship a deliberate, reviewed cut of the codebase. And the
reviewer approves a green check mark: no eval report is put in front of them,
because the viewer at `evals/viewer/` is local-only and no workflow publishes
it.

The team wants to merge to `main` freely and have that reach dev immediately,
then release to prod weekly or on demand behind a human who has actually read
the eval results.

## Decision

Adopt trunk-based development with release-from-trunk by tag.

- `main` is the trunk. Short-lived branches merge into it via PR.
- Every merge to `main` deploys to dev automatically. No approval.
- A release is a tag on `main`. Pushing the tag starts the prod pipeline.
- Prod requires three things to pass, in order: dev is provably serving the
  tagged commit, the eval suite passes its regression gate, and a human
  approves after reading the report.

There are no release branches and no back-merges. A release branch that merges
back into trunk is GitFlow, and that merge-back is the specific practice this
strategy exists to remove.

## Branching model

| Rule | Detail |
| --- | --- |
| Trunk | `main`. Always releasable. |
| Feature work | Short-lived branch, PR into `main`. |
| Dev deploy | Automatic on merge. No gate. |
| Release | CalVer tag on `main` HEAD: `vYYYY.MM.DD`, suffixed `-2`, `-3` for multiple releases in one day. |
| Hotfix | An ordinary PR into `main`, then a new tag. No hotfix branch. |
| Rollback | Tag the previous good commit. It redeploys through the same pipeline. |

Tags are cut from `main` HEAD. Tagging an older commit is rejected by
`verify-tag` below, because dev would no longer be serving that build and the
eval results would describe the wrong code.

## Pipeline architecture

`deploy.yml` splits into two workflows.

### `deploy-dev.yml`

The current file with the `deploy-prod` job deleted. Triggers, jobs, and steps
are otherwise unchanged: `push: branches: [main]` and `workflow_dispatch`,
running `deploy-dev` then `eval-gate`.

The eval gate stays here deliberately. It is a fast regression tripwire that
catches a bad merge on the day it lands rather than at release time a week
later, and it is cheap relative to the cost of discovering the regression
during a release.

### `release-prod.yml`

New. Triggers on `push: tags: ['v*']`. Concurrency group is the constant
`release-prod`, so two releases serialise rather than racing on prod.

**Job `verify-tag`** — no environment, no approval. Fetches `/version` from
`vars.DEV_BASE_URL` (same variable and fallback the existing workflow uses) and
compares the returned commit to the tag's commit. Fails with an explicit
message naming both values when they differ. Requires no special permissions.

Bootstrap edge case: `/version` does not exist until this change is deployed to
dev. That happens automatically when this work merges to `main`, so the first
release after the merge is fine — but a 404 must fail with a message saying dev
predates `/version`, not with a confusing SHA mismatch.

**Job `release-eval`** — `needs: verify-tag`. Installs the eval suite and runs
`scripts/run_eval.py` against dev, then `scripts/check_eval_gate.py`. **The
gate is hard: a regression against `evals/baselines/dev.json` fails the job and
the release stops before any human is asked.** Regardless of outcome the job
renders a step summary and uploads the run directory as an artifact, so a
failed gate is as readable as a passing one.

**Job `deploy-prod`** — `needs: release-eval`, `environment: prod` with required
reviewers. Steps are lifted unchanged from today's `deploy-prod`: OIDC auth,
Copilot install, write gitignored manifests from secrets, `copilot svc deploy
--env prod`, verify prod `/health`.

The three gates are therefore: mechanical provenance check, mechanical quality
check, then human judgement. The human is the only gate that exercises
discretion; the other two are pass/fail preconditions.

## Build provenance

Nothing today records which commit a deployed environment is running. `/health`
returns component status only, and `version="1.2.0"` at
`fastapi_app/app/main.py:139` is hardcoded.

Because `.dockerignore` excludes `.git/`, the running container cannot derive
its own commit. The SHA is therefore written into the build context before the
image is built:

1. Both deploy workflows write `$GITHUB_SHA` to `fastapi_app/app/BUILD_SHA`
   immediately before `copilot svc deploy`. The Dockerfile's `COPY . .` bakes it
   into the image. No pattern in `.dockerignore` matches an extensionless
   `BUILD_SHA`, and no Copilot manifest or repo secret needs editing.
2. `fastapi_app/app/BUILD_SHA` is added to `.gitignore`.
3. The app reads it once at startup from `Path(__file__).parent / "BUILD_SHA"`,
   falling back to `"unknown"` when absent, which is the normal case for local
   development.

A new `GET /version` endpoint returns `{"commit": ..., "version": "1.2.0"}`. It
is deliberately separate from `/health`: `/health` performs live S3, BigQuery,
and Gemini checks and returns 503 when any is unhealthy, so it cannot be relied
on to report provenance at the moment provenance matters most. `/version`
touches nothing and always returns 200. The same `commit` field is also added to
the `/health` payload, where operators will look for it first.

## Human review surface

`release-eval` writes a GitHub step summary. This is what the approver reads
before deciding, and it is the substance of the human-eval requirement.

Contents:

- Release tag and commit, confirmed as the build dev is serving.
- Mean scores per judge dimension (groundedness, factfulness, goal_completion,
  persona_handling, coherence) for `by_category.positive`, each shown against
  the baseline with its delta. `by_category.positive` is used rather than
  `overall`, which mixes in refusal personas that score `goal_completion=1.0`
  by design.
- Fail-verdict count against the baseline's.
- A **Verify it yourself** block with live dev URLs, so the approver can
  exercise the build with their own client rather than trusting the numbers
  alone: `/docs` (Swagger UI, interactive, already enabled), `/openapi.json`,
  `POST /chat/stream`, `POST /fast_chat_v2`, `POST /chat`,
  `GET /financial/metadata`, `GET /health`, `GET /version`.
- A link to the uploaded run artifact, which contains the full transcripts and
  can be opened locally with `python evals/serve.py`.

## Files touched

| Path | Change |
| --- | --- |
| `.github/workflows/deploy.yml` | Renamed to `deploy-dev.yml`; `deploy-prod` job removed; BUILD_SHA step added. |
| `.github/workflows/release-prod.yml` | New. |
| `fastapi_app/app/main.py` | Read `BUILD_SHA` at startup; add `GET /version`; add `commit` to `/health`. |
| `.gitignore` | Ignore `fastapi_app/app/BUILD_SHA`. |
| `scripts/render_eval_summary.py` | New. Renders a run's `summary.json` and the baseline into step-summary markdown. |
| `docs/CI_CD_DEPLOY_PIPELINE.md` | Rewrite the flow description and Environments section for two workflows; document the release procedure. |
| `docs/adr/0004-trunk-based-development.md` | New ADR recording the strategy. |

## Testing

- Unit tests for `render_eval_summary.py` against a recorded `summary.json`
  fixture, covering improvement, regression, and missing-baseline cases. The
  existing eval fixtures under `evals/tests/fixtures/` are the model to follow.
- Unit test for the BUILD_SHA read path: present, absent, and trailing
  whitespace.
- Endpoint test for `GET /version` alongside the existing endpoint tests.
- End-to-end validation is a real release: tag a commit, confirm `verify-tag`
  passes, read the step summary, approve, confirm prod `/version` returns the
  tagged commit. This is the acceptance criterion for the change and cannot be
  simulated in unit tests.
- Negative validation: push a tag on a deliberately stale commit and confirm
  `verify-tag` fails with a readable message rather than deploying.

## Out of scope

- `/docs` and `/openapi.json` are publicly reachable on both dev and prod ALBs.
  Fine on dev, and it is what makes manual verification easy, but the prod
  exposure deserves its own review. Separate ticket.
- `test.yml` and `lint.yml` trigger on `branches: [main, refactor]`. The
  long-lived `refactor` branch is contrary to this strategy and the entry is
  probably vestigial, but removing it is not required for this change.
- No GitHub ruleset restricting who may push `v*` tags. The prod Environment's
  required reviewers are the single human gate, by decision.

## Consequences

Prod is decoupled from merge velocity: the team merges as fast as review allows,
and release cadence becomes an independent choice. The cost is that a release
now requires someone to read a report, which is the intent.

The eval suite runs twice for a released commit: once on merge, once at release.
This is deliberate duplication. The merge run gives fast feedback; the release
run is generated against the exact build being shipped and is the artifact the
approver reads. At weekly cadence the additional cost is roughly one suite run
per week.

Because the gate is hard, a genuine regression blocks the release outright. The
escape hatch is the existing one: fix forward on trunk, or promote a reviewed
new baseline with `check_eval_gate.py --update-baseline`.

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
   is asked. Provenance is re-checked after the suite finishes as well: the
   suite runs for 10–25 minutes and `deploy-dev.yml` deploys on its own
   concurrency group, so a merge to `main` mid-suite would otherwise leave the
   later personas exercising a build that is not the one being released, with
   nothing recording it.
3. **Judgement** — a required reviewer on the `prod` Environment approves,
   having read a rendered report of dimension scores, baseline deltas, fail
   counts, and a list of live dev endpoints to exercise by hand.

There are **no release branches and no back-merges**. A release branch that
merges back into trunk is GitFlow, and that merge-back is the specific practice
this strategy exists to remove. A hotfix is an ordinary PR to `main` followed
by a new tag.

A rollback is deliberately **not** a tag on the previous good commit: gate 1
rejects that, because dev is serving trunk's HEAD and not that commit, so the
release would stop at the first gate in the middle of an incident. Rollback is
two moves instead — `copilot svc rollback --name api --env prod` returns the
prod service to its previous task definition immediately, outside this
pipeline; then a revert commit on trunk, tagged once dev has redeployed it,
ships the fix through the normal gates so that it is evaluated and reviewed
like any other release. The runbook in
[docs/CI_CD_DEPLOY_PIPELINE.md](../CI_CD_DEPLOY_PIPELINE.md) has the commands.

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

That guarantee has a real limit: it proves *source* identity, not *artifact*
identity. `copilot svc deploy` rebuilds the container image from the tagged
commit — it does not promote the image dev was evaluated against. Because
`fastapi_app/requirements.txt` is almost entirely unpinned (`>=` floors), the
Docker base image is a floating `python:3.11-slim`, and the `apt-get install`
step names no versions, a build kicked off after a long approval wait can
resolve a different dependency closure than the one `release-eval` scored.
Neither prod's `GET /version` nor `BUILD_SHA` can catch this, because both
record the commit that was built, not the artifact that resulted. This is a
known limitation of the decision as it stands today, not a problem the
pipeline already solves: the durable fix is pinning dependencies end-to-end
(requirements, base image, apt packages) so a rebuild is reproducible, or
changing the release mechanism to promote dev's already-built image by digest
instead of rebuilding from source. Neither is implemented yet.

A genuine regression blocks the release outright. The escape hatch is the
existing one: fix forward on trunk, or promote a reviewed new baseline with
`check_eval_gate.py --update-baseline`.

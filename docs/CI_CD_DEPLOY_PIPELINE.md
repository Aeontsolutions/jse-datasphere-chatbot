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

## 1. IAM role for GitHub OIDC

A GitHub OIDC provider already exists in this account (`arn:aws:iam::925030480327:oidc-provider/token.actions.githubusercontent.com`) — no need to create one. Create a role that trusts it, scoped to this repo.

**Important:** the `deploy-dev` and `deploy-prod` jobs each declare `environment: dev` / `environment: prod`. GitHub sets the OIDC token's `sub` claim to `repo:<owner>/<repo>:environment:<name>` for jobs that declare an environment (not the branch ref) — the trust policy below matches on that.

```bash
export AWS_PROFILE=ats-jse-elroy AWS_DEFAULT_REGION=us-east-1

cat > /tmp/gha-trust-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Federated": "arn:aws:iam::925030480327:oidc-provider/token.actions.githubusercontent.com" },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": { "token.actions.githubusercontent.com:aud": "sts.amazonaws.com" },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": [
          "repo:Aeontsolutions/jse-datasphere-chatbot:environment:dev",
          "repo:Aeontsolutions/jse-datasphere-chatbot:environment:prod"
        ]
      }
    }
  }]
}
EOF

aws iam create-role \
  --role-name jse-datasphere-chatbot-gha-deploy \
  --assume-role-policy-document file:///tmp/gha-trust-policy.json \
  --description "GitHub Actions OIDC role for jse-datasphere-chatbot deploys"
```

Attach a permissions policy. `copilot svc deploy` drives a CloudFormation change set that touches ECS, ECR, SSM, logs, ELB target groups, and IAM role passing — this is a **scoped starting point**, not guaranteed complete. Expect an `AccessDenied` on the first real run; add the missing action (CloudTrail will name it) rather than widening broadly.

```bash
cat > /tmp/gha-deploy-policy.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    { "Sid": "AccountWideReadOnly", "Effect": "Allow", "Action": [
        "sts:GetCallerIdentity", "ecr:GetAuthorizationToken", "ecs:DescribeClusters",
        "cloudformation:DescribeStacks", "cloudformation:DescribeStackEvents",
        "cloudformation:ListStacks", "elasticloadbalancing:DescribeLoadBalancers",
        "elasticloadbalancing:DescribeTags", "iam:ListRoles"
      ], "Resource": "*" },
    { "Sid": "CloudFormationStack", "Effect": "Allow", "Action": [
        "cloudformation:CreateChangeSet", "cloudformation:DescribeChangeSet",
        "cloudformation:ExecuteChangeSet", "cloudformation:DeleteChangeSet",
        "cloudformation:GetTemplate", "cloudformation:GetTemplateSummary",
        "cloudformation:ListStackResources", "cloudformation:UpdateStack"
      ], "Resource": "arn:aws:cloudformation:us-east-1:925030480327:stack/jse-datasphere-chatbot-*/*" },
    { "Sid": "ECR", "Effect": "Allow", "Action": [
        "ecr:BatchCheckLayerAvailability", "ecr:GetDownloadUrlForLayer", "ecr:BatchGetImage",
        "ecr:PutImage", "ecr:InitiateLayerUpload", "ecr:UploadLayerPart", "ecr:CompleteLayerUpload",
        "ecr:DescribeRepositories", "ecr:DescribeImages"
      ], "Resource": "arn:aws:ecr:us-east-1:925030480327:repository/jse-datasphere-chatbot/*" },
    { "Sid": "ECS", "Effect": "Allow", "Action": [
        "ecs:DescribeServices", "ecs:UpdateService", "ecs:RegisterTaskDefinition",
        "ecs:DescribeTaskDefinition", "ecs:DeregisterTaskDefinition", "ecs:ListTasks",
        "ecs:DescribeTasks", "ecs:TagResource"
      ], "Resource": "*", "Condition": { "StringLike": { "ecs:cluster": "arn:aws:ecs:us-east-1:925030480327:cluster/jse-datasphere-chatbot-*" } } },
    { "Sid": "SSMSecrets", "Effect": "Allow", "Action": ["ssm:GetParameter", "ssm:GetParameters", "ssm:GetParametersByPath"],
      "Resource": [
        "arn:aws:ssm:us-east-1:925030480327:parameter/copilot/jse-datasphere-chatbot/*",
        "arn:aws:ssm:us-east-1:925030480327:parameter/copilot/applications/jse-datasphere-chatbot*"
      ] },
    { "Sid": "Logs", "Effect": "Allow", "Action": [
        "logs:CreateLogGroup", "logs:PutRetentionPolicy", "logs:DescribeLogGroups", "logs:TagResource"
      ], "Resource": "arn:aws:logs:us-east-1:925030480327:log-group:/copilot/jse-datasphere-chatbot-*" },
    { "Sid": "PassRole", "Effect": "Allow", "Action": "iam:PassRole",
      "Resource": "arn:aws:iam::925030480327:role/jse-datasphere-chatbot-*" },
    { "Sid": "AssumeCopilotEnvRoles", "Effect": "Allow", "Action": "sts:AssumeRole",
      "Resource": [
        "arn:aws:iam::925030480327:role/jse-datasphere-chatbot-*-EnvManagerRole",
        "arn:aws:iam::925030480327:role/jse-datasphere-chatbot-*-CFNExecutionRole"
      ] },
    { "Sid": "AppStackSet", "Effect": "Allow", "Action": [
        "cloudformation:DescribeStackSet", "cloudformation:ListStackInstances",
        "cloudformation:ListStackSetOperations", "cloudformation:DescribeStackSetOperation"
      ], "Resource": "arn:aws:cloudformation:us-east-1:925030480327:stackset/jse-datasphere-chatbot-infrastructure:*" }
  ]
}
EOF

aws iam put-role-policy \
  --role-name jse-datasphere-chatbot-gha-deploy \
  --policy-name deploy-scoped \
  --policy-document file:///tmp/gha-deploy-policy.json

aws iam get-role --role-name jse-datasphere-chatbot-gha-deploy --query 'Role.Arn' --output text
rm /tmp/gha-trust-policy.json /tmp/gha-deploy-policy.json
```

Save the printed ARN — it's `AWS_DEPLOY_ROLE_ARN` in step 3.

## 2. GitHub Environments (approval gate + per-env config)

Repo → **Settings → Environments**:
- **`dev`** — no protection rules needed (auto-deploys on push to `main`).
- **`prod`** — add **Required reviewers** (yourself, at minimum). This gates the
  `deploy-prod` job of `release-prod.yml`, and it is the **only human gate in
  the pipeline**. It is reached only after `verify-tag` and `release-eval` have
  both passed, so an approver is never asked about a build that failed the
  regression gate.

**Without a required reviewer configured, the `prod` Environment enforces
nothing.** GitHub Environments have no default protection — an environment
with no rules attached lets any job that declares it run immediately. Skip
this step and `deploy-prod` deploys to prod unattended the moment
`release-eval` passes; there is no other approval step anywhere in this
pipeline.

**Set `prod`'s "Deployment branches and tags" to allow the release tag.** The
same Environment settings page carries a second rule that is easy to miss.
`deploy-prod` runs on `refs/tags/v2026.08.30`, not on a branch, so either leave
this set to **All branches** or choose **Selected branches and tags** and add a
**tag** rule with the pattern `v*`. If `prod` is left on "Selected branches"
with only `main` allowed — the natural state for an environment that was only
ever deployed from `main`, which is what this repo had before the release
pipeline — GitHub **refuses the job outright, before any reviewer is
prompted**: the run fails with a branch/tag protection-rules error, no approval
request is sent to anyone, and nothing in the message points at the tag
pattern. `verify-tag` and `release-eval` will both have passed by then, so the
release dies at the last step for a reason that looks nothing like its cause.
This is the likeliest way a first release stalls.

There is deliberately **no ruleset restricting who may push `v*` tags** — the
reviewer requirement is the control. If repo write access ever widens beyond
people trusted to release, revisit that.

The OIDC trust policy in step 1 matches on `environment:dev` / `environment:prod`
and carries no branch or ref condition, so moving the prod deploy onto a tag
trigger needs **no IAM change**. Jobs that declare `environment: prod` get the
same `sub` claim whether they were started by a branch push or a tag push.

## 3. Repo secrets and variables

Repo → **Settings → Secrets and variables → Actions**.

**Variables** (not secret — an OIDC role ARN and public ALB hostnames aren't credentials):
| Name | Value |
|---|---|
| `AWS_DEPLOY_ROLE_ARN` | ARN from step 1 |
| `DEV_BASE_URL` *(optional)* | `http://jse-da-Publi-3w4oxbvTf5j0-374994583.us-east-1.elb.amazonaws.com` — workflow already defaults to this; only add if the ALB is re-provisioned |
| `PROD_BASE_URL` *(optional)* | `http://jse-da-Publi-2tSlV7zf7Ysl-685234288.us-east-1.elb.amazonaws.com` — same |

**Secrets**:
| Name | Value |
|---|---|
| `GEMINI_API_KEY` | A Gemini API key for the eval suite's persona actor + LLM judge (separate from the app's own runtime key in SSM) |
| `COPILOT_API_MANIFEST` | Full contents of your working `fastapi_app/copilot/api/manifest.yml` |
| `COPILOT_DEV_ENV_MANIFEST` | Full contents of `fastapi_app/copilot/environments/dev/manifest.yml` |
| `COPILOT_PROD_ENV_MANIFEST` | Full contents of `fastapi_app/copilot/environments/prod/manifest.yml` |

Set the manifest secrets from your local working copy (see [[deploy-to-aws-copilot]] if you need to fetch them from the main checkout first):

```bash
gh secret set COPILOT_API_MANIFEST < fastapi_app/copilot/api/manifest.yml
gh secret set COPILOT_DEV_ENV_MANIFEST < fastapi_app/copilot/environments/dev/manifest.yml
gh secret set COPILOT_PROD_ENV_MANIFEST < fastapi_app/copilot/environments/prod/manifest.yml
gh secret set GEMINI_API_KEY   # paste the key at the prompt
gh variable set AWS_DEPLOY_ROLE_ARN --body "arn:aws:iam::925030480327:role/jse-datasphere-chatbot-gha-deploy"
```

**Keeping the manifest secrets in sync:** these are a snapshot, not a live reference to your working tree. If you hand-edit `manifest.yml` locally (e.g. bumping `cpu`/`memory`), re-run the matching `gh secret set` command or the pipeline will deploy the old config. This is the same manifest-drift risk noted in [[project_aws_account_migration]] — the pipeline doesn't fix it, it just relocates "the one place manifests live" from a laptop's working tree to GitHub secrets.

## 4. Seed the regression baseline (done — 2026-08-26)

`scripts/check_eval_gate.py` gates on regression vs. a baseline, not an absolute score — there's no principled absolute cutoff for this judge's 1–5 scale yet.

```bash
python scripts/run_eval.py --base-url http://jse-da-Publi-3w4oxbvTf5j0-374994583.us-east-1.elb.amazonaws.com --replicates 1 --run-id baseline_seed
python scripts/check_eval_gate.py evals/runs/baseline_seed --update-baseline
git add evals/baselines/dev.json
git commit -m "chore(evals): seed dev regression baseline"
```

`evals/baselines/dev.json` is now committed. Without it, every `eval-gate` run fails immediately with a clear "no baseline" error rather than silently passing.

**Important — the seed run had 10/26 `fail` verdicts (positive category), not 0.** Most trace to known data-coverage gaps, not app bugs: e.g. `investor_compare_ncb_vs_jmmb` and `senior_analyst_ncb_financials` ask for NCBFG years the DB doesn't have (see [[reference_dev_table_net_profit_years]]), and `technical_trader_market_data` asks for trading-volume/market-cap fields that were never in this data model. Because of this, `check_eval_gate.py`'s fail-verdict check is **relative to the baseline's own fail count** (`--max-fail-increase`, default tolerance 2), not an absolute cap of 0 — an absolute-zero cap would have failed every future run, including a perfect no-op deploy, since today's floor already has 10.

Re-baseline deliberately after any change that legitimately shifts scores (a reviewed prompt fix, new data coverage) — re-run `--update-baseline` by hand and commit the new file with a commit message explaining why. Never wire `--update-baseline` into the pipeline itself — that would let a slow score decline become the new normal one green build at a time.

## Notes / things worth revisiting later

- The IAM policy in step 1 is a starting point; tighten or extend it as real deploys reveal what's actually needed.
- **New finding from the baseline seed run, unrelated to this pipeline:** several judge-flagged `hallucination` moments where the bot treats 2025/2026 events as futuristic and fabricates details (a nonexistent "2025 Annual Report," a fabricated hurricane) — e.g. `analyst_finds_latest_annual_report`, `diaspora_investor_jse_access`, `esg_conscious_investor` in `evals/runs/baseline_seed/`. Looks like a temporal-grounding bug (the model not being told "today" is actually in the past relative to these dates), separate from anything this pipeline fixes. Worth its own investigation.
- `check_eval_gate.py` doesn't check the `record_count: 10557` red flag from [[reference_eval_suite]] (financial tool dumping the whole table) — a real regression could hide behind passing scores if the judge doesn't penalize it. Worth adding if it recurs. (Not seen in the baseline seed run.)
- `eval-gate` costs ~$0.35 and ~10 min per run (26 personas, `--replicates 1`) — cheap enough to run on every push to `main`, per [[reference_eval_suite]].

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

1. **verify-tag** — fails if dev is not serving the tagged commit, usually
   because dev's deploy of that commit has not finished yet. Wait for
   `deploy-dev.yml` to go green, then hit **Re-run jobs** on the failed release
   run in the Actions UI. **Re-pushing the same tag does nothing** — an
   unchanged tag is not a new ref, GitHub emits no `push` event, and no run
   starts; pushing and seeing nothing happen is the expected behaviour, not a
   broken pipeline. If you actually need the tag on a *different* commit, move
   it explicitly with `git tag -f v2026.08.30 <commit> && git push -f origin
   v2026.08.30`.
2. **release-eval** — runs the suite against dev. Read the **step summary** on
   the run page: dimension scores against baseline, fail-verdict delta, and
   live dev URLs to exercise by hand before approving. A regression fails this
   job and the release stops here. The job also **re-checks dev's `/version`
   after the suite finishes**: the suite takes 10–25 minutes, and any merge to
   `main` in that window redeploys dev underneath it. If dev moved, the job
   fails with "dev moved to … while the suite was running" — the results
   describe a different build, so the report is void. Re-run the workflow once
   dev is serving the tagged commit again.
3. **deploy-prod** — waits for your approval, then deploys and verifies prod's
   `/version` matches the tag.

**Only one release can be in flight, and only one more can be queued behind
it.** `release-prod.yml` serialises on a single concurrency group
(`release-prod`), but GitHub holds at most one *queued* run per group in
addition to the one in progress. If a release is running and a second tag is
pushed, the second waits; if a third tag is pushed before the second starts,
the third silently cancels the second in the queue — no error, no
notification, it just never runs. A `deploy-prod` job sitting on your approval
still counts as "in progress" for this purpose, so an unreviewed release holds
the group indefinitely and blocks every tag pushed after it from starting.
Approve or explicitly cancel a stuck release before relying on a later tag to
go through.

**Hotfix:** an ordinary PR into `main`, then a new tag. No hotfix branch.

### Rollback

**There is no `copilot svc rollback` command.** Copilot has never had one — the
`copilot svc` verbs are `init`, `ls`, `package`, `deploy`, `delete`, `show`,
`status`, `logs`, `exec`, `override`, `pause`, `resume`, and the feature request
for a rollback verb ([aws/copilot-cli#2519][i2519]) was closed without one; the
`copilot-cli` repo itself was archived in June 2026. What Copilot does give you
is automatic rollback when a **deploy itself fails**: CloudFormation reverts the
stack unless `--no-rollback` was passed, and this pipeline never passes it
([svc deploy docs][svcdeploy]). What it has no answer for is returning an
already-healthy service to its previous version — which is exactly the case you
are in when a release deployed cleanly and turned out to be bad.

**Do not tag an older commit either** — `verify-tag` rejects it, because dev is
serving trunk's HEAD and not that commit, so the release stops at the first gate.

Immediate recovery is therefore done against ECS directly, outside Copilot and
outside CI.

#### 1. Find the cluster, service, and current task definition

Copilot generates these names with hashes, so look them up rather than guessing:

```bash
export AWS_PROFILE=ats-jse-elroy AWS_DEFAULT_REGION=us-east-1

# Copilot's own view, to confirm you are looking at the right service:
cd fastapi_app && copilot svc status --name api --env prod

# The ECS names the AWS CLI needs (cluster is jse-datasphere-chatbot-prod-*):
aws ecs list-clusters
aws ecs list-services --cluster <cluster-arn-from-above>

# What prod is running right now -- the ARN ends in <family>:<revision>:
aws ecs describe-services --cluster <cluster> --services <service> \
  --query 'services[0].taskDefinition' --output text
```

#### 2a. If the bad deployment is still rolling out

ECS can roll an in-flight deployment back to the previous service revision
([Stopping Amazon ECS service deployments][stopdep]):

```bash
aws ecs list-service-deployments --cluster <cluster> --service <service>
aws ecs stop-service-deployment \
  --service-deployment-arn <serviceDeploymentArn-from-above> \
  --stop-type ROLLBACK
```

Console equivalent: ECS → the cluster → the service → **Deployments** → **Roll
back**. Eligible deployment states are `PENDING`, `IN_PROGRESS`,
`STOP_REQUESTED`, `ROLLBACK_REQUESTED`, and `ROLLBACK_IN_PROGRESS` — **not** a
deployment that has already completed. If the release finished cleanly and the
problem surfaced later, this is not your command; use 2b.

#### 2b. If the deployment already completed

Point the service back at the previous task definition revision. `--task-definition`
takes `family:revision` and "triggers a new service deployment"
([update-service docs][updsvc]):

```bash
# Newest revisions first, so the one below the current is the previous:
aws ecs list-task-definitions --family-prefix <family> --sort DESC

aws ecs update-service --cluster <cluster> --service <service> \
  --task-definition <family>:<previous-revision>

curl -s http://jse-da-Publi-2tSlV7zf7Ysl-685234288.us-east-1.elb.amazonaws.com/version
```

Confirm `/version` reports the commit you expected to land on.

**Two things are now true and both matter.** Prod is serving a commit that no
tag points at. And this edit is out-of-band: CloudFormation still believes the
stack describes the version you just backed out, so the next `copilot svc
deploy --env prod` will put it back. Immediate recovery buys time; it does not
close the incident.

#### 3. Durable fix — revert on trunk, then tag

`git revert` the bad commit into `main` via PR. That deploys to dev
automatically, and a new tag on the reverted HEAD ships it to prod through the
normal pipeline — so the rollback is itself evaluated by the eval suite and
reviewed by a human, and CloudFormation and the running service agree again.

```bash
git checkout main && git pull
git revert <bad-commit> && git push   # via PR, per the usual review rules
# wait for deploy-dev.yml to go green, then:
git tag v2026.08.31 && git push origin v2026.08.31
```

**This is not the incident-time answer.** It costs a full eval cycle (10–25
minutes of suite) plus a human approval that may not be sitting at a keyboard.
Do step 2 first, then this.

[i2519]: https://github.com/aws/copilot-cli/issues/2519
[svcdeploy]: https://aws.github.io/copilot-cli/docs/commands/svc-deploy/
[stopdep]: https://docs.aws.amazon.com/AmazonECS/latest/developerguide/stop-service-deployment.html
[updsvc]: https://docs.aws.amazon.com/cli/latest/reference/ecs/update-service.html

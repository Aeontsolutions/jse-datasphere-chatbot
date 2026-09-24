# ECR lifecycle policy for `jse-datasphere-chatbot/api`

`lifecycle-policy.json` is the source of truth for the image-retention policy on the
`jse-datasphere-chatbot/api` ECR repository (account 925030480327, us-east-1). It is applied
by hand; nothing deploys it automatically.

```bash
aws ecr put-lifecycle-policy --repository-name jse-datasphere-chatbot/api \
  --lifecycle-policy-text file://infra/ecr/lifecycle-policy.json --profile ats-jse-elroy
```

Preview first with `aws ecr start-lifecycle-policy-preview` (same arguments) and read the
result with `get-lifecycle-policy-preview`. Nothing is expired during a preview.

## Prod deploys must use a `prod-<sha>` tag

```powershell
$sha = git rev-parse --short HEAD
& "$env:USERPROFILE\bin\aws-copilot.exe" svc deploy --name api --env prod --tag "prod-$sha"
```

Copilot pins the ECS task definition to an image **digest**. ECR lifecycle rules act on
**tags**. If the image a service pins loses its tags, the rules can delete it while the
service is still running, and the next task restart fails to pull.

Rule 5 keeps the last five `prod-*` images. AWS documents that an image matching a
higher-priority rule cannot be expired by a lower-priority one, so `prod-*` images are also
shielded from rule 10 (untagged after 1 day), rule 20 (last 10 tagged) and rule 30 (90 days).
Anything not tagged `prod-*` gets no such protection.

## Incident this prevents (2026-09-23)

- **Sep 8, 10:44 UTC** - a hand-run `copilot svc deploy --env prod` pushed the image tagged
  only `latest`. Task definition `:14` pinned its digest.
- **Sep 8, 12:49 UTC** - CI pushed a new image and moved `latest` to it. Prod's digest was
  now untagged.
- **Sep 9, 12:42 UTC** - rule 10 expired it (`PolicyExecutionEvent` in CloudTrail).
- **Sep 9 - 23** - prod kept serving from the copy its running tasks had already pulled.
- **Sep 23, 12:54 UTC** - the last task was replaced. Every replacement failed with
  `CannotPullContainerError ... not found`. The ALB had no targets and returned 503, tripping
  `prod-alb-elb-5xx` and `prod-no-healthy-hosts`.

## Check that prod's pinned image is protected

```powershell
$img = aws ecs describe-task-definition --task-definition jse-datasphere-chatbot-prod-api `
  --query "taskDefinition.containerDefinitions[0].image" --output text
aws ecr describe-images --repository-name jse-datasphere-chatbot/api `
  --image-ids imageDigest=$($img.Split('@')[1]) --query "imageDetails[0].imageTags" --output text
```

The output must include a `prod-*` tag. An error or a `latest`-only result means the same
outage is queued up for the next task restart.

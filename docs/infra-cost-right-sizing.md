# Infra cost right-sizing: prod to 1 task, dev to 0

**Status:** applied 2026-10-13 via `aws ecs update-service`. Prod is at 1 task
(1/1 healthy, `/health` OK); dev is at 0 tasks. **Still to do:** set `count: 1` under
`environments.prod` in the gitignored `fastapi_app/copilot/api/manifest.yml` (main
checkout, line 92). Until then, the next prod `copilot svc deploy` will scale prod back to
2, because CloudFormation still records the old desired count.

**Rollback:** re-run the commands in "How to apply" with `--desired-count 2` (prod) and
`--desired-count 1` (dev), and set the manifest back to `count: 2`. Takes about 2 minutes;
nothing is deleted.

## Decision

1. **Prod:** run 1 task instead of 2 (`count: 2` becomes `count: 1`).
2. **Dev:** scale the service to 0 tasks while it is idle.
3. **Do not** add a scheduled overnight or weekend shutdown to prod.

Expected saving: about **$62/month** (about 15% of the $403 September 2026 account bill).

## Why

### Real usage is tiny

Source: the `interactions` BigQuery table (`jse_raw_financial_data_dev_elroy.interactions`),
`environment = 'prod'`, 2026-08-17 to 2026-10-12, Jamaica time (UTC-5). Excluded: the
2026-08-15/16 load test (1,003 requests in 7 minutes) and the recurring canned smoke
queries.

- 158 requests in 57 sessions over about 8 weeks, which is roughly one session a day.
  A session is a run of requests with gaps under 30 minutes. The table has no user ID.
- Two tasks give high availability that this traffic does not need.

### Usage is not confined to office hours

| Window | Share of real requests inside it |
|---|---|
| Mon-Fri 07:00-18:00 | 58% |
| Mon-Fri 06:00-21:00 | 75% |
| Above, plus Sat/Sun 09:00-22:00 | 93% |
| Daily 06:00-22:00 | 96% |

Evening and weekend sessions are retail-investor style questions (stock returns,
dividend yields, company news). Saturday had more requests than Monday and Thursday
combined. A weekday 9-to-5 shutdown would fail about four in ten real requests, so
we are not doing it. Overnight (23:00-05:00) is only about 4% of requests and would save
about $12/month, which is not worth a cold-start 503.

### Dev has had no real use since 2026-09-08

- The last dev interaction was 2026-09-08. Dev logged 1,402 requests in August and 2,275
  in September (10 active days in total), then none.
- The dev ALB still receives 150-1,200 requests a day. These are probes and health checks
  (`/health`, `/.env`, `/.git/config`), not chatbot calls. No chat requests appear in the
  last 48 hours of dev logs.
- ALB request counts are not a usage measure for either environment: prod's ALB has the
  same baseline noise. Use the interactions table.

## Savings (September 2026, list prices; cost-allocation tags are not active, so these are estimates)

| Change | Saves per month |
|---|---|
| Prod 2 tasks to 1 (1 vCPU / 2 GB each) | about $36 compute plus about $3.65 public IPv4 |
| Dev 1 task to 0 (0.5 vCPU / 1 GB) | about $18 compute plus about $3.65 public IPv4 |
| **Total** | **about $62** |

Not affected by this change: ALBs (about $16 each, four in the account), VPC public IPv4
on the load balancers, the site-to-site VPN ($36), RDS ($52), Lightsail ($43). Deleting the
dev environment entirely would save about $45/month but means rebuilding it to test a deploy.

## Risks

- **Prod single point of failure.** One task means a task crash or deploy causes a short
  outage until ECS replaces it. The `recreate` rolling deployment already has a brief gap
  during deploys, so this is not a new kind of exposure. Revert by setting the count back
  to 2.
- **Dev comes back on its own at the next deploy.** The CloudFormation stack owns the
  desired count, so a later `copilot svc deploy --env dev` should restore 1 task. Confirm
  on the first dev deploy after this change.
- **Estimates.** Dollar figures use list prices; confirm against Cost Explorer next month.
- **Data limits.** 158 requests is a small sample, and the 2026-09-10 to 2026-09-18 gap may
  reflect the ECR image-expiry outage rather than real quiet.

## How to apply

Prod (live users; the service has no autoscaling target):

```bash
export MSYS_NO_PATHCONV=1 AWS_PROFILE=ats-jse-elroy
aws ecs update-service --region us-east-1 \
  --cluster jse-datasphere-chatbot-prod-Cluster-6xeFgs2NAs0Q \
  --service jse-datasphere-chatbot-prod-api-Service-j609aKELOJ8V --desired-count 1
```

Then set `count: 1` under `environments.prod` in the gitignored
`fastapi_app/copilot/api/manifest.yml`, so the next deploy does not put it back to 2.

Dev:

```bash
aws ecs update-service --region us-east-1 \
  --cluster jse-datasphere-chatbot-dev-Cluster-vRpfe2GqbpMC \
  --service jse-datasphere-chatbot-dev-api-Service-LL9b2nxp9rSu --desired-count 0
```

Verify: prod shows 1 healthy target in its target group and a clean `/health`; dev shows
`desiredCount 0`. Re-check the September-to-October Cost Explorer ECS and public IPv4
lines after a full month.

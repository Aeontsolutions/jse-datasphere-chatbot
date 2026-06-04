# Terraform Migration Guide

Migration from AWS Copilot to Terraform. Follow these steps in order.
Third-party apps must not experience downtime during migration.

---

## Prerequisites

**On your Windows/WSL machine:**

```bash
# Install Terraform CLI in WSL
curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo gpg --dearmor -o /usr/share/keyrings/hashicorp.gpg
echo "deb [signed-by=/usr/share/keyrings/hashicorp.gpg] https://apt.releases.hashicorp.com $(lsb_release -cs) main" \
  | sudo tee /etc/apt/sources.list.d/hashicorp.list
sudo apt update && sudo apt install terraform

# Verify
terraform -version  # should be >= 1.9

# Install AWS CLI v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o awscliv2.zip
unzip awscliv2.zip && sudo ./aws/install

# Configure your AWS credentials (use the same profile as your Copilot setup)
aws configure
# or: export AWS_PROFILE=your-profile
```

---

## Step 1 — Add a custom domain in front of Copilot (if not already done)

> Skip this step if you already have a custom domain (e.g. `api.yourdomain.com`)
> pointing at your Copilot ALB. If you're using raw `*.elb.amazonaws.com` URLs,
> do this first — it decouples the public URL from the ALB resource.

1. **Request a certificate in ACM** (free):
   - AWS Console → Certificate Manager → Request → Public certificate
   - Domain: `api.yourdomain.com`
   - Validation: DNS (add the CNAME records it gives you to your DNS provider)
   - Wait for status to become **Issued** (usually < 5 min after DNS validates)

2. **Add an HTTPS listener to your existing Copilot ALB:**
   ```bash
   # Get your Copilot ALB ARN
   aws elbv2 describe-load-balancers \
     --query 'LoadBalancers[?contains(LoadBalancerName, `jse-datasphere`)].LoadBalancerArn' \
     --output text

   # Get the target group ARN from the existing HTTP listener
   aws elbv2 describe-listeners \
     --load-balancer-arn <ALB_ARN> \
     --query 'Listeners[?Port==`80`].DefaultActions[0].TargetGroupArn' \
     --output text

   # Add HTTPS listener pointing at the same target group
   aws elbv2 create-listener \
     --load-balancer-arn <ALB_ARN> \
     --protocol HTTPS \
     --port 443 \
     --certificates CertificateArn=<ACM_CERT_ARN> \
     --default-actions Type=forward,TargetGroupArn=<TARGET_GROUP_ARN>
   ```

3. **Create a DNS record** at your provider:
   - Type: `ALIAS` (Route 53) or `CNAME` (any other provider)
   - Name: `api.yourdomain.com`
   - Target: the Copilot ALB DNS name
   - TTL: **60 seconds** (low TTL makes cutover fast and reversible)

4. **Verify it works before continuing:**
   ```bash
   curl https://api.yourdomain.com/health
   # should return 200
   ```

Third-party apps should now use `api.yourdomain.com`. The underlying ALB can
change without affecting them.

---

## Step 2 — Run the bootstrap (one-time, local)

This creates the S3 state bucket, DynamoDB lock table, and ECR repository.

```bash
cd terraform/bootstrap

# Use local state — this is intentional (it manages the bucket for everything else)
terraform init
terraform plan
terraform apply

# Note the outputs — you will need them in the next step:
# ecr_repository_url = "123456789012.dkr.ecr.us-east-1.amazonaws.com/jse-datasphere-api"
# tfstate_bucket     = "jse-datasphere-tfstate"
# tfstate_lock_table = "jse-datasphere-tfstate-lock"
```

---

## Step 3 — Fill in the tfvars files

Edit `terraform/environments/*.tfvars` and replace all `REPLACE_WITH_*` placeholders:

| Placeholder | Where to find the value |
|---|---|
| `REPLACE_WITH_BOOTSTRAP_OUTPUT` | `ecr_repository_url` output from Step 2 |
| `REPLACE_WITH_YOUR_BUCKET_*` | Your existing S3 bucket names |
| `REPLACE_WITH_YOUR_GCP_PROJECT` | GCP Console → Project settings |
| `REPLACE_WITH_YOUR_DATASET/TABLE` | BigQuery Console |

For `prod.tfvars` specifically, also set:
```hcl
domain_name     = "api.yourdomain.com"          # from Step 1
certificate_arn = "arn:aws:acm:..."              # from Step 1
```

---

## Step 4 — Migrate SSM Parameter Store paths

Copilot stores secrets at:
```
/copilot/jse-datasphere-chatbot/<env>/secrets/<NAME>
```

Terraform reads them from:
```
/jse-datasphere/<env>/<NAME>
```

Copy existing secrets to the new paths (run once per environment):

```bash
ENV=staging  # repeat for prod, dev, test

for SECRET in GOOGLE_API_KEY GCP_SERVICE_ACCOUNT_INFO; do
  VALUE=$(aws ssm get-parameter \
    --name "/copilot/jse-datasphere-chatbot/$ENV/secrets/$SECRET" \
    --with-decryption \
    --query 'Parameter.Value' \
    --output text)

  aws ssm put-parameter \
    --name "/jse-datasphere/$ENV/$SECRET" \
    --value "$VALUE" \
    --type SecureString \
    --overwrite
done

echo "Done. Verify with:"
aws ssm get-parameters-by-path \
  --path "/jse-datasphere/$ENV/" \
  --with-decryption \
  --query 'Parameters[].Name'
```

---

## Step 5 — Deploy to staging first (local)

```bash
cd terraform

terraform init

# Preview what will be created
terraform plan -var-file=environments/staging.tfvars

# Create the staging infrastructure
terraform apply -var-file=environments/staging.tfvars

# Note the ALB DNS name from the output:
# alb_dns_name = "jse-datasphere-staging-xxxxxxx.us-east-1.elb.amazonaws.com"
```

Push your first Docker image to the new ECR so the ECS service can start:

```bash
# Authenticate Docker to ECR
aws ecr get-login-password --region us-east-1 \
  | docker login --username AWS \
    --password-stdin \
    123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build and push
docker build -t 123456789012.dkr.ecr.us-east-1.amazonaws.com/jse-datasphere-api:latest \
  ./fastapi_app
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/jse-datasphere-api:latest
```

Verify staging works on the raw ALB DNS before touching DNS:

```bash
curl http://<staging_alb_dns>/health
# should return 200
```

---

## Step 6 — Set up GitHub OIDC and variables

The workflows use **AWS OIDC** (OpenID Connect) instead of long-lived IAM access keys.
GitHub Actions assumes a short-lived IAM role directly — no stored credentials,
no rotation needed, no leakage risk.

### 6a — Create the OIDC provider in AWS (one-time per account)

```bash
# Check if the provider already exists first
aws iam list-open-id-connect-providers \
  --query 'OpenIDConnectProviderList[*].Arn' --output text

# Create it if not present
aws iam create-open-id-connect-provider \
  --url https://token.actions.githubusercontent.com \
  --client-id-list sts.amazonaws.com \
  --thumbprint-list 6938fd4d98bab03faadb97b34396831e3780aea1
```

### 6b — Create the IAM role GitHub Actions will assume

```bash
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
REPO="Aeontsolutions/jse-datasphere-chatbot"

# Trust policy: only this repo's main branch and PRs can assume the role
cat > /tmp/trust-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": {
      "Federated": "arn:aws:iam::${ACCOUNT_ID}:oidc-provider/token.actions.githubusercontent.com"
    },
    "Action": "sts:AssumeRoleWithWebIdentity",
    "Condition": {
      "StringEquals": {
        "token.actions.githubusercontent.com:aud": "sts.amazonaws.com"
      },
      "StringLike": {
        "token.actions.githubusercontent.com:sub": "repo:${REPO}:*"
      }
    }
  }]
}
EOF

aws iam create-role \
  --role-name jse-datasphere-github-actions \
  --assume-role-policy-document file:///tmp/trust-policy.json

# Attach the permissions Terraform + ECS deploys need
for POLICY in \
  AmazonEC2FullAccess \
  AmazonECS_FullAccess \
  ElasticLoadBalancingFullAccess \
  AmazonElastiCacheFullAccess \
  IAMFullAccess \
  AmazonSSMFullAccess \
  AmazonS3FullAccess \
  AmazonDynamoDBFullAccess \
  AmazonEC2ContainerRegistryFullAccess \
  CloudWatchLogsFullAccess; do
  aws iam attach-role-policy \
    --role-name jse-datasphere-github-actions \
    --policy-arn "arn:aws:iam::aws:policy/${POLICY}"
done

echo "Role ARN: arn:aws:iam::${ACCOUNT_ID}:role/jse-datasphere-github-actions"
```

> **Note on permissions:** The `*FullAccess` policies are a pragmatic starting point.
> Once the infrastructure is stable, tighten these to least-privilege using
> AWS IAM Access Analyzer to identify what's actually used.

### 6c — Add GitHub Variables

In your GitHub repo → **Settings → Secrets and variables → Actions → Variables**:

| Variable name | Value |
|---|---|
| `TF_AWS_ROLE_ARN` | `arn:aws:iam::ACCOUNT_ID:role/jse-datasphere-github-actions` |
| `ECR_REPOSITORY_NAME` | `jse-datasphere-api` |
| `STAGING_API_URL` | `https://staging.yourdomain.com` (or raw ALB DNS once deployed) |

No secrets needed — OIDC eliminates stored credentials entirely.

### 6d — Create GitHub Environments

In GitHub repo → **Settings → Environments**, create:
- `staging` — no protection rules (auto-deploys)
- `production` — add **Required reviewers**: yourself

---

## Step 7 — DNS cutover (zero-downtime)

Once staging is stable for at least 24 hours, cut over prod.

1. **Deploy prod infrastructure:**
   ```bash
   terraform apply -var-file=environments/prod.tfvars
   # Note the new ALB DNS: alb_dns_name = "jse-datasphere-prod-xxxxxxx...."
   ```

2. **Verify the new prod ALB works on its raw DNS before touching the domain:**
   ```bash
   curl http://<prod_alb_dns>/health
   # should return 200 — confirm your app is healthy on the new infra
   ```

3. **Update your DNS record** (the one you set up in Step 1):
   - Change the ALIAS/CNAME target from the **Copilot ALB** to the **new Terraform ALB**
   - TTL is already 60s, so propagation takes ≤ 1 minute globally

4. **Verify immediately after DNS update:**
   ```bash
   # Watch for the new ALB DNS to appear in resolution
   watch -n5 "dig +short api.yourdomain.com"

   # Once it resolves to the new ALB, test the endpoints
   curl https://api.yourdomain.com/health
   curl https://api.yourdomain.com/chroma/query  # or any endpoint 3rd parties use
   ```

5. **Monitor for 30 minutes.** If anything is wrong:
   ```bash
   # Rollback: point DNS back at Copilot ALB — takes effect in 60 seconds
   # (update the DNS record at your provider, or via aws route53 change-resource-record-sets)
   ```

---

## Step 8 — Decommission Copilot

Only after prod has been stable on Terraform for at least 48 hours.

```bash
# List Copilot environments to confirm what will be deleted
copilot env ls

# Delete services first, then environments
copilot svc delete --name api --env prod
copilot svc delete --name api --env staging
copilot env delete --name prod
copilot env delete --name staging

# Optionally delete the Copilot app itself
copilot app delete
```

> **Warning:** `copilot app delete` is irreversible. Keep the Copilot manifest
> files in the repo under `fastapi_app/copilot/examples/` for reference.

---

## WSL-specific notes

When working in WSL on Windows:
- Your AWS credentials file is at `~/.aws/credentials` inside WSL (not the Windows path)
- Docker Desktop for Windows shares its daemon with WSL2 — `docker` commands work in WSL terminal
- File paths in WSL use `/home/<user>/...` — avoid copying Windows-style paths (`C:\...`)
- If `terraform init` is slow, ensure you're working inside the WSL filesystem
  (`~/projects/...`), not on a mounted Windows drive (`/mnt/c/...`) — the latter
  is significantly slower for I/O-heavy operations like provider downloads

---

## Quick reference — local Terraform commands

```bash
cd terraform

# First time or after provider changes
terraform init

# Preview changes for one environment (safe, no changes made)
terraform plan -var-file=environments/staging.tfvars

# Apply changes
terraform apply -var-file=environments/staging.tfvars

# Show current state
terraform show

# Detect drift (what changed outside Terraform)
terraform plan -var-file=environments/prod.tfvars -detailed-exitcode

# Destroy an environment (use with caution)
terraform destroy -var-file=environments/test.tfvars
```

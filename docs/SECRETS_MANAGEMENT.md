# Secrets Management Guide

## Overview

This document describes how secrets and credentials are managed securely in the JSE DataSphere Chatbot application. **All sensitive credentials are stored in AWS Systems Manager Parameter Store** and injected at runtime by AWS Copilot.

## Security Principles

1. **Never commit secrets to version control** - All `.env` files and manifests with secrets are git ignored
2. **Use AWS Parameter Store for all environments** - Secrets are centrally managed and encrypted
3. **Least privilege access** - Only necessary IAM roles can access secrets
4. **Secret rotation** - Credentials can be rotated without code changes
5. **Audit trail** - All secret access is logged in CloudTrail

---

## Secrets Architecture

### Stored Secrets

The following secrets are stored in AWS Systems Manager Parameter Store:

| Secret Name | Description | Type |
|------------|-------------|------|
| `AWS_ACCESS_KEY_ID` | AWS credentials for S3 access | SecureString |
| `AWS_SECRET_ACCESS_KEY` | AWS secret key for S3 access | SecureString |
| `GOOGLE_API_KEY` | Google Gemini AI API key | SecureString |
| `GCP_SERVICE_ACCOUNT_INFO` | GCP service account JSON (contains private key) | SecureString |

### Parameter Store Paths

Secrets are stored with environment-specific paths:

```
/copilot/jse-datasphere-chatbot/<environment>/secrets/<SECRET_NAME>
```

**Environments:**
- `dev` - Development environment
- `staging` - Staging environment
- `prod` - Production environment
- `test` - Testing environment

**Example:**
```
/copilot/jse-datasphere-chatbot/prod/secrets/GOOGLE_API_KEY
/copilot/jse-datasphere-chatbot/dev/secrets/AWS_ACCESS_KEY_ID
```

---

## Local Development

### Setup

1. **Copy the environment template:**
   ```bash
   cp fastapi_app/.env.example fastapi_app/.env
   ```

2. **Request credentials from team lead** and populate `.env`:
   ```bash
   # AWS Configuration
   AWS_ACCESS_KEY_ID=your-aws-access-key
   AWS_SECRET_ACCESS_KEY=your-aws-secret-key
   AWS_DEFAULT_REGION=us-east-1
   DOCUMENT_METADATA_S3_BUCKET=jse-renamed-docs-copy

   # Google Cloud Configuration
   GOOGLE_API_KEY=your-google-api-key
   GCP_SERVICE_ACCOUNT_INFO='{"type":"service_account",...}'
   GCP_PROJECT_ID=jse-datasphere

   # BigQuery Configuration
   BIGQUERY_DATASET=jse_raw_financial_data_dev_elroy
   BIGQUERY_TABLE=multiyear_financial_data
   BIGQUERY_LOCATION=US
   ```

3. **Verify `.env` is git ignored:**
   ```bash
   git check-ignore fastapi_app/.env
   # Should output: fastapi_app/.env
   ```

### Pre-commit Hooks

Pre-commit hooks prevent accidental credential commits:

```bash
# Install pre-commit hooks
source venv/bin/activate
pip install pre-commit
pre-commit install

# Test the hooks
pre-commit run --all-files
```

The `detect-private-key` hook will block commits containing:
- AWS access keys
- Private keys (RSA, SSH, etc.)
- API keys in common formats

---

## Production Deployment

### How Secrets are Injected

AWS Copilot automatically injects secrets from Parameter Store at container startup:

1. **Copilot manifest** references secrets by path:
   ```yaml
   secrets:
     GOOGLE_API_KEY: /copilot/${COPILOT_APPLICATION_NAME}/${COPILOT_ENVIRONMENT_NAME}/secrets/GOOGLE_API_KEY
   ```

2. **Environment variables** are set from Parameter Store values before the container starts

3. **Application** reads secrets from environment variables (never from files)

### Deployment Workflow

```bash
# 1. Ensure secrets exist in Parameter Store (one-time setup)
AWS_PROFILE=jse-datasphere-elroy aws ssm get-parameter \
  --name "/copilot/jse-datasphere-chatbot/prod/secrets/GOOGLE_API_KEY"

# 2. Deploy with Copilot
AWS_PROFILE=jse-datasphere-elroy copilot svc deploy --name api --env prod

# 3. Copilot fetches secrets from Parameter Store and injects them
```

---

## Adding New Secrets

### Step 1: Add to Parameter Store

Use the AWS CLI with your SSO profile:

```bash
export AWS_PROFILE=jse-datasphere-elroy

# Create secret in all environments
for env in dev staging prod test; do
  aws ssm put-parameter \
    --name "/copilot/jse-datasphere-chatbot/$env/secrets/NEW_SECRET_NAME" \
    --value "secret-value-here" \
    --type "SecureString" \
    --description "Description of the secret" \
    --overwrite
done
```

### Step 2: Update Copilot Manifest

Add secret reference to `fastapi_app/copilot/api/manifest.yml`:

```yaml
secrets:
  # Existing secrets...
  NEW_SECRET_NAME: /copilot/${COPILOT_APPLICATION_NAME}/${COPILOT_ENVIRONMENT_NAME}/secrets/NEW_SECRET_NAME
```

### Step 3: Update Configuration

Add to `fastapi_app/app/config.py`:

```python
class NewServiceConfig(BaseSettings):
    secret_key: str = Field(..., description="New secret")

    model_config = SettingsConfigDict(
        env_prefix="NEW_SERVICE_",
        env_file=".env"
    )
```

### Step 4: Update .env.example

Document the new secret (with placeholder value):

```bash
# In .env.example
NEW_SECRET_NAME=your-secret-here
```

### Step 5: Deploy

```bash
AWS_PROFILE=jse-datasphere-elroy copilot svc deploy --name api --env <environment>
```

---

## Rotating Secrets

### Process

1. **Generate new credentials** in the service (AWS Console, Google Cloud Console, etc.)

2. **Update Parameter Store** with new values:
   ```bash
   export AWS_PROFILE=jse-datasphere-elroy

   aws ssm put-parameter \
     --name "/copilot/jse-datasphere-chatbot/prod/secrets/GOOGLE_API_KEY" \
     --value "new-api-key-value" \
     --type "SecureString" \
     --overwrite
   ```

3. **Redeploy the service** to pick up new secrets:
   ```bash
   AWS_PROFILE=jse-datasphere-elroy copilot svc deploy --name api --env prod
   ```

4. **Verify** the service is working with new credentials:
   ```bash
   curl https://your-prod-url/health
   ```

5. **Revoke old credentials** in the service console

### Best Practices

- **Test in staging first** before rotating production secrets
- **Schedule rotations** during low-traffic windows
- **Monitor logs** after rotation for authentication errors
- **Have rollback plan** - keep old Parameter Store version for 24 hours

---

## Troubleshooting

### Secret Not Found Error

**Error:** `Missing required environment variable: GOOGLE_API_KEY`

**Solution:**
1. Verify secret exists in Parameter Store:
   ```bash
   AWS_PROFILE=jse-datasphere-elroy aws ssm get-parameter \
     --name "/copilot/jse-datasphere-chatbot/dev/secrets/GOOGLE_API_KEY"
   ```

2. Check Copilot manifest has correct path:
   ```yaml
   secrets:
     GOOGLE_API_KEY: /copilot/${COPILOT_APPLICATION_NAME}/${COPILOT_ENVIRONMENT_NAME}/secrets/GOOGLE_API_KEY
   ```

3. Redeploy the service:
   ```bash
   copilot svc deploy --name api --env dev
   ```

### Invalid JSON in GCP Service Account

**Error:** `GCP_SERVICE_ACCOUNT_INFO must be valid JSON`

**Solution:**
1. Verify JSON is valid:
   ```bash
   echo "$GCP_SERVICE_ACCOUNT_INFO" | jq .
   ```

2. If storing in Parameter Store, ensure proper escaping:
   ```bash
   # Use file input for complex JSON
   aws ssm put-parameter \
     --name "/copilot/.../GCP_SERVICE_ACCOUNT_INFO" \
     --value file://service-account.json \
     --type "SecureString" \
     --overwrite
   ```

### Pre-commit Hook Blocking Commit

**Error:** `Detected private key - commit blocked`

**Solution:**
1. **If legitimate secret in test file:**
   - Add to `.secrets.baseline`:
     ```bash
     detect-secrets scan --baseline .secrets.baseline
     ```

2. **If accidental secret:**
   - Remove from staged files
   - Use environment variables or Parameter Store instead

---

## Security Checklist

- [ ] All secrets in `.env` files (never committed)
- [ ] All production secrets in AWS Parameter Store
- [ ] Pre-commit hooks installed (`pre-commit install`)
- [ ] `.env.example` has placeholder values only
- [ ] No hardcoded credentials in code
- [ ] Secrets rotated quarterly (at minimum)
- [ ] CloudTrail logging enabled for Parameter Store access
- [ ] IAM roles follow least privilege principle

---

## Additional Resources

- [AWS Systems Manager Parameter Store](https://docs.aws.amazon.com/systems-manager/latest/userguide/systems-manager-parameter-store.html)
- [AWS Copilot Secrets Guide](https://aws.github.io/copilot-cli/docs/developing/secrets/)
- [Pre-commit Hook Documentation](https://pre-commit.com/)
- [Detect Secrets Tool](https://github.com/Yelp/detect-secrets)

---

## Support

For questions or issues with secrets management:
1. Check this documentation first
2. Review CloudWatch logs for error details
3. Contact the platform team lead
4. Create an issue in the repository (never include actual secrets)

# AWS Copilot Configuration Examples

**Version:** 1.0.0  
**Last Updated:** 2025-01-XX

This directory contains sanitized example AWS Copilot configuration files that are safe to commit to GitHub. All sensitive information (API keys, credentials, etc.) has been replaced with placeholders.

## 📁 Directory Structure

```
examples/
├── README.md                          # This file
├── api/
│   ├── manifest.yml.example          # Main API service manifest
│   └── addons/
│       └── redis.yml.example         # Redis addon configuration
├── chroma/
│   └── manifest.yml.example          # ChromaDB service manifest
└── environments/
    ├── dev/
    │   └── manifest.yml.example      # Development environment
    ├── staging/
    │   └── manifest.yml.example      # Staging environment
    ├── prod/
    │   └── manifest.yml.example      # Production environment
    └── test/
        └── manifest.yml.example      # Test environment
```

## 🚀 Quick Start

### 1. Copy Example Files to Actual Copilot Directory

The actual Copilot configuration files are located in `fastapi_app/copilot/` (which is excluded from Git). To use these examples:

```bash
# Copy the example files to your actual copilot directory
cp -r fastapi_app/copilot/examples/api/manifest.yml.example fastapi_app/copilot/api/manifest.yml
cp -r fastapi_app/copilot/examples/chroma/manifest.yml.example fastapi_app/copilot/chroma/manifest.yml
cp -r fastapi_app/copilot/examples/api/addons/redis.yml.example fastapi_app/copilot/api/addons/redis.yml
cp -r fastapi_app/copilot/examples/environments/*/manifest.yml.example fastapi_app/copilot/environments/*/manifest.yml
```

### 2. Replace Placeholders

Open each manifest file and replace all `<PLACEHOLDER>` values with your actual configuration:

- `<YOUR_AWS_ACCESS_KEY_ID>` - Your AWS access key ID
- `<YOUR_AWS_SECRET_ACCESS_KEY>` - Your AWS secret access key
- `<YOUR_AWS_REGION>` - Your AWS region (e.g., `us-east-1`)
- `<YOUR_S3_BUCKET_NAME>` - Your S3 bucket name
- `<YOUR_GOOGLE_API_KEY>` - Your Google AI API key
- `<YOUR_GCP_PROJECT_ID>` - Your GCP project ID
- `<YOUR_GCP_SERVICE_ACCOUNT_JSON_STRING>` - Your GCP service account JSON (as a single-line string)
- `<YOUR_BIGQUERY_DATASET>` - Your BigQuery dataset name
- `<YOUR_BIGQUERY_TABLE>` - Your BigQuery table name
- `<YOUR_BIGQUERY_LOCATION>` - Your BigQuery location (e.g., `US`, `EU`)

### 3. Use AWS Secrets Manager (Recommended)

For production deployments, **strongly recommend** using AWS Secrets Manager instead of plain text environment variables. The example files include commented-out `secrets` sections showing how to configure this.

#### Setting up Secrets in AWS Secrets Manager:

```bash
# Create secrets using AWS CLI or AWS Console
aws secretsmanager create-secret \
  --name /copilot/<your-app-name>/<env-name>/secrets/GOOGLE_API_KEY \
  --secret-string "your-actual-api-key"

aws secretsmanager create-secret \
  --name /copilot/<your-app-name>/<env-name>/secrets/AWS_ACCESS_KEY_ID \
  --secret-string "your-actual-access-key"

# ... repeat for other secrets
```

Then uncomment and configure the `secrets` section in your manifest files.

## 📋 File Descriptions

### API Service Manifest (`api/manifest.yml.example`)

Main configuration for the FastAPI application service:
- **Type:** Load Balanced Web Service
- **Resources:** Configurable CPU/memory per environment
- **Features:**
  - Health check endpoint configuration
  - Sticky sessions for async job polling
  - Service Connect for intra-service communication
  - Environment-specific overrides (staging, prod, test)

### Chroma Service Manifest (`chroma/manifest.yml.example`)

Configuration for the ChromaDB vector database service:
- **Type:** Backend Service (internal only)
- **Storage:** EFS volume for persistent vector storage
- **Network:** Service Connect enabled for service discovery

### Redis Addon (`api/addons/redis.yml.example`)

CloudFormation template for Redis cache:
- **Type:** AWS ElastiCache Redis cluster
- **Instance:** `cache.t3.micro` (adjustable for production)
- **Network:** VPC-integrated with security groups

### Environment Manifests

Environment-specific configurations:
- **dev:** Development environment
- **staging:** Staging environment
- **prod:** Production environment
- **test:** Test environment

Each environment can have its own observability, network, and load balancer settings.

## 🔒 Security Best Practices

1. **Never commit actual credentials** - The actual `fastapi_app/copilot/` directory is excluded from Git via `.gitignore`
2. **Use AWS Secrets Manager** - For production, store sensitive values in AWS Secrets Manager
3. **Rotate credentials regularly** - Update API keys and access credentials periodically
4. **Use IAM roles** - Prefer IAM roles over access keys when possible
5. **Review permissions** - Ensure services have only the minimum required permissions

## 📝 Version Information

All example files include version headers:
- **Version:** 1.0.0
- **Last Updated:** 2025-01-XX

When updating these examples, increment the version number and update the date.

## 🛠️ Customization

### Adjusting Resources

You can adjust CPU and memory allocations per environment in the `api/manifest.yml.example`:

```yaml
environments:
  prod:
    cpu: 1024      # Adjust based on your needs
    memory: 2048   # Adjust based on your needs
    count: 2       # Number of instances
```

### Redis Instance Size

In `api/addons/redis.yml.example`, adjust the `CacheNodeType`:

```yaml
RedisCluster:
  Properties:
    CacheNodeType: cache.t3.micro  # Options: cache.t3.micro, cache.t3.small, cache.t3.medium, etc.
```

### Health Check Configuration

Modify health check settings in the API manifest:

```yaml
http:
  healthcheck:
    path: '/health'
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 30s
```

## 📚 Additional Resources

- [AWS Copilot Documentation](https://aws.github.io/copilot-cli/docs/)
- [Load Balanced Web Service Manifest](https://aws.github.io/copilot-cli/docs/manifest/lb-web-service/)
- [Backend Service Manifest](https://aws.github.io/copilot-cli/docs/manifest/backend-service/)
- [Environment Manifest](https://aws.github.io/copilot-cli/docs/manifest/environment/)
- [AWS Secrets Manager](https://docs.aws.amazon.com/secretsmanager/)

## ❓ Troubleshooting

### Common Issues

1. **Placeholder values not replaced:** Ensure all `<PLACEHOLDER>` values are replaced with actual values
2. **Secrets not found:** Verify secrets exist in AWS Secrets Manager with correct paths
3. **Service Connect issues:** Ensure `network.connect: true` is set in both services
4. **Redis connection failures:** Check security group rules and subnet configuration

## 📞 Support

For issues or questions:
1. Check the AWS Copilot documentation
2. Review the actual deployment logs in CloudWatch
3. Verify all placeholders have been replaced
4. Ensure AWS credentials have proper permissions

---

**Note:** These are example files. Always review and customize them for your specific use case before deployment.



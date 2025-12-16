# AWS Copilot Deployment Guide - Fast Chat V2

This guide covers deploying the FastAPI application with the new `fast_chat_v2` endpoint using AWS Copilot.

## 🎯 Overview

Your Copilot configuration has been optimized for the financial data chatbot:
- **API Service**: Increased memory/CPU for financial data processing
- **ChromaDB Service**: Backend service with persistent EFS storage
- **Environment**: Staging configuration ready for deployment

## 📋 Pre-deployment Checklist

### 1. Verify Application Structure
```bash
cd fastapi_app

# Ensure all required files are present
ls -la financial_data.csv metadata_for_fast_chat_v2.json companies.json

# Validate local setup
./validate_fast_chat_v2.sh
```

### 2. Check Copilot Configuration
```bash
# Verify Copilot is initialized
copilot app ls

# Check services
copilot svc ls

# Should show:
# Name    Type
# api     Load Balanced Web Service
# chroma  Backend Service
```

## 🚀 Deployment Steps

### 1. Deploy ChromaDB Service First
```bash
# Deploy the ChromaDB backend service
copilot svc deploy --name chroma --env staging

# Wait for deployment to complete
copilot svc status --name chroma --env staging
```

### 2. Deploy API Service
```bash
# Deploy the main API with fast_chat_v2 endpoint
copilot svc deploy --name api --env staging

# Monitor deployment
copilot svc logs --name api --env staging --follow
```

### 3. Verify Deployment
```bash
# Get the application URL
copilot svc show --name api --env staging

# Test health endpoint
curl https://your-app-url.region.elb.amazonaws.com/health

# Expected response:
# {
#   "status": "healthy",
#   "s3_client": "available",
#   "metadata": "available",
#   "financial_data": {
#     "status": "available",
#     "records": 2099
#   }
# }
```

## 🧪 Testing Fast Chat V2

### Basic Test
```bash
curl -X POST "https://your-app-url/fast_chat_v2" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Show me MDS revenue for 2024",
    "memory_enabled": true
  }'
```

### Advanced Test with Follow-up
```bash
# First query
curl -X POST "https://your-app-url/fast_chat_v2" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Compare JBG and CPJ profit margins",
    "memory_enabled": true
  }'

# Follow-up query (using conversation history from first response)
curl -X POST "https://your-app-url/fast_chat_v2" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What about 2022?",
    "memory_enabled": true,
    "conversation_history": [...]
  }'
```

## 📊 Resource Configuration

### Current Allocations

**API Service (Staging):**
- CPU: 512 units (0.5 vCPU)
- Memory: 1024 MB
- Instances: 1

**ChromaDB Service:**
- CPU: 256 units (0.25 vCPU)
- Memory: 512 MB
- Storage: EFS persistent volume
- Instances: 1

### Environment-Specific Overrides

```yaml
# Staging: Balanced resources
cpu: 512
memory: 1024
count: 1

# Production: High availability
cpu: 1024
memory: 2048
count: 2

# Test: Minimal resources
cpu: 256
memory: 512
count: 1
```

## 🔧 Configuration Details

### Required Environment Variables
- `GOOGLE_API_KEY`: For AI-powered query parsing ✅
- `GCP_SERVICE_ACCOUNT_INFO`: Google Cloud credentials ✅
- `AWS_*`: S3 and other AWS services ✅
- `CHROMA_HOST`: Points to ChromaDB service ✅

### Data Files Included
- `financial_data.csv`: 2099+ financial records ✅
- `metadata_for_fast_chat_v2.json`: AI metadata ✅
- `companies.json`: Company information ✅

## 🔍 Monitoring & Troubleshooting

### View Logs
```bash
# API service logs
copilot svc logs --name api --env staging --follow

# ChromaDB logs
copilot svc logs --name chroma --env staging --follow

# Filter for fast_chat_v2
copilot svc logs --name api --env staging | grep "fast_chat_v2"
```

### Common Issues

**1. Financial data not loading**
```bash
# Check if CSV file is in container
copilot task run --image api --command "ls -la financial_data.csv"

# Check pandas import
copilot task run --image api --command "python -c 'import pandas; print(pandas.__version__)'"
```

**2. AI parsing not working**
```bash
# Verify Google API key
copilot svc logs --name api --env staging | grep "GOOGLE_API_KEY"

# Check if fallback parsing is working
curl -X POST "https://your-app-url/fast_chat_v2" \
  -d '{"query": "test query", "memory_enabled": false}'
```

**3. ChromaDB connection issues**
```bash
# Check ChromaDB service status
copilot svc status --name chroma --env staging

# Test connectivity from API
copilot task run --image api --command "curl http://chroma:8000/api/v1/heartbeat"
```

### Performance Monitoring
```bash
# Check resource utilization
copilot svc status --name api --env staging
copilot svc status --name chroma --env staging

# View CloudWatch metrics
aws cloudwatch get-metric-statistics \
  --namespace AWS/ECS \
  --metric-name CPUUtilization \
  --dimensions Name=ServiceName,Value=api
```

## 🔒 Security Best Practices

### Current Setup
- API keys in environment variables (functional but not ideal)
- Services communicate via private Service Connect
- EFS storage encrypted at rest

### Recommended Improvements
```bash
# Store secrets in Parameter Store
aws ssm put-parameter \
  --name "/copilot/jse-datasphere-chatbot/staging/secrets/GOOGLE_API_KEY" \
  --value "your-api-key" \
  --type "SecureString"

# Update manifest to use secrets (see commented section)
```

## 🚀 Next Steps

1. **Deploy to staging** using commands above
2. **Test thoroughly** with various financial queries
3. **Monitor performance** and adjust resources if needed
4. **Promote to production** when ready:
   ```bash
   copilot svc deploy --name api --env prod
   copilot svc deploy --name chroma --env prod
   ```

## 📞 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review service logs for error messages
3. Ensure all environment variables are properly set
4. Verify financial data files are present in the container

Your `fast_chat_v2` endpoint is now ready for AWS deployment! 🎉

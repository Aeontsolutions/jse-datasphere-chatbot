# Deploying FastAPI API to AWS ECS Fargate

This guide walks you through deploying your FastAPI API (with Docker) to AWS ECS Fargate, including ECR, IAM, VPC, networking, and testing steps.

---

## Prerequisites
- AWS account (with admin or sufficient permissions)
- AWS CLI installed and configured
- Docker installed
- Your FastAPI app and Dockerfile ready

---

## 1. IAM User and Permissions
- Create an IAM user (e.g., `ecs-deployer`) with programmatic access.
- Attach a custom policy with permissions for ECS, ECR, EC2 (VPC), IAM, and CloudWatch Logs.
- Make sure the policy includes `iam:CreateRole`, `iam:PassRole`, and `iam:CreateServiceLinkedRole`.

## 2. Configure AWS CLI
```bash
aws configure
```
Enter your new IAM user's credentials and set region (e.g., `us-east-1`).

---

## 3. Create ECR Repository
```bash
aws ecr create-repository --repository-name jse-datasphere-chatbot
```

---

## 4. Build and Push Docker Image (for amd64/x86_64)
```bash
docker buildx build --platform linux/amd64 -t jse-datasphere-chatbot:latest .
docker tag jse-datasphere-chatbot:latest <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/jse-datasphere-chatbot:latest
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin <your-account-id>.dkr.ecr.us-east-1.amazonaws.com
docker push <your-account-id>.dkr.ecr.us-east-1.amazonaws.com/jse-datasphere-chatbot:latest
```

---

## 5. Create VPC, Subnet, and Internet Gateway
- Create a VPC and a public subnet.
- Create and attach an Internet Gateway to the VPC.
- Edit the subnet's route table to add a route: `0.0.0.0/0` → your Internet Gateway.
- Enable **Auto-assign public IPv4 address** on the subnet.

---

## 6. Create Security Group
- Create a security group (e.g., `jse-chatbot-sg`) in your VPC.
- Add an inbound rule: TCP, port 8000, source `0.0.0.0/0` (or restrict as needed).

---

## 7. Create ECS Cluster
```bash
aws ecs create-cluster --cluster-name jse-chatbot-cluster
```

---

## 8. Create IAM Role for ECS Tasks
- In the AWS Console, go to IAM > Roles > Create role.
- Select **Elastic Container Service > Elastic Container Service Task**.
- Attach the `AmazonECSTaskExecutionRolePolicy`.
- Name the role `ecsTaskExecutionRole`.

---

## 9. Create Task Definition
- Use the ECS Console or a JSON file (see `task-definition.json` in this repo).
- Set the image to your ECR image URI.
- Set port mapping to 8000.
- Set `executionRoleArn` to your new `ecsTaskExecutionRole` ARN.
- Set up log configuration for CloudWatch.

---

## 10. Create CloudWatch Log Group
```bash
aws logs create-log-group --log-group-name /ecs/jse-chatbot
```

---

## 11. Create ECS Service
```bash
aws ecs create-service \
  --cluster jse-chatbot-cluster \
  --service-name jse-chatbot-service \
  --task-definition jse-chatbot:1 \
  --desired-count 1 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[<your-subnet-id>],securityGroups=[<your-sg-id>],assignPublicIp=ENABLED}"
```

---

## 12. Troubleshooting
- If tasks can't pull from ECR: check subnet has a route to an Internet Gateway and auto-assign public IP is enabled.
- If you see `exec format error`: rebuild your Docker image for `linux/amd64`.
- If tasks exit with code 255: check logs in ECS/CloudWatch for Python or app errors.

---

## 13. Testing Your API
- Find the public IP of your running ECS task (in the ECS Console > Tasks > Networking).
- Visit `http://<public-ip>:8000/` to see the API info page.
- Visit `http://<public-ip>:8000/docs` for the interactive Swagger UI.
- Test the `/chat` endpoint using Swagger UI, curl, or Postman.

---

## 14. Security
- For production, restrict the security group to trusted IPs.
- Consider using a load balancer and HTTPS (ACM/ALB).
- Store secrets in AWS Secrets Manager or SSM Parameter Store.

---

## 15. Cleanup
- Delete ECS service, cluster, ECR repo, and VPC resources when done to avoid charges.

---

**Congratulations! Your FastAPI app is now running on AWS ECS Fargate.** 
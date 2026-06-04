variable "project" {
  description = "Project name prefix used in all resource names"
  type        = string
  default     = "jse-datasphere"
}

variable "environment" {
  description = "Deployment environment: dev | staging | prod | test"
  type        = string
  validation {
    condition     = contains(["dev", "staging", "prod", "test"], var.environment)
    error_message = "environment must be one of: dev, staging, prod, test"
  }
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "nat_gateway_ha" {
  description = "Deploy one NAT Gateway per AZ (recommended for prod, adds ~$32/month per extra gateway)"
  type        = bool
  default     = false
}

variable "vpc_cidr" {
  description = "CIDR block for the environment VPC"
  type        = string
  default     = "10.0.0.0/16"
}

# ---------------------------------------------------------------------------
# ECS sizing
# ---------------------------------------------------------------------------

variable "task_cpu" {
  description = "Fargate task CPU units (256 | 512 | 1024 | 2048 | 4096)"
  type        = number
  default     = 512
}

variable "task_memory" {
  description = "Fargate task memory in MiB"
  type        = number
  default     = 1024
}

variable "task_count" {
  description = "Number of ECS task replicas"
  type        = number
  default     = 1
}

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------

variable "ecr_repository_url" {
  description = "ECR repository URL without tag (from bootstrap output)"
  type        = string
}

variable "image_tag" {
  description = "Docker image tag to deploy — overridden by app deploy pipeline"
  type        = string
  default     = "latest"
}

# ---------------------------------------------------------------------------
# Application config
# ---------------------------------------------------------------------------

variable "document_metadata_s3_bucket" {
  description = "S3 bucket name for document metadata"
  type        = string
}

variable "gcp_project_id" {
  description = "Google Cloud project ID"
  type        = string
}

variable "bigquery_dataset" {
  description = "BigQuery dataset name"
  type        = string
}

variable "bigquery_table" {
  description = "BigQuery table name"
  type        = string
}

variable "bigquery_location" {
  description = "BigQuery dataset location (US | EU | etc.)"
  type        = string
  default     = "US"
}

variable "async_job_mode" {
  description = "Enable async job processing"
  type        = string
  default     = "true"
}

variable "async_job_ttl_seconds" {
  description = "Async job TTL in seconds"
  type        = string
  default     = "900"
}

variable "async_job_max_progress_history" {
  description = "Max async job progress history entries per job"
  type        = string
  default     = "50"
}

# ---------------------------------------------------------------------------
# Custom domain (optional — leave empty to use raw ALB DNS)
# ---------------------------------------------------------------------------

variable "domain_name" {
  description = "Custom domain for the API (e.g. api.yourdomain.com). Leave empty to use raw ALB DNS."
  type        = string
  default     = ""
}

variable "certificate_arn" {
  description = "ACM certificate ARN for HTTPS. Required when domain_name is set."
  type        = string
  default     = ""
}

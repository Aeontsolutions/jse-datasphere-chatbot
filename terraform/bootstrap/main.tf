# Bootstrap — run once manually before any other Terraform commands.
# Creates the S3 state bucket, DynamoDB lock table, and ECR repository.
#
# Usage:
#   cd terraform/bootstrap
#   terraform init
#   terraform apply
#   # Note the outputs — paste ecr_repository_url into your *.tfvars files.

terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }
  # Bootstrap uses local state — it manages the bucket that stores all other state.
}

provider "aws" {
  region = var.aws_region
}

variable "aws_region" {
  type    = string
  default = "us-east-1"
}

variable "project" {
  type    = string
  default = "jse-datasphere"
}

# ---------------------------------------------------------------------------
# Terraform state storage
# ---------------------------------------------------------------------------

resource "aws_s3_bucket" "tfstate" {
  bucket = "${var.project}-tfstate"
}

resource "aws_s3_bucket_versioning" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "tfstate" {
  bucket = aws_s3_bucket.tfstate.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "tfstate" {
  bucket                  = aws_s3_bucket.tfstate.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_dynamodb_table" "tfstate_lock" {
  name         = "${var.project}-tfstate-lock"
  billing_mode = "PAY_PER_REQUEST"
  hash_key     = "LockID"

  attribute {
    name = "LockID"
    type = "S"
  }
}

# ---------------------------------------------------------------------------
# ECR repository (bootstrapped here so it exists before the first app build)
# ---------------------------------------------------------------------------

resource "aws_ecr_repository" "api" {
  name                 = "${var.project}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }

  lifecycle {
    prevent_destroy = true
  }
}

resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "Keep last 20 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 20
      }
      action = { type = "expire" }
    }]
  })
}

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

output "tfstate_bucket" {
  value       = aws_s3_bucket.tfstate.bucket
  description = "Paste into terraform/main.tf backend block"
}

output "tfstate_lock_table" {
  value       = aws_dynamodb_table.tfstate_lock.name
  description = "Paste into terraform/main.tf backend block"
}

output "ecr_repository_url" {
  value       = aws_ecr_repository.api.repository_url
  description = "Paste into terraform/environments/*.tfvars as ecr_repository_url"
}

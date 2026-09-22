terraform {
  required_version = ">= 1.9"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 6.0"
    }
  }

  backend "s3" {
    # TODO: update bucket/table names if you changed the project variable in bootstrap
    bucket         = "jse-datasphere-tfstate"
    key            = "terraform.tfstate"
    region         = "us-east-1"
    dynamodb_table = "jse-datasphere-tfstate-lock"
    encrypt        = true
  }
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "jse-datasphere-chatbot"
      Environment = var.environment
      ManagedBy   = "terraform"
    }
  }
}

# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------

module "networking" {
  source         = "./modules/networking"
  project        = var.project
  environment    = var.environment
  aws_region     = var.aws_region
  vpc_cidr       = var.vpc_cidr
  nat_gateway_ha = var.nat_gateway_ha
}

module "iam" {
  source      = "./modules/iam"
  project     = var.project
  environment = var.environment
  s3_bucket   = var.document_metadata_s3_bucket
}

module "redis" {
  source      = "./modules/redis"
  project     = var.project
  environment = var.environment
  vpc_id      = module.networking.vpc_id
  subnet_ids  = module.networking.private_subnet_ids
}

# Security group rule defined here to break the circular dependency:
# redis module needs ecs_service SG id, ecs_service module needs redis URL.
resource "aws_security_group_rule" "ecs_to_redis" {
  type                     = "ingress"
  from_port                = 6379
  to_port                  = 6379
  protocol                 = "tcp"
  security_group_id        = module.redis.security_group_id
  source_security_group_id = module.ecs_service.ecs_security_group_id
}

module "ecs_service" {
  source = "./modules/ecs-service"

  project     = var.project
  environment = var.environment

  vpc_id             = module.networking.vpc_id
  public_subnet_ids  = module.networking.public_subnet_ids
  private_subnet_ids = module.networking.private_subnet_ids

  execution_role_arn = module.iam.execution_role_arn
  task_role_arn      = module.iam.task_role_arn

  ecr_repository_url = var.ecr_repository_url
  image_tag          = var.image_tag

  task_cpu    = var.task_cpu
  task_memory = var.task_memory
  task_count  = var.task_count

  redis_url = module.redis.redis_url

  app_port                       = 8000
  document_metadata_s3_bucket    = var.document_metadata_s3_bucket
  gcp_project_id                 = var.gcp_project_id
  bigquery_dataset               = var.bigquery_dataset
  bigquery_table                 = var.bigquery_table
  bigquery_location              = var.bigquery_location
  async_job_mode                 = var.async_job_mode
  async_job_ttl_seconds          = var.async_job_ttl_seconds
  async_job_max_progress_history = var.async_job_max_progress_history

  domain_name     = var.domain_name
  certificate_arn = var.certificate_arn
}

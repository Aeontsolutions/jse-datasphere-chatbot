variable "project" { type = string }
variable "environment" { type = string }

variable "vpc_id" { type = string }
variable "public_subnet_ids" { type = list(string) }
variable "private_subnet_ids" { type = list(string) }

variable "execution_role_arn" { type = string }
variable "task_role_arn" { type = string }

variable "ecr_repository_url" { type = string }
variable "image_tag" { type = string }

variable "task_cpu" { type = number }
variable "task_memory" { type = number }
variable "task_count" { type = number }

variable "app_port" { type = number }
variable "redis_url" { type = string }

variable "document_metadata_s3_bucket" { type = string }
variable "gcp_project_id" { type = string }
variable "bigquery_dataset" { type = string }
variable "bigquery_table" { type = string }
variable "bigquery_location" { type = string }
variable "async_job_mode" { type = string }
variable "async_job_ttl_seconds" { type = string }
variable "async_job_max_progress_history" { type = string }

variable "domain_name" { type = string }
variable "certificate_arn" { type = string }

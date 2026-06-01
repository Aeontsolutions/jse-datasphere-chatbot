environment = "prod"
aws_region  = "us-east-1" # TODO: set your region

task_cpu    = 1024
task_memory = 2048
task_count  = 2

# TODO: paste ecr_repository_url from bootstrap output
ecr_repository_url = "REPLACE_WITH_BOOTSTRAP_OUTPUT"

document_metadata_s3_bucket = "REPLACE_WITH_YOUR_BUCKET_prod"
gcp_project_id              = "REPLACE_WITH_YOUR_GCP_PROJECT"
bigquery_dataset            = "REPLACE_WITH_YOUR_DATASET"
bigquery_table              = "REPLACE_WITH_YOUR_TABLE"
bigquery_location           = "US"

async_job_mode                 = "true"
async_job_ttl_seconds          = "900"
async_job_max_progress_history = "50"

# TODO: fill these in before DNS cutover (Step 5 in TERRAFORM_MIGRATION.md)
domain_name     = "api.yourdomain.com"
certificate_arn = "arn:aws:acm:us-east-1:ACCOUNT_ID:certificate/CERT_ID"

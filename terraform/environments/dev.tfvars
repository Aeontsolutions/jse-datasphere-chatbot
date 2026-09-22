environment = "dev"
aws_region  = "us-east-1" # TODO: set your region

task_cpu    = 256
task_memory = 512
task_count  = 1

# TODO: paste ecr_repository_url from bootstrap output
ecr_repository_url = "REPLACE_WITH_BOOTSTRAP_OUTPUT"

document_metadata_s3_bucket = "REPLACE_WITH_YOUR_BUCKET_dev"
gcp_project_id              = "REPLACE_WITH_YOUR_GCP_PROJECT"
bigquery_dataset            = "REPLACE_WITH_YOUR_DATASET"
bigquery_table              = "REPLACE_WITH_YOUR_TABLE"
bigquery_location           = "US"

async_job_mode                 = "true"
async_job_ttl_seconds          = "900"
async_job_max_progress_history = "50"

domain_name     = ""
certificate_arn = ""

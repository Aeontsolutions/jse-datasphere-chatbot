environment = "test"
aws_region  = "us-east-1" # TODO: set your region

task_cpu    = 256
task_memory = 512
task_count  = 1

ecr_repository_url = "REPLACE_WITH_BOOTSTRAP_OUTPUT"

document_metadata_s3_bucket = "REPLACE_WITH_YOUR_BUCKET_test"
gcp_project_id              = "REPLACE_WITH_YOUR_GCP_PROJECT"
bigquery_dataset            = "REPLACE_WITH_YOUR_DATASET"
bigquery_table              = "REPLACE_WITH_YOUR_TABLE"
bigquery_location           = "US"

async_job_mode                 = "false"
async_job_ttl_seconds          = "300"
async_job_max_progress_history = "10"

domain_name     = ""
certificate_arn = ""

variable "project" { type = string }
variable "environment" { type = string }
variable "s3_bucket" {
  description = "S3 bucket the task role needs read/write access to"
  type        = string
}

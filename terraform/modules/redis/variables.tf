variable "project" { type = string }
variable "environment" { type = string }
variable "vpc_id" { type = string }
variable "subnet_ids" { type = list(string) }
variable "node_type" {
  description = "ElastiCache node type"
  type        = string
  default     = "cache.t3.micro"
}

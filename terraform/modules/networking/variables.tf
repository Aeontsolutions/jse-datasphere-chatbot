variable "project" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "vpc_cidr" { type = string }
variable "nat_gateway_ha" {
  description = "Deploy one NAT Gateway per AZ for high availability (recommended for prod)"
  type        = bool
  default     = false
}

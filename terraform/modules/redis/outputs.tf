output "redis_url" {
  value       = "redis://${aws_elasticache_replication_group.main.primary_endpoint_address}:6379/0"
  description = "Redis connection URL injected into the ECS task as REDIS_URL"
}

output "security_group_id" {
  value       = aws_security_group.redis.id
  description = "Redis security group ID — used by root main.tf to attach the ECS ingress rule"
}

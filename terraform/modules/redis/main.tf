resource "aws_security_group" "redis" {
  name        = "${var.project}-${var.environment}-redis"
  description = "Allow Redis access from ECS tasks only"
  vpc_id      = var.vpc_id
  # Ingress rule is defined in the root main.tf as aws_security_group_rule.ecs_to_redis
  # to avoid a circular dependency with the ecs-service module.
}

resource "aws_elasticache_subnet_group" "main" {
  name       = "${var.project}-${var.environment}-redis"
  subnet_ids = var.subnet_ids
}

resource "aws_elasticache_replication_group" "main" {
  replication_group_id = "${var.project}-${var.environment}"
  description          = "Redis for ${var.project}-${var.environment}"
  node_type            = var.node_type
  port                 = 6379
  parameter_group_name = "default.redis7"
  engine_version       = "7.1"
  subnet_group_name    = aws_elasticache_subnet_group.main.name
  security_group_ids   = [aws_security_group.redis.id]

  num_cache_clusters         = 1
  automatic_failover_enabled = false
}

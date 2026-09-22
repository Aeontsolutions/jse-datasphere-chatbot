output "alb_dns_name" {
  description = "ALB DNS name — CNAME/ALIAS target for your custom domain"
  value       = aws_lb.main.dns_name
}

output "alb_zone_id" {
  description = "ALB hosted zone ID — for Route 53 ALIAS records"
  value       = aws_lb.main.zone_id
}

output "ecs_cluster_name" {
  value = aws_ecs_cluster.main.name
}

output "ecs_service_name" {
  value = aws_ecs_service.api.name
}

output "ecs_security_group_id" {
  description = "ECS task security group — referenced by the Redis module"
  value       = aws_security_group.ecs.id
}

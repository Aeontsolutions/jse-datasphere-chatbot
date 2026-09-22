output "alb_dns_name" {
  description = "ALB DNS name — point your domain CNAME/ALIAS record here during cutover"
  value       = module.ecs_service.alb_dns_name
}

output "alb_zone_id" {
  description = "ALB hosted zone ID — used for Route 53 ALIAS records"
  value       = module.ecs_service.alb_zone_id
}

output "ecs_cluster_name" {
  description = "ECS cluster name — used by the app deploy pipeline"
  value       = module.ecs_service.ecs_cluster_name
}

output "ecs_service_name" {
  description = "ECS service name — used by the app deploy pipeline"
  value       = module.ecs_service.ecs_service_name
}

output "ecr_repository_url" {
  description = "ECR repository URL — used by the app deploy pipeline"
  value       = var.ecr_repository_url
}

output "vpc_id" {
  value = module.networking.vpc_id
}

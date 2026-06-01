locals {
  name = "${var.project}-${var.environment}"
}

data "aws_region" "current" {}

# ---------------------------------------------------------------------------
# CloudWatch log group
# ---------------------------------------------------------------------------

resource "aws_cloudwatch_log_group" "api" {
  name              = "/ecs/${local.name}/api"
  retention_in_days = 30
}

# ---------------------------------------------------------------------------
# ECS cluster
# ---------------------------------------------------------------------------

resource "aws_ecs_cluster" "main" {
  name = local.name

  setting {
    name  = "containerInsights"
    value = "disabled"
  }
}

resource "aws_ecs_cluster_capacity_providers" "main" {
  cluster_name       = aws_ecs_cluster.main.name
  capacity_providers = ["FARGATE"]

  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight            = 1
  }
}

# ---------------------------------------------------------------------------
# Security groups
# ---------------------------------------------------------------------------

resource "aws_security_group" "alb" {
  name        = "${local.name}-alb"
  description = "ALB: allow inbound HTTP/HTTPS from internet"
  vpc_id      = var.vpc_id

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  dynamic "ingress" {
    for_each = var.certificate_arn != "" ? [1] : []
    content {
      from_port   = 443
      to_port     = 443
      protocol    = "tcp"
      cidr_blocks = ["0.0.0.0/0"]
    }
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "ecs" {
  name        = "${local.name}-ecs"
  description = "ECS tasks: allow inbound from ALB only"
  vpc_id      = var.vpc_id

  ingress {
    from_port       = var.app_port
    to_port         = var.app_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

# ---------------------------------------------------------------------------
# Application Load Balancer
# ---------------------------------------------------------------------------

resource "aws_lb" "main" {
  name               = local.name
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = var.public_subnet_ids
}

resource "aws_lb_target_group" "api" {
  name        = "${local.name}-api"
  port        = var.app_port
  protocol    = "HTTP"
  vpc_id      = var.vpc_id
  target_type = "ip"

  health_check {
    path                = "/health"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 3
    matcher             = "200"
  }

  # Required for async job polling — client always hits the same task
  stickiness {
    type            = "lb_cookie"
    cookie_duration = 86400
    enabled         = true
  }

  # Match Copilot prod behaviour; allow in-flight requests to finish
  deregistration_delay = var.environment == "prod" ? 120 : 30
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    # Redirect to HTTPS when a certificate is present, otherwise forward directly
    type = var.certificate_arn != "" ? "redirect" : "forward"

    dynamic "redirect" {
      for_each = var.certificate_arn != "" ? [1] : []
      content {
        port        = "443"
        protocol    = "HTTPS"
        status_code = "HTTP_301"
      }
    }

    dynamic "forward" {
      for_each = var.certificate_arn == "" ? [1] : []
      content {
        target_group {
          arn = aws_lb_target_group.api.arn
        }
      }
    }
  }
}

resource "aws_lb_listener" "https" {
  count             = var.certificate_arn != "" ? 1 : 0
  load_balancer_arn = aws_lb.main.arn
  port              = 443
  protocol          = "HTTPS"
  ssl_policy        = "ELBSecurityPolicy-TLS13-1-2-2021-06"
  certificate_arn   = var.certificate_arn

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.api.arn
  }
}

# ---------------------------------------------------------------------------
# ECS task definition
# ---------------------------------------------------------------------------

resource "aws_ecs_task_definition" "api" {
  family                   = "${local.name}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  container_definitions = jsonencode([
    {
      name  = "api"
      image = "${var.ecr_repository_url}:${var.image_tag}"

      portMappings = [{
        containerPort = var.app_port
        protocol      = "tcp"
        name          = "api"
      }]

      # Matches Copilot StopTimeout override — allows in-flight async jobs to finish
      stopTimeout = 120

      environment = [
        { name = "AWS_DEFAULT_REGION",               value = data.aws_region.current.name },
        { name = "DOCUMENT_METADATA_S3_BUCKET",      value = var.document_metadata_s3_bucket },
        { name = "GCP_PROJECT_ID",                   value = var.gcp_project_id },
        { name = "BIGQUERY_DATASET",                 value = var.bigquery_dataset },
        { name = "BIGQUERY_TABLE",                   value = var.bigquery_table },
        { name = "BIGQUERY_LOCATION",                value = var.bigquery_location },
        { name = "CHROMA_PERSIST_DIRECTORY",         value = "/app/chroma_db" },
        { name = "REDIS_URL",                        value = var.redis_url },
        { name = "ASYNC_JOB_MODE",                   value = var.async_job_mode },
        { name = "ASYNC_JOB_TTL_SECONDS",            value = var.async_job_ttl_seconds },
        { name = "ASYNC_JOB_MAX_PROGRESS_HISTORY",   value = var.async_job_max_progress_history },
        { name = "LOG_LEVEL",                        value = "INFO" },
      ]

      # Secrets pulled from SSM Parameter Store at task startup.
      # SSM paths mirror the Copilot convention, just without the /copilot prefix.
      secrets = [
        { name = "GOOGLE_API_KEY",           valueFrom = "/${var.project}/${var.environment}/GOOGLE_API_KEY" },
        { name = "SUMMARIZER_API_KEY",       valueFrom = "/${var.project}/${var.environment}/GOOGLE_API_KEY" },
        { name = "CHATBOT_API_KEY",          valueFrom = "/${var.project}/${var.environment}/GOOGLE_API_KEY" },
        { name = "GCP_SERVICE_ACCOUNT_INFO", valueFrom = "/${var.project}/${var.environment}/GCP_SERVICE_ACCOUNT_INFO" },
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.api.name
          "awslogs-region"        = data.aws_region.current.name
          "awslogs-stream-prefix" = "api"
        }
      }

      # Container-level health check intentionally omitted: the ALB target group
      # already checks /health every 30s. A redundant container check would also
      # fail on slim Python images that don't include curl.
      # TODO: If ChromaDB persistence is required across task restarts, mount an
      # EFS volume here instead of relying on ephemeral /app/chroma_db storage.
    }
  ])
}

# ---------------------------------------------------------------------------
# ECS service
# ---------------------------------------------------------------------------

resource "aws_service_discovery_http_namespace" "main" {
  name = local.name
}

resource "aws_ecs_service" "api" {
  name            = "api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.api.arn
  desired_count   = var.task_count
  launch_type     = "FARGATE"

  deployment_minimum_healthy_percent = 100
  deployment_maximum_percent         = 200

  network_configuration {
    subnets          = var.private_subnet_ids
    security_groups  = [aws_security_group.ecs.id]
    assign_public_ip = false
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.api.arn
    container_name   = "api"
    container_port   = var.app_port
  }

  service_connect_configuration {
    enabled   = true
    namespace = aws_service_discovery_http_namespace.main.arn
  }

  lifecycle {
    # The app deploy pipeline updates task_definition independently of Terraform.
    # Preventing drift detection here avoids spurious plan diffs on every app deploy.
    ignore_changes = [task_definition]
  }

  depends_on = [aws_lb_listener.http]
}

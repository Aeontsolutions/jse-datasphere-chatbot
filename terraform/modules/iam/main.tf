data "aws_iam_policy_document" "ecs_assume_role" {
  statement {
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

# ---------------------------------------------------------------------------
# Execution role — used by the ECS agent (pull image, write logs, read SSM)
# ---------------------------------------------------------------------------

resource "aws_iam_role" "execution" {
  name               = "${var.project}-${var.environment}-execution"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
}

resource "aws_iam_role_policy_attachment" "execution_base" {
  role       = aws_iam_role.execution.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "execution_ssm" {
  name = "ssm-secrets"
  role = aws_iam_role.execution.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = ["ssm:GetParameters", "ssm:GetParameter"]
        # Scoped to this project/environment's secret namespace
        Resource = "arn:aws:ssm:*:*:parameter/${var.project}/${var.environment}/*"
      },
      {
        # Required to decrypt SecureString parameters — without this ECS tasks
        # fail to start with "AccessDeniedException" when pulling secrets.
        Effect   = "Allow"
        Action   = ["kms:Decrypt"]
        Resource = "*"
      }
    ]
  })
}

# ---------------------------------------------------------------------------
# Task role — permissions the app itself uses at runtime
# Replaces passing AWS_ACCESS_KEY_ID/SECRET as env vars (Copilot approach).
# ---------------------------------------------------------------------------

resource "aws_iam_role" "task" {
  name               = "${var.project}-${var.environment}-task"
  assume_role_policy = data.aws_iam_policy_document.ecs_assume_role.json
}

resource "aws_iam_role_policy" "task_s3" {
  name = "s3-document-metadata"
  role = aws_iam_role.task.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect = "Allow"
      Action = [
        "s3:GetObject",
        "s3:PutObject",
        "s3:DeleteObject",
        "s3:ListBucket"
      ]
      Resource = [
        "arn:aws:s3:::${var.s3_bucket}",
        "arn:aws:s3:::${var.s3_bucket}/*"
      ]
    }]
  })
}

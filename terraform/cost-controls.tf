# Cost Control and Spending Limits for EPYC-testing
# Helps prevent unexpected AWS charges and provides automated pausing

# CloudWatch Alarm for High CPU (indicates heavy usage)
resource "aws_cloudwatch_metric_alarm" "high_cpu" {
  count           = var.enable_cost_controls ? 1 : 0
  alarm_name          = "epyc-testing-high-cpu"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "2"
  metric_name         = "CPUUtilization"
  namespace           = "AWS/EC2"
  period              = "300"  # 5 minutes
  statistic           = "Average"
  threshold           = "80"
  alarm_description   = "This metric monitors ec2 cpu utilization"
  alarm_actions       = var.enable_cost_controls ? [aws_sns_topic.cost_alerts[0].arn] : []

  dimensions = {
    InstanceId = aws_instance.epyc_testing.id
  }

  tags = {
    Name        = "epyc-testing-cpu-alarm"
    Environment = var.environment
    Purpose     = "Cost Control"
  }
}

# CloudWatch Alarm for Network Out (indicates data transfer costs)
resource "aws_cloudwatch_metric_alarm" "high_network_out" {
  count           = var.enable_cost_controls ? 1 : 0
  alarm_name          = "epyc-testing-high-network-out"
  comparison_operator = "GreaterThanThreshold"
  evaluation_periods  = "1"
  metric_name         = "NetworkOut"
  namespace           = "AWS/EC2"
  period              = "3600"  # 1 hour
  statistic           = "Sum"
  threshold           = "10737418240"  # 10 GB per hour
  alarm_description   = "High network data transfer - potential cost issue"
  alarm_actions       = var.enable_cost_controls ? [aws_sns_topic.cost_alerts[0].arn] : []

  dimensions = {
    InstanceId = aws_instance.epyc_testing.id
  }

  tags = {
    Name        = "epyc-testing-network-alarm"
    Environment = var.environment
    Purpose     = "Cost Control"
  }
}

# SNS Topic for Cost Alerts
resource "aws_sns_topic" "cost_alerts" {
  count = var.enable_cost_controls ? 1 : 0
  name = "epyc-testing-cost-alerts"

  tags = {
    Name        = "epyc-testing-cost-alerts"
    Environment = var.environment
    Purpose     = "Cost Control"
  }
}

# SNS Topic Subscription (email alerts)
resource "aws_sns_topic_subscription" "cost_alerts_email" {
  count     = var.enable_cost_controls && var.alert_email != "" ? 1 : 0
  topic_arn = var.enable_cost_controls ? aws_sns_topic.cost_alerts[0].arn : ""
  protocol  = "email"
  endpoint  = var.alert_email
}

# Note: AWS Budget resource removed due to complex syntax
# You can create budgets manually in the AWS Console:
# 1. Go to AWS Billing & Cost Management
# 2. Click "Budgets" 
# 3. Create a new budget with $200/month limit
# 4. Set alerts at 80% and 100%

# Lambda function for automated instance stopping
resource "aws_lambda_function" "instance_stopper" {
  count            = var.enable_cost_controls ? 1 : 0
  filename         = "instance_stopper.zip"
  function_name    = "epyc-testing-instance-stopper"
  role            = aws_iam_role.lambda_stopper_role[0].arn
  handler         = "index.handler"
  source_code_hash = data.archive_file.lambda_stopper_zip[0].output_base64sha256
  runtime         = "python3.9"
  timeout         = 60

  environment {
    variables = {
      INSTANCE_ID = aws_instance.epyc_testing.id
    }
  }

  tags = {
    Name        = "epyc-testing-stopper"
    Environment = var.environment
    Purpose     = "Cost Control"
  }
}

# Create Lambda deployment package
data "archive_file" "lambda_stopper_zip" {
  count       = var.enable_cost_controls ? 1 : 0
  type        = "zip"
  output_path = "instance_stopper.zip"
  source {
    content = templatefile("${path.module}/lambda/instance_stopper.py", {
      instance_id = aws_instance.epyc_testing.id
    })
    filename = "index.py"
  }
}

# IAM Role for Lambda function
resource "aws_iam_role" "lambda_stopper_role" {
  count = var.enable_cost_controls ? 1 : 0
  name = "epyc-testing-lambda-stopper-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = {
    Name        = "epyc-testing-lambda-role"
    Environment = var.environment
    Purpose     = "Cost Control"
  }
}

# IAM Policy for Lambda to stop EC2 instances
resource "aws_iam_role_policy" "lambda_stopper_policy" {
  count = var.enable_cost_controls ? 1 : 0
  name = "epyc-testing-lambda-stopper-policy"
  role = aws_iam_role.lambda_stopper_role[0].id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:*:*:*"
      },
      {
        Effect = "Allow"
        Action = [
          "ec2:StopInstances",
          "ec2:DescribeInstances"
        ]
        Resource = "*"
      },
      {
        Effect = "Allow"
        Action = [
          "sns:Publish"
        ]
        Resource = var.enable_cost_controls ? aws_sns_topic.cost_alerts[0].arn : "*"
      }
    ]
  })
}

# EventBridge rule for daily cost check (runs at 6 PM UTC)
resource "aws_cloudwatch_event_rule" "daily_cost_check" {
  count               = var.enable_cost_controls ? 1 : 0
  name                = "epyc-testing-daily-cost-check"
  description         = "Check costs daily and stop instance if needed"
  schedule_expression = "cron(0 18 * * ? *)"  # 6 PM UTC daily

  tags = {
    Name        = "epyc-testing-cost-check"
    Environment = var.environment
    Purpose     = "Cost Control"
  }
}

# EventBridge target to trigger Lambda
resource "aws_cloudwatch_event_target" "lambda_target" {
  count     = var.enable_cost_controls ? 1 : 0
  rule      = aws_cloudwatch_event_rule.daily_cost_check[0].name
  target_id = "TriggerLambda"
  arn       = aws_lambda_function.instance_stopper[0].arn
}

# Permission for EventBridge to invoke Lambda
resource "aws_lambda_permission" "allow_eventbridge" {
  count         = var.enable_cost_controls ? 1 : 0
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.instance_stopper[0].function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.daily_cost_check[0].arn
} 
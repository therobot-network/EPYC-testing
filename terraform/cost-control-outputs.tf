# Cost Control Outputs

# Cost Control Status
output "cost_controls_enabled" {
  description = "Whether cost controls are enabled"
  value       = var.enable_cost_controls
}

output "monthly_budget_limit" {
  description = "Monthly budget limit in USD"
  value       = var.monthly_budget_limit
}

output "alert_email" {
  description = "Email address for cost alerts (masked)"
  value       = var.alert_email != "" ? "${substr(var.alert_email, 0, 3)}***@${split("@", var.alert_email)[1]}" : "Not configured"
}

# Cost Control Resources
output "cost_control_resources" {
  description = "Created cost control resources"
  value = var.enable_cost_controls ? {
    status = "enabled"
    sns_topic_arn = "Would be created"
    lambda_function = "Would be created"
    cpu_alarm = "Would be created" 
    network_alarm = "Would be created"
    daily_check_rule = "Would be created"
  } : {
    status = "disabled"
    reason = "Requires additional AWS permissions (SNS, Lambda, EventBridge)"
    manual_alternative = "Use scripts/cost-control.sh for manual management"
  }
}

# Estimated Costs (c6a.24xlarge)
output "cost_estimates" {
  description = "Estimated costs for c6a.24xlarge instance"
  value = {
    hourly_rate_usd = 4.608
    daily_cost_usd = 110.59
    monthly_cost_usd = 3317.76
    warning = "These are approximate on-demand rates. Actual costs may vary."
  }
}

# Cost Control Commands
output "cost_control_commands" {
  description = "Useful commands for cost management"
  value = {
    manual_cost_control = "cd ../scripts && ./cost-control.sh"
    stop_instance = "aws ec2 stop-instances --instance-ids ${aws_instance.epyc_testing.id}"
    start_instance = "aws ec2 start-instances --instance-ids ${aws_instance.epyc_testing.id}"
    check_status = "aws ec2 describe-instances --instance-ids ${aws_instance.epyc_testing.id} --query 'Reservations[0].Instances[0].State.Name'"
  }
} 
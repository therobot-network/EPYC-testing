# Terraform Variables for EPYC-testing Infrastructure

# AWS Configuration
variable "aws_region" {
  description = "AWS region for resources"
  type        = string
  default     = "us-west-1"
}

variable "environment" {
  description = "Environment name (dev, staging, prod)"
  type        = string
  default     = "dev"
}

# EC2 Instance Configuration
variable "instance_type" {
  description = "EC2 instance type"
  type        = string
  default     = "c6a.large"
  
  validation {
    condition = contains([
      "t3.micro", "t3.small", "t3.medium", "t3.large",
      "c6a.large", "c6a.xlarge", "c6a.2xlarge", "c6a.24xlarge",
      "m5.large", "m5.xlarge", "m5.2xlarge"
    ], var.instance_type)
    error_message = "Instance type must be a valid EC2 instance type suitable for ML workloads."
  }
}

variable "availability_zone" {
  description = "Availability zone for the EC2 instance"
  type        = string
  default     = "us-west-1c"
}

variable "ec2_user" {
  description = "EC2 user for SSH connections"
  type        = string
  default     = "ubuntu"
}

# Storage Configuration
variable "volume_type" {
  description = "EBS volume type"
  type        = string
  default     = "gp3"
}

variable "volume_size" {
  description = "EBS volume size in GB"
  type        = number
  default     = 20
}

# Key Pair Configuration
variable "key_pair_name" {
  description = "Name of the AWS key pair"
  type        = string
  default     = "griffin-connect"
}

variable "public_key_path" {
  description = "Path to the public key file"
  type        = string
  default     = "~/.ssh/griffin-connect.pub"
}

# Security Configuration
variable "allowed_ssh_cidrs" {
  description = "CIDR blocks allowed for SSH access"
  type        = list(string)
  default     = ["0.0.0.0/0"]  # WARNING: Restrict this in production
}

variable "allowed_http_cidrs" {
  description = "CIDR blocks allowed for HTTP/HTTPS access"
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

# Networking Configuration
variable "use_elastic_ip" {
  description = "Whether to use an Elastic IP for the instance"
  type        = bool
  default     = false
}

# Monitoring Configuration
variable "enable_detailed_monitoring" {
  description = "Enable detailed CloudWatch monitoring"
  type        = bool
  default     = true
}

variable "log_retention_days" {
  description = "CloudWatch logs retention period in days"
  type        = number
  default     = 7
}

# S3 Configuration (optional)
variable "s3_bucket_name" {
  description = "S3 bucket name for model storage (optional)"
  type        = string
  default     = ""
}

# Application Configuration
variable "app_port" {
  description = "Port for the ML application"
  type        = number
  default     = 8000
}

# Import Configuration (for existing resources)
variable "existing_instance_id" {
  description = "Existing EC2 instance ID to import (optional)"
  type        = string
  default     = ""
}

variable "existing_security_group_id" {
  description = "Existing security group ID to import (optional)"
  type        = string
  default     = ""
}

# Cost Control Configuration
variable "monthly_budget_limit" {
  description = "Monthly budget limit in USD"
  type        = number
  default     = 100
}

variable "alert_email" {
  description = "Email address for cost and usage alerts"
  type        = string
  default     = ""
}

variable "enable_cost_controls" {
  description = "Enable automated cost control features"
  type        = bool
  default     = true
}

variable "auto_stop_idle_hours" {
  description = "Hours of idle time before auto-stopping instance"
  type        = number
  default     = 2
} 
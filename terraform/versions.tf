terraform {
  required_version = ">= 1.0"
  
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    archive = {
      source  = "hashicorp/archive"
      version = "~> 2.0"
    }
  }
  
  # Uncomment for remote state (recommended for teams)
  # backend "s3" {
  #   bucket         = "your-terraform-state-bucket"
  #   key            = "epyc-testing/terraform.tfstate"
  #   region         = "us-west-1"
  #   encrypt        = true
  #   dynamodb_table = "terraform-state-locks"
  # }
} 
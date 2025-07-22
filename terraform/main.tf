# Import-only configuration for existing resources
# This file contains resources that are imported but should not be changed

# Import existing instance with exact current configuration
resource "aws_instance" "epyc_testing" {
  # Core configuration - matches existing instance
  ami                    = "ami-014e30c8a36252ae5"  # Exact AMI from existing instance
  instance_type          = "c6a.24xlarge"           # Exact instance type
  key_name              = "griffin-connect"         # Existing key pair
  vpc_security_group_ids = ["sg-07680b7fd8121efa4"] # Existing security group
  subnet_id             = "subnet-069a84ad6fd48acd0" # Existing subnet
  availability_zone      = "us-west-1c"             # Existing AZ
  
  # Match existing monitoring
  monitoring = false
  
  # Match existing root volume
  root_block_device {
    volume_type           = "gp3"  # Default for new instances
    volume_size          = 512    # Match existing volume size
    delete_on_termination = true
    encrypted            = false   # Match existing
  }
  
  # Keep existing tags
  tags = {
    Name = "epyc-testing-large"  # Match existing tag
  }
  
  # Prevent accidental changes
  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      ami,
      user_data,
      associate_public_ip_address,
      private_ip,
      vpc_security_group_ids,
      subnet_id,
    ]
  }
}

# Import existing security group as-is
resource "aws_security_group" "epyc_testing" {
  name        = "launch-wizard-3"
  description = "launch-wizard-3 created 2025-07-22T15:30:16.570Z"
  vpc_id      = "vpc-097d4299ceb637f23"
  
  # Existing ingress rules
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Existing egress rules  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Prevent changes
  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      ingress,
      egress,
      description,
    ]
  }
}

# Import existing key pair as-is
resource "aws_key_pair" "epyc_testing" {
  key_name   = "griffin-connect"
  public_key = file("../griffin-connect.pub")
  
  # Prevent changes
  lifecycle {
    prevent_destroy = true
    ignore_changes = [
      public_key,
      tags,
    ]
  }
} 
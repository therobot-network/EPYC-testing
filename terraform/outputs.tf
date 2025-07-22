# Simplified Terraform Outputs for Import-Only Configuration

# EC2 Instance Information
output "instance_id" {
  description = "ID of the EC2 instance"
  value       = aws_instance.epyc_testing.id
}

output "instance_type" {
  description = "Type of the EC2 instance"
  value       = aws_instance.epyc_testing.instance_type
}

output "instance_state" {
  description = "State of the EC2 instance"
  value       = aws_instance.epyc_testing.instance_state
}

output "public_ip" {
  description = "Public IP address of the EC2 instance"
  value       = aws_instance.epyc_testing.public_ip
}

output "private_ip" {
  description = "Private IP address of the EC2 instance"
  value       = aws_instance.epyc_testing.private_ip
}

output "public_dns" {
  description = "Public DNS name of the EC2 instance"
  value       = aws_instance.epyc_testing.public_dns
}

output "availability_zone" {
  description = "Availability zone of the EC2 instance"
  value       = aws_instance.epyc_testing.availability_zone
}

# Network Information
output "security_group_id" {
  description = "ID of the security group"
  value       = aws_security_group.epyc_testing.id
}

output "security_group_name" {
  description = "Name of the security group"
  value       = aws_security_group.epyc_testing.name
}

# Key Pair Information
output "key_pair_name" {
  description = "Name of the key pair"
  value       = aws_key_pair.epyc_testing.key_name
}

# Connection Information
output "ssh_connection_command" {
  description = "SSH command to connect to the instance"
  value       = "ssh -i ../griffin-connect.pem ec2-user@${aws_instance.epyc_testing.public_ip}"
}

output "application_url" {
  description = "URL to access the application"
  value       = "http://${aws_instance.epyc_testing.public_ip}:8000"
}

# Resource Summary
output "resource_summary" {
  description = "Summary of managed resources"
  value = {
    instance_id       = aws_instance.epyc_testing.id
    instance_type     = aws_instance.epyc_testing.instance_type
    public_ip         = aws_instance.epyc_testing.public_ip
    private_ip        = aws_instance.epyc_testing.private_ip
    security_group    = aws_security_group.epyc_testing.name
    key_pair          = aws_key_pair.epyc_testing.key_name
    availability_zone = aws_instance.epyc_testing.availability_zone
    region            = "us-west-1"
  }
} 
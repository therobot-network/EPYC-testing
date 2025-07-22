# EPYC-testing Terraform Infrastructure

This directory contains Terraform configuration for managing the EPYC-testing EC2 infrastructure as code.

## 🚀 Quick Start

### Prerequisites

1. **Install Terraform** (>= 1.0)
   ```bash
   # macOS
   brew install terraform
   
   # Or download from: https://developer.hashicorp.com/terraform/downloads
   ```

2. **Configure AWS CLI**
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region
   ```

3. **Create SSH Key Pair** (if you don't have one)
   ```bash
   ssh-keygen -t rsa -b 4096 -f ~/.ssh/griffin-connect
   ```

### Import Existing Infrastructure

If you have an existing EC2 instance (like the current setup), import it:

```bash
cd terraform
chmod +x import-existing.sh
./import-existing.sh
```

### Fresh Deployment

For a completely new deployment:

```bash
cd terraform

# Copy and customize variables
cp terraform.tfvars.example terraform.tfvars
# Edit terraform.tfvars with your values

# Initialize and apply
terraform init
terraform plan
terraform apply
```

## 📁 File Structure

```
terraform/
├── main.tf                    # Main Terraform configuration
├── variables.tf               # Variable definitions
├── outputs.tf                 # Output definitions
├── terraform.tfvars.example   # Example variables file
├── user_data.sh              # EC2 initialization script
├── import-existing.sh        # Import script for existing resources
└── README.md                 # This file
```

## 🔧 Configuration

### Variables File

Copy `terraform.tfvars.example` to `terraform.tfvars` and customize:

```hcl
# AWS Configuration
aws_region = "us-west-1"
environment = "dev"

# EC2 Configuration
instance_type = "c6a.large"
availability_zone = "us-west-1c"

# Security (IMPORTANT: Restrict in production)
allowed_ssh_cidrs = ["YOUR_IP/32"]  # Replace with your IP
allowed_http_cidrs = ["0.0.0.0/0"]

# Key pair
key_pair_name = "griffin-connect"
public_key_path = "~/.ssh/griffin-connect.pub"
```

### Key Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `aws_region` | AWS region | `us-west-1` |
| `instance_type` | EC2 instance type | `c6a.large` |
| `allowed_ssh_cidrs` | IPs allowed SSH access | `["0.0.0.0/0"]` |
| `use_elastic_ip` | Use Elastic IP | `false` |
| `enable_detailed_monitoring` | CloudWatch detailed monitoring | `true` |
| `log_retention_days` | Log retention period | `7` |

## 🏗️ Infrastructure Components

The Terraform configuration creates:

### Core Resources
- **EC2 Instance** (`aws_instance.epyc_testing`)
  - c6a.large instance optimized for ML workloads
  - Encrypted EBS root volume
  - User data script for initialization
  - Lifecycle protection against accidental deletion

- **Security Group** (`aws_security_group.epyc_testing`)
  - SSH access (port 22)
  - HTTP access (port 8000)
  - HTTPS access (port 443)
  - All outbound traffic allowed

- **Key Pair** (`aws_key_pair.epyc_testing`)
  - SSH key for instance access

### Optional Resources
- **Elastic IP** (`aws_eip.epyc_testing`)
  - Consistent public IP (if `use_elastic_ip = true`)

- **IAM Role & Policies** (`aws_iam_role.epyc_testing`)
  - CloudWatch logs access
  - S3 access (if bucket specified)
  - Instance profile for EC2

- **CloudWatch Log Group** (`aws_cloudwatch_log_group.epyc_testing`)
  - Centralized logging for the application

## 📊 Outputs

After `terraform apply`, you'll get useful outputs:

```bash
terraform output
```

Key outputs include:
- `instance_id` - EC2 instance ID
- `public_ip` - Public IP address
- `ssh_connection_command` - Ready-to-use SSH command
- `application_url` - URL to access your application

## 🔄 Common Operations

### View Current State
```bash
terraform show
```

### Plan Changes
```bash
terraform plan
```

### Apply Changes
```bash
terraform apply
```

### View Outputs
```bash
terraform output
terraform output -json  # JSON format
```

### Destroy Infrastructure
```bash
terraform destroy
```

### Import Existing Resources
```bash
# Import specific resource
terraform import aws_instance.epyc_testing i-07784f133e33f426c

# Or use the import script
./import-existing.sh
```

## 🔒 Security Best Practices

### 1. Restrict SSH Access
```hcl
allowed_ssh_cidrs = ["YOUR_PUBLIC_IP/32"]
```

### 2. Use Elastic IP for Production
```hcl
use_elastic_ip = true
```

### 3. Enable Detailed Monitoring
```hcl
enable_detailed_monitoring = true
```

### 4. Encrypt EBS Volumes
```hcl
# Already enabled in main.tf
root_block_device {
  encrypted = true
}
```

## 🚨 Troubleshooting

### Import Issues

**Problem**: Resource already exists
```bash
Error: resource already exists
```

**Solution**: Import the existing resource first
```bash
terraform import aws_instance.epyc_testing i-07784f133e33f426c
```

### State Issues

**Problem**: State file conflicts
```bash
Error: state file locked
```

**Solution**: Force unlock (use carefully)
```bash
terraform force-unlock LOCK_ID
```

### Permission Issues

**Problem**: AWS permissions denied
```bash
Error: UnauthorizedOperation
```

**Solution**: Ensure your AWS user has required permissions:
- EC2 full access
- IAM role creation
- CloudWatch logs access

### Key Pair Issues

**Problem**: Key pair not found
```bash
Error: InvalidKeyPair.NotFound
```

**Solution**: 
1. Create the public key file:
   ```bash
   ssh-keygen -y -f ~/.ssh/griffin-connect > ~/.ssh/griffin-connect.pub
   ```
2. Or update `public_key_path` in `terraform.tfvars`

## 🔄 Migration from Manual Setup

If you're migrating from the existing manual setup:

1. **Backup Current State**
   ```bash
   # Create AMI of current instance
   aws ec2 create-image --instance-id i-07784f133e33f426c --name "epyc-testing-backup"
   ```

2. **Run Import Script**
   ```bash
   ./import-existing.sh
   ```

3. **Verify Configuration**
   ```bash
   terraform plan
   # Should show minimal or no changes
   ```

4. **Apply if Needed**
   ```bash
   terraform apply
   ```

## 🔧 Advanced Configuration

### Multiple Environments

Create environment-specific tfvars files:
```bash
# terraform/environments/dev.tfvars
environment = "dev"
instance_type = "t3.medium"

# terraform/environments/prod.tfvars
environment = "prod"
instance_type = "c6a.large"
use_elastic_ip = true
```

Use with:
```bash
terraform plan -var-file="environments/prod.tfvars"
```

### Remote State

For team collaboration, use remote state:
```hcl
terraform {
  backend "s3" {
    bucket = "your-terraform-state-bucket"
    key    = "epyc-testing/terraform.tfstate"
    region = "us-west-1"
  }
}
```

### Modules

Extract reusable components:
```hcl
# modules/ec2-ml/main.tf
module "ml_instance" {
  source = "./modules/ec2-ml"
  
  instance_type = var.instance_type
  environment   = var.environment
}
```

## 📈 Monitoring and Logging

The Terraform configuration includes:

- **CloudWatch Metrics**: CPU, memory, disk usage
- **CloudWatch Logs**: Application logs centralized
- **Log Rotation**: Automatic log cleanup
- **Health Checks**: Instance monitoring

Access logs:
```bash
# View logs
aws logs describe-log-groups --log-group-name-prefix "/aws/ec2/epyc-testing"

# Stream logs
aws logs tail /aws/ec2/epyc-testing --follow
```

## 🤝 Contributing

When making changes to the Terraform configuration:

1. **Test Changes**
   ```bash
   terraform plan
   ```

2. **Document Changes**
   - Update this README
   - Add comments to configuration files

3. **Version Control**
   - Commit `.tf` files
   - **Never commit** `terraform.tfvars` (contains sensitive data)
   - **Never commit** `.terraform/` directory

## 📚 Additional Resources

- [Terraform AWS Provider Documentation](https://registry.terraform.io/providers/hashicorp/aws/latest/docs)
- [AWS EC2 Best Practices](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-best-practices.html)
- [Terraform Best Practices](https://www.terraform.io/docs/cloud/guides/recommended-practices/index.html) 
#!/bin/bash
# Import Existing EC2 Resources into Terraform
# This script imports your current EC2 instance and related resources

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration from your existing setup
INSTANCE_ID="i-00268dae9fd36421f"
REGION="us-west-1"

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              EPYC-testing Terraform Import                   ║"
echo "║            Import Existing EC2 Resources                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}🔍 Starting Terraform import process...${NC}"

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo -e "${RED}❌ Terraform is not installed. Please install Terraform first.${NC}"
    echo "Visit: https://developer.hashicorp.com/terraform/downloads"
    exit 1
fi

# Check if AWS CLI is configured
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}❌ AWS CLI is not configured. Please configure AWS credentials first.${NC}"
    echo "Run: aws configure"
    exit 1
fi

echo -e "${GREEN}✅ Prerequisites check passed${NC}"

# Initialize Terraform
echo -e "${YELLOW}🔧 Initializing Terraform...${NC}"
terraform init

# Create terraform.tfvars if it doesn't exist
if [ ! -f "terraform.tfvars" ]; then
    echo -e "${YELLOW}📝 Creating terraform.tfvars from example...${NC}"
    cp terraform.tfvars.example terraform.tfvars
    echo -e "${YELLOW}⚠️  Please review and update terraform.tfvars with your specific values${NC}"
fi

# Get instance details
echo -e "${BLUE}🔍 Gathering instance information...${NC}"
INSTANCE_INFO=$(aws ec2 describe-instances --instance-ids $INSTANCE_ID --region $REGION --query 'Reservations[0].Instances[0]')

if [ "$INSTANCE_INFO" = "null" ]; then
    echo -e "${RED}❌ Instance $INSTANCE_ID not found in region $REGION${NC}"
    exit 1
fi

# Extract instance details
SECURITY_GROUP_ID=$(echo $INSTANCE_INFO | jq -r '.SecurityGroups[0].GroupId')
KEY_NAME=$(echo $INSTANCE_INFO | jq -r '.KeyName')
SUBNET_ID=$(echo $INSTANCE_INFO | jq -r '.SubnetId')

echo -e "${GREEN}✅ Found instance details:${NC}"
echo -e "   Instance ID: $INSTANCE_ID"
echo -e "   Security Group: $SECURITY_GROUP_ID"
echo -e "   Key Name: $KEY_NAME"
echo -e "   Subnet ID: $SUBNET_ID"

# Import EC2 instance
echo -e "${YELLOW}📥 Importing EC2 instance...${NC}"
if terraform import aws_instance.epyc_testing $INSTANCE_ID; then
    echo -e "${GREEN}✅ EC2 instance imported successfully${NC}"
else
    echo -e "${RED}❌ Failed to import EC2 instance${NC}"
    exit 1
fi

# Import security group
echo -e "${YELLOW}📥 Importing security group...${NC}"
if terraform import aws_security_group.epyc_testing $SECURITY_GROUP_ID; then
    echo -e "${GREEN}✅ Security group imported successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Security group import failed - this might be okay if using default SG${NC}"
fi

# Import key pair
echo -e "${YELLOW}📥 Importing key pair...${NC}"
if terraform import aws_key_pair.epyc_testing $KEY_NAME; then
    echo -e "${GREEN}✅ Key pair imported successfully${NC}"
else
    echo -e "${YELLOW}⚠️  Key pair import failed - you may need to create a new one${NC}"
fi

# Check for existing IAM role and instance profile
IAM_ROLE_NAME="epyc-testing-ec2-role"
INSTANCE_PROFILE_NAME="epyc-testing-instance-profile"

echo -e "${YELLOW}📥 Checking for existing IAM resources...${NC}"

# Try to import IAM role if it exists
if aws iam get-role --role-name $IAM_ROLE_NAME &> /dev/null; then
    echo -e "${BLUE}🔍 Found existing IAM role, importing...${NC}"
    terraform import aws_iam_role.epyc_testing $IAM_ROLE_NAME
    echo -e "${GREEN}✅ IAM role imported${NC}"
else
    echo -e "${YELLOW}ℹ️  No existing IAM role found - will be created${NC}"
fi

# Try to import instance profile if it exists
if aws iam get-instance-profile --instance-profile-name $INSTANCE_PROFILE_NAME &> /dev/null; then
    echo -e "${BLUE}🔍 Found existing instance profile, importing...${NC}"
    terraform import aws_iam_instance_profile.epyc_testing $INSTANCE_PROFILE_NAME
    echo -e "${GREEN}✅ Instance profile imported${NC}"
else
    echo -e "${YELLOW}ℹ️  No existing instance profile found - will be created${NC}"
fi

# Check for existing CloudWatch log group
LOG_GROUP_NAME="/aws/ec2/epyc-testing"
if aws logs describe-log-groups --log-group-name-prefix $LOG_GROUP_NAME --region $REGION | grep -q $LOG_GROUP_NAME; then
    echo -e "${BLUE}🔍 Found existing CloudWatch log group, importing...${NC}"
    terraform import aws_cloudwatch_log_group.epyc_testing $LOG_GROUP_NAME
    echo -e "${GREEN}✅ CloudWatch log group imported${NC}"
else
    echo -e "${YELLOW}ℹ️  No existing CloudWatch log group found - will be created${NC}"
fi

# Plan the changes
echo -e "${YELLOW}📋 Running Terraform plan to check configuration...${NC}"
terraform plan -detailed-exitcode

PLAN_EXIT_CODE=$?
if [ $PLAN_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}✅ No changes needed - infrastructure matches configuration${NC}"
elif [ $PLAN_EXIT_CODE -eq 2 ]; then
    echo -e "${YELLOW}⚠️  Terraform plan shows changes will be made${NC}"
    echo -e "${YELLOW}Review the plan above carefully before applying${NC}"
else
    echo -e "${RED}❌ Terraform plan failed${NC}"
    exit 1
fi

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  🎉 Import Complete! 🎉                     ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}📝 Next Steps:${NC}"
echo -e "${GREEN}1. Review terraform.tfvars and update any values${NC}"
echo -e "${GREEN}2. Review the Terraform plan output above${NC}"
echo -e "${GREEN}3. If everything looks good, run: terraform apply${NC}"
echo -e "${GREEN}4. Your existing instance is now managed by Terraform!${NC}"
echo ""

echo -e "${YELLOW}💡 Useful Commands:${NC}"
echo -e "${GREEN}  terraform plan${NC}     - Preview changes"
echo -e "${GREEN}  terraform apply${NC}    - Apply changes"
echo -e "${GREEN}  terraform show${NC}     - Show current state"
echo -e "${GREEN}  terraform output${NC}   - Show output values"
echo ""

echo -e "${BLUE}🔒 Your existing EC2 instance is now safely managed by Terraform!${NC}" 
#!/bin/bash
# Terraform Setup Script for EPYC-testing
# Helps set up Terraform for managing EC2 infrastructure

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              EPYC-testing Terraform Setup                   ║"
echo "║            Infrastructure as Code Management                 ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}🚀 Setting up Terraform for EPYC-testing...${NC}"

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macOS"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="Linux"
else
    echo -e "${RED}❌ Unsupported operating system: $OSTYPE${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Operating System: $OS${NC}"

# Check for Terraform
echo -e "${YELLOW}🔍 Checking for Terraform...${NC}"
if command -v terraform &> /dev/null; then
    TERRAFORM_VERSION=$(terraform --version | head -n1)
    echo -e "${GREEN}✅ $TERRAFORM_VERSION${NC}"
else
    echo -e "${YELLOW}⚠️  Terraform not found. Installing...${NC}"
    
    if [[ "$OS" == "macOS" ]]; then
        if command -v brew &> /dev/null; then
            brew install terraform
        else
            echo -e "${RED}❌ Homebrew not found. Please install Terraform manually:${NC}"
            echo "https://developer.hashicorp.com/terraform/downloads"
            exit 1
        fi
    else
        echo -e "${YELLOW}Please install Terraform manually:${NC}"
        echo "https://developer.hashicorp.com/terraform/downloads"
        exit 1
    fi
fi

# Check for AWS CLI
echo -e "${YELLOW}🔍 Checking for AWS CLI...${NC}"
if command -v aws &> /dev/null; then
    AWS_VERSION=$(aws --version)
    echo -e "${GREEN}✅ $AWS_VERSION${NC}"
    
    # Check if AWS is configured
    if aws sts get-caller-identity &> /dev/null; then
        AWS_ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
        AWS_REGION=$(aws configure get region)
        echo -e "${GREEN}✅ AWS configured for account: $AWS_ACCOUNT in region: $AWS_REGION${NC}"
    else
        echo -e "${YELLOW}⚠️  AWS CLI not configured. Please run: aws configure${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  AWS CLI not found. Please install it first.${NC}"
    if [[ "$OS" == "macOS" ]]; then
        echo "Run: brew install awscli"
    else
        echo "Visit: https://aws.amazon.com/cli/"
    fi
fi

# Check for jq (needed for import script)
echo -e "${YELLOW}🔍 Checking for jq...${NC}"
if command -v jq &> /dev/null; then
    echo -e "${GREEN}✅ jq is installed${NC}"
else
    echo -e "${YELLOW}⚠️  jq not found. Installing...${NC}"
    if [[ "$OS" == "macOS" ]]; then
        if command -v brew &> /dev/null; then
            brew install jq
        fi
    else
        echo -e "${YELLOW}Please install jq manually${NC}"
    fi
fi

# Navigate to terraform directory
cd terraform

# Initialize Terraform if not already done
if [ ! -d ".terraform" ]; then
    echo -e "${YELLOW}🔧 Initializing Terraform...${NC}"
    terraform init
else
    echo -e "${GREEN}✅ Terraform already initialized${NC}"
fi

# Create terraform.tfvars if it doesn't exist
if [ ! -f "terraform.tfvars" ]; then
    echo -e "${YELLOW}📝 Creating terraform.tfvars from example...${NC}"
    cp terraform.tfvars.example terraform.tfvars
    echo -e "${YELLOW}⚠️  Please edit terraform.tfvars with your specific values${NC}"
    echo -e "${BLUE}📍 Location: $(pwd)/terraform.tfvars${NC}"
else
    echo -e "${GREEN}✅ terraform.tfvars already exists${NC}"
fi

# Check if SSH key exists
SSH_KEY_PATH="$HOME/.ssh/griffin-connect"
if [ -f "$SSH_KEY_PATH" ]; then
    echo -e "${GREEN}✅ SSH key found at $SSH_KEY_PATH${NC}"
    
    # Check if public key exists
    if [ ! -f "$SSH_KEY_PATH.pub" ]; then
        echo -e "${YELLOW}🔑 Generating public key from private key...${NC}"
        ssh-keygen -y -f "$SSH_KEY_PATH" > "$SSH_KEY_PATH.pub"
        echo -e "${GREEN}✅ Public key created${NC}"
    fi
else
    echo -e "${YELLOW}⚠️  SSH key not found. You may need to:${NC}"
    echo "1. Copy your PEM key to $SSH_KEY_PATH"
    echo "2. Or generate a new key pair: ssh-keygen -t rsa -b 4096 -f $SSH_KEY_PATH"
fi

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                  🎉 Setup Complete! 🎉                      ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}📝 Next Steps:${NC}"
echo -e "${GREEN}1. Review and edit terraform.tfvars:${NC}"
echo -e "   ${BLUE}nano terraform.tfvars${NC}"
echo ""
echo -e "${GREEN}2. For existing infrastructure, run import:${NC}"
echo -e "   ${BLUE}./import-existing.sh${NC}"
echo ""
echo -e "${GREEN}3. For new infrastructure, run:${NC}"
echo -e "   ${BLUE}terraform plan${NC}"
echo -e "   ${BLUE}terraform apply${NC}"
echo ""

echo -e "${YELLOW}💡 Useful Commands:${NC}"
echo -e "${GREEN}  terraform plan${NC}     - Preview changes"
echo -e "${GREEN}  terraform apply${NC}    - Apply changes"
echo -e "${GREEN}  terraform output${NC}   - Show outputs"
echo -e "${GREEN}  terraform destroy${NC}  - Destroy infrastructure"
echo ""

echo -e "${BLUE}📚 Documentation: terraform/README.md${NC}" 
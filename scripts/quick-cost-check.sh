#!/bin/bash
# Quick Cost Check - Simplified version for macOS compatibility
# Shows current status and cost estimates for EPYC-testing instance

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration
INSTANCE_ID="i-00268dae9fd36421f"
REGION="us-west-1"

# Set AWS profile if not set
export AWS_PROFILE=${AWS_PROFILE:-epyc-testing}

echo -e "${BLUE}💰 EPYC-testing Quick Cost Check${NC}"
echo "=================================="

# Get instance status
echo -e "${YELLOW}🔍 Checking instance status...${NC}"
STATUS=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --region $REGION \
    --query 'Reservations[0].Instances[0].State.Name' \
    --output text 2>/dev/null)

if [ $? -eq 0 ]; then
    echo -e "Instance Status: ${GREEN}$STATUS${NC}"
else
    echo -e "${RED}❌ Failed to get instance status${NC}"
    exit 1
fi

# Cost estimates
HOURLY_RATE="4.61"
DAILY_COST="110.64"
MONTHLY_COST="3317.76"

echo ""
echo -e "${BLUE}💸 Cost Estimates (c6a.24xlarge):${NC}"
echo "  Hourly: \$$HOURLY_RATE"
echo "  Daily: \$$DAILY_COST" 
echo "  Monthly: \$$MONTHLY_COST"

if [ "$STATUS" = "running" ]; then
    echo ""
    echo -e "${RED}⚠️  WARNING: Instance is running and incurring costs!${NC}"
    echo -e "${YELLOW}💡 To save money, stop the instance when not in use:${NC}"
    echo "   aws ec2 stop-instances --instance-ids $INSTANCE_ID"
    echo ""
    echo -e "${YELLOW}🔄 To start it again later:${NC}"
    echo "   aws ec2 start-instances --instance-ids $INSTANCE_ID"
elif [ "$STATUS" = "stopped" ]; then
    echo ""
    echo -e "${GREEN}✅ Instance is stopped - no compute costs!${NC}"
    echo -e "${YELLOW}🔄 To start when needed:${NC}"
    echo "   aws ec2 start-instances --instance-ids $INSTANCE_ID"
else
    echo ""
    echo -e "${YELLOW}ℹ️  Instance status: $STATUS${NC}"
fi

echo ""
echo -e "${BLUE}🛠️  Management Tools:${NC}"
echo "  Full control: ./cost-control.sh"
echo "  Terraform:    cd ../terraform && terraform output"
echo "  SSH connect:  ssh -i ../griffin-connect.pem ubuntu@54.151.76.197" 
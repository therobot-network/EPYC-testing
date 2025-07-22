#!/bin/bash

# EC2 Connection Script
# Automatically connects to the EPYC-testing EC2 instance

# Load EC2 configuration from configs/ec2.yaml
CONFIG_FILE="configs/ec2.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ Error: Configuration file '$CONFIG_FILE' not found!"
    exit 1
fi

# Extract configuration using python
EC2_IP=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['ec2']['public_ip'])")
EC2_USER=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['connection']['user'])")
PEM_KEY=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['connection']['pem_key'])")
INSTANCE_ID=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['ec2']['instance_id'])")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 EPYC-testing EC2 Connection Script${NC}"
echo -e "${YELLOW}Instance ID: ${INSTANCE_ID}${NC}"
echo -e "${YELLOW}IP Address: ${EC2_IP}${NC}"
echo -e "${YELLOW}User: ${EC2_USER}${NC}"

# Check if PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo -e "${RED}❌ Error: PEM key file '$PEM_KEY' not found!${NC}"
    echo "Make sure you're running this script from the project root directory."
    exit 1
fi

# Check PEM key permissions
PEM_PERMS=$(stat -f "%A" "$PEM_KEY" 2>/dev/null || stat -c "%a" "$PEM_KEY" 2>/dev/null)
if [ "$PEM_PERMS" != "400" ]; then
    echo -e "${YELLOW}⚠️  Setting correct permissions for PEM key...${NC}"
    chmod 400 "$PEM_KEY"
fi

# Test connection first
echo -e "${YELLOW}🔍 Testing connection to EC2 instance...${NC}"
if ssh -i "$PEM_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$EC2_USER@$EC2_IP" exit 2>/dev/null; then
    echo -e "${GREEN}✅ Connection test successful!${NC}"
else
    echo -e "${RED}❌ Connection test failed. Checking possible issues:${NC}"
    echo "1. Instance might be stopped or starting up"
    echo "2. Security group might not allow SSH from your IP"
    echo "3. Username might be incorrect (current: $EC2_USER)"
    echo ""
    echo "Attempting connection anyway..."
fi

# Connect to EC2 instance
echo -e "${GREEN}🔗 Connecting to EC2 instance...${NC}"
echo -e "${YELLOW}To exit, type 'exit' or press Ctrl+D${NC}"
echo ""

ssh -i "$PEM_KEY" "$EC2_USER@$EC2_IP" 
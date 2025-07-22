#!/bin/bash

# Quick Sync to EC2 Script
# Quickly syncs local changes to EC2 for development

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
REMOTE_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['connection']['remote_directory'])")

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}⚡ Quick Sync to EC2${NC}"
echo -e "${YELLOW}Target: $EC2_USER@$EC2_IP:$REMOTE_DIR${NC}"

# Check if PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo "❌ Error: PEM key file '$PEM_KEY' not found!"
    exit 1
fi

# Quick sync of app directory and configs
echo -e "${BLUE}📦 Syncing changes...${NC}"
rsync -avz --progress \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='venv/' \
    --exclude='*.log' \
    -e "ssh -i $PEM_KEY" \
    ./app/ ./configs/ ./requirements.txt \
    "$EC2_USER@$EC2_IP:$REMOTE_DIR/"

echo -e "${GREEN}✅ Sync complete!${NC}"
echo -e "${YELLOW}💡 To restart the service on EC2:${NC}"
echo "ssh -i $PEM_KEY $EC2_USER@$EC2_IP 'cd $REMOTE_DIR && source venv/bin/activate && pkill -f \"python app/main.py\" && nohup python app/main.py > app.log 2>&1 &'" 
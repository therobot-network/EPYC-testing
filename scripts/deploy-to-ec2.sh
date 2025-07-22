#!/bin/bash

# Deploy to EC2 Script
# Syncs the project to EC2 and sets up the environment

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
REMOTE_DIR=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['connection']['remote_directory'])")

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}🚀 EPYC-testing EC2 Deployment Script${NC}"
echo -e "${YELLOW}Target: $EC2_USER@$EC2_IP:$REMOTE_DIR${NC}"
echo -e "${YELLOW}Instance ID: $INSTANCE_ID${NC}"

# Check if PEM key exists
if [ ! -f "$PEM_KEY" ]; then
    echo -e "${RED}❌ Error: PEM key file '$PEM_KEY' not found!${NC}"
    exit 1
fi

# Test connection
echo -e "${BLUE}🔍 Testing connection...${NC}"
if ! ssh -i "$PEM_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$EC2_USER@$EC2_IP" exit 2>/dev/null; then
    echo -e "${RED}❌ Cannot connect to EC2 instance. Please check:${NC}"
    echo "1. Instance is running"
    echo "2. Security group allows SSH"
    echo "3. Correct username (current: $EC2_USER)"
    exit 1
fi

echo -e "${GREEN}✅ Connection successful!${NC}"

# Create remote directory
echo -e "${BLUE}📁 Creating remote directory...${NC}"
ssh -i "$PEM_KEY" "$EC2_USER@$EC2_IP" "mkdir -p $REMOTE_DIR"

# Sync project files (excluding sensitive files and directories)
echo -e "${BLUE}📦 Syncing project files...${NC}"
rsync -avz --progress \
    --exclude='.git/' \
    --exclude='__pycache__/' \
    --exclude='*.pyc' \
    --exclude='.pytest_cache/' \
    --exclude='venv/' \
    --exclude='env/' \
    --exclude='.env' \
    --exclude='griffin-connect.pem' \
    --exclude='*.log' \
    --exclude='.DS_Store' \
    -e "ssh -i $PEM_KEY" \
    ./ "$EC2_USER@$EC2_IP:$REMOTE_DIR/"

# Setup environment on EC2
echo -e "${BLUE}🔧 Setting up environment on EC2...${NC}"
ssh -i "$PEM_KEY" "$EC2_USER@$EC2_IP" << EOF
cd $REMOTE_DIR

# Update system packages
echo "📦 Updating system packages..."
sudo yum update -y 2>/dev/null || sudo apt update -y 2>/dev/null || echo "Package manager not recognized"

# Install Python 3.9+ if not available
if ! python3 --version | grep -E "3\.[9-9]|3\.1[0-9]" > /dev/null 2>&1; then
    echo "🐍 Installing Python 3.9+..."
    sudo yum install -y python39 python39-pip 2>/dev/null || \
    sudo apt install -y python3.9 python3.9-venv python3.9-pip 2>/dev/null || \
    echo "Please install Python 3.9+ manually"
fi

# Install Docker if not present
if ! command -v docker &> /dev/null; then
    echo "🐳 Installing Docker..."
    sudo yum install -y docker 2>/dev/null || \
    sudo apt install -y docker.io 2>/dev/null
    
    sudo systemctl start docker 2>/dev/null || sudo service docker start 2>/dev/null
    sudo systemctl enable docker 2>/dev/null
    sudo usermod -aG docker \$USER
fi

# Install Docker Compose if not present
if ! command -v docker-compose &> /dev/null; then
    echo "🐳 Installing Docker Compose..."
    sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-\$(uname -s)-\$(uname -m)" -o /usr/local/bin/docker-compose
    sudo chmod +x /usr/local/bin/docker-compose
fi

# Create Python virtual environment
echo "🔧 Setting up Python virtual environment..."
python3 -m venv venv || python3.9 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Make scripts executable
chmod +x scripts/*.sh 2>/dev/null || echo "No scripts directory found"

echo "✅ Environment setup complete!"
echo "📍 Project location: $REMOTE_DIR"
echo "🔧 To activate virtual environment: source $REMOTE_DIR/venv/bin/activate"
echo "🚀 To run the application: cd $REMOTE_DIR && source venv/bin/activate && python app/main.py"
EOF

echo -e "${GREEN}✅ Deployment complete!${NC}"
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Connect to EC2: ./scripts/connect-ec2.sh"
echo "2. Navigate to project: cd $REMOTE_DIR"
echo "3. Activate environment: source venv/bin/activate"
echo "4. Run application: python app/main.py" 
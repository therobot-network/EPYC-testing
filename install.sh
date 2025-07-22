#!/bin/bash

# EPYC-testing Installation Script
# Sets up the complete development environment for EC2 integration

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# Configuration
CONFIG_FILE="configs/ec2.yaml"

if [ ! -f "$CONFIG_FILE" ]; then
    echo -e "${RED}❌ Error: Configuration file '$CONFIG_FILE' not found!${NC}"
    echo -e "${YELLOW}Please ensure configs/ec2.yaml exists with your EC2 configuration.${NC}"
    exit 1
fi

# Extract configuration using python
EC2_IP=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['ec2']['public_ip'])" 2>/dev/null)
EC2_USER=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['connection']['user'])" 2>/dev/null)
PEM_KEY=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['connection']['pem_key'])" 2>/dev/null)
INSTANCE_ID=$(python3 -c "import yaml; config=yaml.safe_load(open('$CONFIG_FILE')); print(config['ec2']['instance_id'])" 2>/dev/null)

if [ -z "$EC2_IP" ] || [ -z "$EC2_USER" ] || [ -z "$PEM_KEY" ] || [ -z "$INSTANCE_ID" ]; then
    echo -e "${RED}❌ Error: Could not read EC2 configuration from $CONFIG_FILE${NC}"
    echo -e "${YELLOW}Please ensure the file has the correct YAML structure with ec2, connection sections.${NC}"
    exit 1
fi

echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    EPYC-testing Setup                       ║"
echo "║              EC2 Integration & Development                   ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${BLUE}🚀 Starting installation...${NC}"

# Step 1: Check prerequisites
echo -e "${YELLOW}📋 Checking prerequisites...${NC}"

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

# Check for required tools
REQUIRED_TOOLS=("ssh" "rsync" "curl")
for tool in "${REQUIRED_TOOLS[@]}"; do
    if command -v "$tool" &> /dev/null; then
        echo -e "${GREEN}✅ $tool is installed${NC}"
    else
        echo -e "${RED}❌ $tool is not installed. Please install it first.${NC}"
        exit 1
    fi
done

# Check Python
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
    echo -e "${GREEN}✅ Python $PYTHON_VERSION is installed${NC}"
else
    echo -e "${RED}❌ Python 3 is not installed. Please install Python 3.8+ first.${NC}"
    exit 1
fi

# Step 2: Check PEM key
echo -e "${YELLOW}🔑 Checking PEM key...${NC}"
if [ -f "$PEM_KEY" ]; then
    echo -e "${GREEN}✅ PEM key found: $PEM_KEY${NC}"
    
    # Set correct permissions
    chmod 400 "$PEM_KEY"
    echo -e "${GREEN}✅ PEM key permissions set to 400${NC}"
else
    echo -e "${RED}❌ PEM key not found: $PEM_KEY${NC}"
    echo "Please make sure the PEM key is in the project root directory."
    exit 1
fi

# Step 3: Create directories
echo -e "${YELLOW}📁 Creating directories...${NC}"
mkdir -p scripts
mkdir -p logs
echo -e "${GREEN}✅ Directories created${NC}"

# Step 4: Make scripts executable
echo -e "${YELLOW}🔧 Setting up scripts...${NC}"
if [ -d "scripts" ]; then
    chmod +x scripts/*.sh 2>/dev/null || true
    echo -e "${GREEN}✅ Scripts made executable${NC}"
fi

# Step 5: Set up local Python environment
echo -e "${YELLOW}🐍 Setting up local Python environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✅ Virtual environment created${NC}"
else
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
fi

# Activate virtual environment and install dependencies
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
echo -e "${GREEN}✅ Dependencies installed${NC}"

# Step 6: Test EC2 connection
echo -e "${YELLOW}🔍 Testing EC2 connection...${NC}"
if ssh -i "$PEM_KEY" -o ConnectTimeout=10 -o BatchMode=yes "$EC2_USER@$EC2_IP" exit 2>/dev/null; then
    echo -e "${GREEN}✅ EC2 connection successful!${NC}"
    EC2_REACHABLE=true
else
    echo -e "${YELLOW}⚠️  Cannot reach EC2 instance. This might be normal if:${NC}"
    echo "   - Instance is stopped"
    echo "   - Security group doesn't allow SSH from your IP"
    echo "   - Username should be 'ubuntu' instead of 'ec2-user'"
    EC2_REACHABLE=false
fi

# Step 7: Create SSH config entry (optional)
echo -e "${YELLOW}🔧 Setting up SSH configuration...${NC}"
SSH_CONFIG_DIR="$HOME/.ssh"
SSH_CONFIG_FILE="$SSH_CONFIG_DIR/config"

# Create .ssh directory if it doesn't exist
mkdir -p "$SSH_CONFIG_DIR"
chmod 700 "$SSH_CONFIG_DIR"

# Add SSH config entry if it doesn't exist
if ! grep -q "Host epyc-testing" "$SSH_CONFIG_FILE" 2>/dev/null; then
    cat >> "$SSH_CONFIG_FILE" << EOF

# EPYC-testing EC2 Instance
Host epyc-testing
    HostName $EC2_IP
    User $EC2_USER
    IdentityFile $(pwd)/$PEM_KEY
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
EOF
    echo -e "${GREEN}✅ SSH config entry added${NC}"
else
    echo -e "${YELLOW}⚠️  SSH config entry already exists${NC}"
fi

# Step 8: Create convenience aliases
echo -e "${YELLOW}🔗 Creating convenience commands...${NC}"
cat > ec2-connect << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./scripts/connect-ec2.sh
EOF

cat > ec2-deploy << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./scripts/deploy-to-ec2.sh
EOF

cat > ec2-sync << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
./scripts/sync-to-ec2.sh
EOF

chmod +x ec2-connect ec2-deploy ec2-sync
echo -e "${GREEN}✅ Convenience commands created${NC}"

# Step 9: Installation complete
echo -e "${BOLD}${GREEN}"
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                    🎉 Installation Complete! 🎉             ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo -e "${NC}"

echo -e "${YELLOW}📋 Available Commands:${NC}"
echo -e "${GREEN}  ./ec2-connect${NC}     - Connect to EC2 instance"
echo -e "${GREEN}  ./ec2-deploy${NC}      - Deploy project to EC2"
echo -e "${GREEN}  ./ec2-sync${NC}        - Quick sync changes to EC2"
echo -e "${GREEN}  ssh epyc-testing${NC}  - Connect using SSH config"
echo ""

echo -e "${YELLOW}📁 Project Structure:${NC}"
echo -e "${GREEN}  scripts/connect-ec2.sh${NC}   - EC2 connection script"
echo -e "${GREEN}  scripts/deploy-to-ec2.sh${NC} - Full deployment script"
echo -e "${GREEN}  scripts/sync-to-ec2.sh${NC}   - Quick sync script"
echo -e "${GREEN}  venv/${NC}                    - Local Python environment"
echo ""

echo -e "${YELLOW}🚀 Quick Start:${NC}"
echo -e "${GREEN}  1. Connect to EC2:${NC} ./ec2-connect"
echo -e "${GREEN}  2. Deploy project:${NC} ./ec2-deploy"
echo -e "${GREEN}  3. Sync changes:${NC}   ./ec2-sync"
echo ""

if [ "$EC2_REACHABLE" = true ]; then
    echo -e "${GREEN}✅ EC2 instance is reachable and ready!${NC}"
    echo -e "${YELLOW}💡 Run './ec2-deploy' to deploy the project to EC2${NC}"
else
    echo -e "${YELLOW}⚠️  EC2 instance connection needs to be verified${NC}"
    echo -e "${YELLOW}💡 Check your EC2 instance status and security groups${NC}"
fi

echo ""
echo -e "${BLUE}Happy coding! 🚀${NC}" 
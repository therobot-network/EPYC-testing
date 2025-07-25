#!/bin/bash
set -e

# ProcessWatcher Agent Setup Script for Ubuntu EC2
# This script installs and configures the ProcessWatcher monitoring agent

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
AGENT_USER="processwatcher"
AGENT_HOME="/opt/processwatcher"
SERVICE_NAME="processwatcher-agent"
LOG_DIR="/var/log/processwatcher"
CONFIG_FILE="/etc/processwatcher/config.json"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if running as root
check_root() {
    if [[ $EUID -ne 0 ]]; then
        print_error "This script must be run as root (use sudo)"
        exit 1
    fi
}

# Function to detect Ubuntu version
detect_ubuntu() {
    if ! grep -q "Ubuntu" /etc/os-release; then
        print_error "This script is designed for Ubuntu systems"
        exit 1
    fi
    
    UBUNTU_VERSION=$(lsb_release -rs)
    print_status "Detected Ubuntu $UBUNTU_VERSION"
}

# Function to update system packages
update_system() {
    print_status "Updating system packages..."
    apt-get update -qq
    apt-get upgrade -y -qq
    print_success "System packages updated"
}

# Function to install dependencies
install_dependencies() {
    print_status "Installing system dependencies..."
    
    # Install Python 3 and pip if not already installed
    apt-get install -y python3 python3-pip python3-venv curl wget unzip jq
    
    # Install additional system monitoring tools
    apt-get install -y htop iotop nethogs sysstat
    
    print_success "System dependencies installed"
}

# Function to create system user
create_user() {
    print_status "Creating system user '$AGENT_USER'..."
    
    if id "$AGENT_USER" &>/dev/null; then
        print_warning "User '$AGENT_USER' already exists"
    else
        useradd --system --shell /bin/false --home-dir $AGENT_HOME --create-home $AGENT_USER
        print_success "User '$AGENT_USER' created"
    fi
}

# Function to create directory structure
create_directories() {
    print_status "Creating directory structure..."
    
    # Create main directories
    mkdir -p $AGENT_HOME
    mkdir -p $LOG_DIR
    mkdir -p /etc/processwatcher
    
    # Set permissions
    chown $AGENT_USER:$AGENT_USER $AGENT_HOME
    chown $AGENT_USER:$AGENT_USER $LOG_DIR
    chmod 755 $AGENT_HOME
    chmod 755 $LOG_DIR
    chmod 755 /etc/processwatcher
    
    print_success "Directory structure created"
}

# Function to install Python dependencies
install_python_deps() {
    print_status "Installing Python dependencies..."
    
    # Create virtual environment
    sudo -u $AGENT_USER python3 -m venv $AGENT_HOME/venv
    
    # Install requirements
    sudo -u $AGENT_USER $AGENT_HOME/venv/bin/pip install --upgrade pip
    sudo -u $AGENT_USER $AGENT_HOME/venv/bin/pip install psutil>=5.9.0 requests>=2.28.0
    
    print_success "Python dependencies installed"
}

# Function to copy agent files
copy_agent_files() {
    print_status "Copying agent files..."
    
    # Copy the monitor script
    cp monitor.py $AGENT_HOME/
    chown $AGENT_USER:$AGENT_USER $AGENT_HOME/monitor.py
    chmod 755 $AGENT_HOME/monitor.py
    
    print_success "Agent files copied"
}

# Function to create configuration file
create_config() {
    print_status "Creating configuration file..."
    
    # Get EC2 instance metadata
    INSTANCE_ID=""
    INSTANCE_TYPE=""
    AVAILABILITY_ZONE=""
    
    if curl -s --max-time 3 http://169.254.169.254/latest/meta-data/instance-id > /dev/null 2>&1; then
        INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id)
        INSTANCE_TYPE=$(curl -s http://169.254.169.254/latest/meta-data/instance-type)
        AVAILABILITY_ZONE=$(curl -s http://169.254.169.254/latest/meta-data/placement/availability-zone)
        print_status "EC2 metadata detected: $INSTANCE_ID ($INSTANCE_TYPE) in $AVAILABILITY_ZONE"
    else
        print_warning "Not running on EC2 or metadata service unavailable"
        INSTANCE_ID=$(hostname)
    fi
    
    # Create config file
    cat > $CONFIG_FILE << EOF
{
    "api_url": "https://your-api-server.com",
    "agent_id": "",
    "instance_id": "$INSTANCE_ID",
    "instance_type": "$INSTANCE_TYPE",
    "availability_zone": "$AVAILABILITY_ZONE",
    "collection_interval": 30,
    "process_limit": 50,
    "log_level": "INFO",
    "auto_register": true,
    "retry_attempts": 3,
    "retry_delay": 5
}
EOF
    
    chmod 644 $CONFIG_FILE
    print_success "Configuration file created at $CONFIG_FILE"
}

# Function to create systemd service
create_service() {
    print_status "Creating systemd service..."
    
    cat > /etc/systemd/system/${SERVICE_NAME}.service << EOF
[Unit]
Description=ProcessWatcher Monitoring Agent
After=network.target
Wants=network.target

[Service]
Type=simple
User=$AGENT_USER
Group=$AGENT_USER
WorkingDirectory=$AGENT_HOME
Environment=PATH=$AGENT_HOME/venv/bin
ExecStart=$AGENT_HOME/venv/bin/python $AGENT_HOME/monitor.py --config $CONFIG_FILE
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal
SyslogIdentifier=processwatcher-agent

# Security settings
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ProtectHome=true
ReadWritePaths=$LOG_DIR /tmp
ProtectKernelTunables=true
ProtectKernelModules=true
ProtectControlGroups=true

[Install]
WantedBy=multi-user.target
EOF
    
    # Reload systemd and enable service
    systemctl daemon-reload
    systemctl enable $SERVICE_NAME
    
    print_success "Systemd service created and enabled"
}

# Function to create logrotate configuration
create_logrotate() {
    print_status "Creating log rotation configuration..."
    
    cat > /etc/logrotate.d/processwatcher << EOF
$LOG_DIR/*.log {
    daily
    missingok
    rotate 7
    compress
    delaycompress
    notifempty
    create 644 $AGENT_USER $AGENT_USER
    postrotate
        systemctl reload $SERVICE_NAME > /dev/null 2>&1 || true
    endscript
}
EOF
    
    print_success "Log rotation configured"
}

# Function to create management script
create_management_script() {
    print_status "Creating management script..."
    
    cat > /usr/local/bin/processwatcher << 'EOF'
#!/bin/bash

SERVICE_NAME="processwatcher-agent"
CONFIG_FILE="/etc/processwatcher/config.json"

case "$1" in
    start)
        echo "Starting ProcessWatcher agent..."
        systemctl start $SERVICE_NAME
        ;;
    stop)
        echo "Stopping ProcessWatcher agent..."
        systemctl stop $SERVICE_NAME
        ;;
    restart)
        echo "Restarting ProcessWatcher agent..."
        systemctl restart $SERVICE_NAME
        ;;
    status)
        systemctl status $SERVICE_NAME
        ;;
    logs)
        journalctl -u $SERVICE_NAME -f
        ;;
    config)
        if [ -n "$2" ]; then
            case "$2" in
                edit)
                    ${EDITOR:-nano} $CONFIG_FILE
                    echo "Configuration updated. Restart the service to apply changes."
                    ;;
                show)
                    cat $CONFIG_FILE
                    ;;
                *)
                    echo "Usage: processwatcher config {edit|show}"
                    ;;
            esac
        else
            echo "Usage: processwatcher config {edit|show}"
        fi
        ;;
    update)
        echo "Updating ProcessWatcher agent..."
        # This would download and install updates
        echo "Update functionality not implemented yet"
        ;;
    *)
        echo "Usage: processwatcher {start|stop|restart|status|logs|config|update}"
        echo ""
        echo "Commands:"
        echo "  start    - Start the agent service"
        echo "  stop     - Stop the agent service"
        echo "  restart  - Restart the agent service"
        echo "  status   - Show service status"
        echo "  logs     - Show live logs"
        echo "  config   - Manage configuration (edit|show)"
        echo "  update   - Update the agent (not implemented)"
        exit 1
        ;;
esac
EOF
    
    chmod +x /usr/local/bin/processwatcher
    print_success "Management script created at /usr/local/bin/processwatcher"
}

# Function to prompt for configuration
configure_agent() {
    print_status "Configuring agent settings..."
    
    echo ""
    echo "Please provide the following information:"
    
    # API URL
    read -p "API Server URL (e.g., https://your-server.com): " API_URL
    if [ -n "$API_URL" ]; then
        sed -i "s|\"api_url\": \".*\"|\"api_url\": \"$API_URL\"|" $CONFIG_FILE
    fi
    
    # Collection interval
    read -p "Collection interval in seconds (default: 30): " INTERVAL
    if [ -n "$INTERVAL" ]; then
        sed -i "s|\"collection_interval\": .*|\"collection_interval\": $INTERVAL,|" $CONFIG_FILE
    fi
    
    print_success "Agent configured"
}

# Function to perform final setup
final_setup() {
    print_status "Performing final setup..."
    
    # Test configuration
    if sudo -u $AGENT_USER $AGENT_HOME/venv/bin/python -c "import json; json.load(open('$CONFIG_FILE'))" 2>/dev/null; then
        print_success "Configuration file is valid"
    else
        print_error "Configuration file is invalid"
        exit 1
    fi
    
    # Start the service
    systemctl start $SERVICE_NAME
    
    # Check if service started successfully
    sleep 2
    if systemctl is-active --quiet $SERVICE_NAME; then
        print_success "ProcessWatcher agent started successfully"
    else
        print_error "Failed to start ProcessWatcher agent"
        print_status "Check logs with: journalctl -u $SERVICE_NAME"
        exit 1
    fi
    
    print_success "Setup completed successfully!"
}

# Function to show completion message
show_completion() {
    echo ""
    echo "=========================================="
    print_success "ProcessWatcher Agent Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Service Status:"
    systemctl status $SERVICE_NAME --no-pager -l
    echo ""
    echo "Management Commands:"
    echo "  processwatcher start    - Start the agent"
    echo "  processwatcher stop     - Stop the agent"
    echo "  processwatcher status   - Check status"
    echo "  processwatcher logs     - View logs"
    echo "  processwatcher config   - Manage configuration"
    echo ""
    echo "Configuration file: $CONFIG_FILE"
    echo "Log directory: $LOG_DIR"
    echo "Service name: $SERVICE_NAME"
    echo ""
    print_status "The agent is now running and will start automatically on boot"
}

# Main execution
main() {
    echo "ProcessWatcher Agent Setup Script"
    echo "================================="
    echo ""
    
    check_root
    detect_ubuntu
    update_system
    install_dependencies
    create_user
    create_directories
    install_python_deps
    copy_agent_files
    create_config
    create_service
    create_logrotate
    create_management_script
    configure_agent
    final_setup
    show_completion
}

# Run main function
main "$@" 
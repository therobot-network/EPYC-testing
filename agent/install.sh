#!/bin/bash
# ProcessWatcher Agent One-Line Installer
# Usage: curl -sSL https://raw.githubusercontent.com/your-repo/ProcessWatcher/main/web/agent/install.sh | sudo bash

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

print_status() {
    echo -e "${BLUE}[INSTALLER]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[INSTALLER]${NC} $1"
}

print_error() {
    echo -e "${RED}[INSTALLER]${NC} $1"
}

# Check if running as root
if [[ $EUID -ne 0 ]]; then
    print_error "This installer must be run as root (use sudo)"
    echo "Usage: curl -sSL https://raw.githubusercontent.com/your-repo/ProcessWatcher/main/web/agent/install.sh | sudo bash"
    exit 1
fi

print_status "ProcessWatcher Agent Installer"
print_status "=============================="

# Create temporary directory
TEMP_DIR=$(mktemp -d)
cd "$TEMP_DIR"

print_status "Downloading agent files..."

# Download the required files
# Note: Replace these URLs with your actual repository URLs
REPO_BASE="https://raw.githubusercontent.com/your-repo/ProcessWatcher/main/web/agent"

if ! curl -sSL "$REPO_BASE/setup.sh" -o setup.sh; then
    print_error "Failed to download setup.sh"
    exit 1
fi

if ! curl -sSL "$REPO_BASE/monitor.py" -o monitor.py; then
    print_error "Failed to download monitor.py"
    exit 1
fi

if ! curl -sSL "$REPO_BASE/requirements.txt" -o requirements.txt; then
    print_error "Failed to download requirements.txt"
    exit 1
fi

if ! curl -sSL "$REPO_BASE/config.template.json" -o config.template.json; then
    print_error "Failed to download config.template.json"
    exit 1
fi

print_success "Files downloaded successfully"

# Make setup script executable
chmod +x setup.sh

print_status "Running setup script..."

# Run the setup script
./setup.sh

print_success "Installation completed!"

# Cleanup
cd /
rm -rf "$TEMP_DIR"

print_status "Temporary files cleaned up"
print_status "Agent installation complete!" 
#!/bin/bash
# ProcessWatcher Terminal Logging Setup
#
# This script sets up your terminal session for automatic ProcessWatcher logging
# 
# Usage: source setup_terminal_logging.sh <api_url>
# Example: source setup_terminal_logging.sh https://your-processwatcher-url.com

API_URL="$1"
if [ -z "$API_URL" ]; then
    echo "Usage: source $0 <api_url>"
    echo "Example: source $0 https://your-processwatcher-url.com"
    return 1 2>/dev/null || exit 1
fi

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Set up environment variables
export PROCESSWATCHER_API_URL="$API_URL"
export PROCESSWATCHER_AGENT_ID="terminal-$(hostname)-$$"
export PROCESSWATCHER_APP_NAME="terminal-session"

# Make pwlog executable
chmod +x "$SCRIPT_DIR/pwlog"

# Create convenient aliases
alias pwlog="$SCRIPT_DIR/pwlog"
alias run="$SCRIPT_DIR/pwlog"
alias log="$SCRIPT_DIR/pwlog"

# Start background monitoring (optional)
if command -v python3 >/dev/null 2>&1; then
    echo "Starting ProcessWatcher background monitoring..."
    python3 "$SCRIPT_DIR/start_monitoring.py" --api-url "$API_URL" --session-name "terminal-$(hostname)-$$" &
    MONITOR_PID=$!
    export PROCESSWATCHER_MONITOR_PID=$MONITOR_PID
    
    # Function to stop monitoring when terminal exits
    cleanup_processwatcher() {
        if [ ! -z "$PROCESSWATCHER_MONITOR_PID" ]; then
            echo "Stopping ProcessWatcher monitoring..."
            kill $PROCESSWATCHER_MONITOR_PID 2>/dev/null
        fi
    }
    trap cleanup_processwatcher EXIT
fi

echo "ProcessWatcher Terminal Logging Setup Complete!"
echo "=========================================="
echo "API URL: $PROCESSWATCHER_API_URL"
echo "Agent ID: $PROCESSWATCHER_AGENT_ID"
echo "Session: $PROCESSWATCHER_APP_NAME"
echo ""
echo "Available commands:"
echo "  pwlog <command>  - Run command with ProcessWatcher logging"
echo "  run <command>    - Alias for pwlog"
echo "  log <command>    - Alias for pwlog"
echo ""
echo "Examples:"
echo "  run ls -la"
echo "  run python3 my_script.py"
echo "  log curl https://api.example.com"
echo ""
echo "All command output will be automatically sent to ProcessWatcher!"
echo "View logs at: $PROCESSWATCHER_API_URL" 
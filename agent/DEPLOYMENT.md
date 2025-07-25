# ProcessWatcher Agent Deployment Guide

## Quick Deployment Options

### Option 1: Terminal Session Logging (NEW - Recommended for Development)

Perfect for development and testing - automatically logs all your terminal commands and output:

```bash
# Download the agent files
git clone https://github.com/GriffinCanCode/ProcessWatcher.git
cd ProcessWatcher/web/agent

# Install dependencies
pip3 install -r requirements.txt

# Set up terminal logging for current session
source setup_terminal_logging.sh https://your-processwatcher-url.com
```

Now any command you run will be automatically logged:
```bash
run ls -la                    # Logs command and output
run python3 my_script.py     # Logs script execution and output
log curl https://api.com     # Alternative alias
```

### Option 2: Background Monitoring

Run system monitoring in the background:

```bash
# Start background monitoring
python3 start_monitoring.py --api-url https://your-processwatcher-url.com &

# Continue using terminal normally - system metrics are automatically collected
```

### Option 3: Full System Installation (Production)

For production environments with systemd service:

```bash
curl -sSL https://raw.githubusercontent.com/GriffinCanCode/ProcessWatcher/main/web/agent/install.sh | sudo bash
```

### Option 4: Manual Deployment

1. **Connect to your EC2 instance:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-instance.com
   ```

2. **Download and run setup:**
   ```bash
   wget https://raw.githubusercontent.com/GriffinCanCode/ProcessWatcher/main/web/agent/setup.sh
   chmod +x setup.sh
   sudo ./setup.sh
   ```

3. **Configure the agent:**
   During setup, you'll be prompted for:
   - API Server URL (e.g., `https://your-processwatcher-server.com`)
   - Collection interval (default: 30 seconds)

## Usage Examples

### Terminal Session Logging

After running `source setup_terminal_logging.sh`, you can use these commands:

```bash
# Basic commands
run ls -la
run pwd
run whoami

# Run scripts and capture output
run python3 my_script.py
run node app.js
run ./my_binary

# Network commands
run curl -s https://api.github.com/users/octocat
run wget https://example.com/file.txt

# Long-running commands
run tail -f /var/log/syslog    # Ctrl+C to stop
run python3 -m http.server     # Starts server, logs when stopped
```

All command output, errors, and execution details are automatically sent to ProcessWatcher!

### What Gets Logged

**Terminal Session Logging captures:**
- Command execution start/end times
- Full command line with arguments
- Standard output (stdout)
- Standard error (stderr)  
- Exit codes
- Execution duration
- Working directory
- User and hostname

**Background Monitoring captures:**
- System metrics (CPU, memory, disk)
- Running processes
- System load and uptime
- Process resource usage

### Background Monitoring

```bash
# Start monitoring in background
python3 start_monitoring.py --api-url https://your-url.com &

# Check if it's running
ps aux | grep start_monitoring

# Stop monitoring
kill %1  # or find PID and kill
```

### Production Installation Post-Setup

1. **Verify installation:**
   ```bash
   processwatcher status
   ```

2. **Check logs:**
   ```bash
   processwatcher logs
   ```

3. **Edit configuration if needed:**
   ```bash
   processwatcher config edit
   processwatcher restart
   ```

## Bulk Deployment

### Using AWS Systems Manager

Create a Systems Manager document to deploy to multiple instances:

```json
{
  "schemaVersion": "2.2",
  "description": "Install ProcessWatcher Agent",
  "mainSteps": [
    {
      "action": "aws:runShellScript",
      "name": "installProcessWatcher",
      "inputs": {
        "runCommand": [
          "curl -sSL https://raw.githubusercontent.com/GriffinCanCode/ProcessWatcher/main/web/agent/install.sh | sudo bash"
        ]
      }
    }
  ]
}
```

### Using Ansible

```yaml
---
- name: Deploy ProcessWatcher Agent
  hosts: ec2_instances
  become: yes
  tasks:
    - name: Download and run installer
      shell: |
        curl -sSL https://raw.githubusercontent.com/GriffinCanCode/ProcessWatcher/main/web/agent/install.sh | bash
```

### Using Terraform

```hcl
resource "aws_instance" "monitored_instance" {
  ami           = "ami-0c55b159cbfafe1d0"
  instance_type = "t3.micro"
  
  user_data = <<-EOF
    #!/bin/bash
    curl -sSL https://raw.githubusercontent.com/your-repo/ProcessWatcher/main/web/agent/install.sh | bash
  EOF
  
  tags = {
    Name = "ProcessWatcher-Monitored"
  }
}
```

## Configuration Management

### Environment-Specific Configurations

Create different configuration templates for different environments:

**Production (`/etc/processwatcher/config.json`):**
```json
{
    "api_url": "https://prod-processwatcher.company.com",
    "collection_interval": 60,
    "process_limit": 30,
    "log_level": "WARNING"
}
```

**Development (`/etc/processwatcher/config.json`):**
```json
{
    "api_url": "https://dev-processwatcher.company.com",
    "collection_interval": 15,
    "process_limit": 100,
    "log_level": "DEBUG"
}
```

### Automated Configuration

Use a configuration management script:

```bash
#!/bin/bash
# configure-agent.sh

API_URL="$1"
ENVIRONMENT="$2"

if [ -z "$API_URL" ]; then
    echo "Usage: $0 <api_url> [environment]"
    exit 1
fi

# Update configuration
sudo tee /etc/processwatcher/config.json > /dev/null <<EOF
{
    "api_url": "$API_URL",
    "collection_interval": 30,
    "process_limit": 50,
    "log_level": "INFO",
    "auto_register": true,
    "retry_attempts": 3,
    "retry_delay": 5
}
EOF

# Restart service
sudo processwatcher restart

echo "Agent configured for $ENVIRONMENT environment"
```

## Monitoring the Deployment

### Health Check Script

```bash
#!/bin/bash
# health-check.sh

echo "ProcessWatcher Agent Health Check"
echo "================================="

# Check service status
if systemctl is-active --quiet processwatcher-agent; then
    echo "✓ Service is running"
else
    echo "✗ Service is not running"
    exit 1
fi

# Check recent logs for errors
if journalctl -u processwatcher-agent --since "5 minutes ago" | grep -q ERROR; then
    echo "✗ Recent errors found in logs"
    journalctl -u processwatcher-agent --since "5 minutes ago" | grep ERROR
    exit 1
else
    echo "✓ No recent errors in logs"
fi

# Check configuration
if python3 -m json.tool /etc/processwatcher/config.json > /dev/null 2>&1; then
    echo "✓ Configuration file is valid"
else
    echo "✗ Configuration file is invalid"
    exit 1
fi

echo "✓ All checks passed"
```

## Troubleshooting Common Deployment Issues

### Permission Issues

```bash
# Fix permissions
sudo chown -R processwatcher:processwatcher /opt/processwatcher
sudo chown -R processwatcher:processwatcher /var/log/processwatcher
sudo chmod 644 /etc/processwatcher/config.json
```

### Network Connectivity

```bash
# Test API connectivity
curl -v https://your-api-server.com/api/v1/health

# Check firewall
sudo ufw status

# Check DNS resolution
nslookup your-api-server.com
```

### Service Issues

```bash
# Check service status
systemctl status processwatcher-agent

# View detailed logs
journalctl -u processwatcher-agent -f

# Restart service
sudo systemctl restart processwatcher-agent
```

## Security Considerations

1. **API Endpoint Security**: Ensure your API server uses HTTPS
2. **Network Access**: Consider using VPC endpoints or private networks
3. **Instance Access**: Limit SSH access to the instances
4. **Log Security**: Ensure log files don't contain sensitive information
5. **Updates**: Regularly update the agent and system packages

## Scaling Considerations

- **Collection Interval**: Increase for large fleets to reduce API load
- **Process Limit**: Adjust based on instance types and requirements
- **Log Rotation**: Ensure adequate log rotation for long-running instances
- **Resource Usage**: Monitor agent resource consumption on smaller instances 
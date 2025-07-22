#!/bin/bash
# User Data Script for EPYC-testing EC2 Instance
# This script runs on instance launch to set up the environment

set -e

# Update system packages
yum update -y

# Install required packages
yum install -y \
    git \
    curl \
    wget \
    unzip \
    htop \
    tree \
    rsync \
    python3 \
    python3-pip \
    python3-devel \
    gcc \
    gcc-c++ \
    make

# Install Docker
yum install -y docker
systemctl start docker
systemctl enable docker
usermod -aG docker ${app_user}

# Install Docker Compose
curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Install CloudWatch agent (optional)
wget https://s3.amazonaws.com/amazoncloudwatch-agent/amazon_linux/amd64/latest/amazon-cloudwatch-agent.rpm
rpm -U ./amazon-cloudwatch-agent.rpm

# Create application directory
mkdir -p /home/${app_user}/${project_name}
chown ${app_user}:${app_user} /home/${app_user}/${project_name}

# Create logs directory
mkdir -p /var/log/${project_name}
chown ${app_user}:${app_user} /var/log/${project_name}

# Set up Python environment
python3 -m pip install --upgrade pip
python3 -m pip install virtualenv

# Create systemd service for the application (template)
cat > /etc/systemd/system/${project_name}.service << EOF
[Unit]
Description=${project_name} ML Model Service
After=network.target

[Service]
Type=simple
User=${app_user}
WorkingDirectory=/home/${app_user}/${project_name}
Environment=PATH=/home/${app_user}/${project_name}/venv/bin
ExecStart=/home/${app_user}/${project_name}/venv/bin/python app/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Enable the service (but don't start it yet - will be started after deployment)
systemctl daemon-reload
systemctl enable ${project_name}

# Create log rotation configuration
cat > /etc/logrotate.d/${project_name} << EOF
/var/log/${project_name}/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 ${app_user} ${app_user}
}
EOF

# Set up CloudWatch agent configuration (optional)
cat > /opt/aws/amazon-cloudwatch-agent/etc/amazon-cloudwatch-agent.json << EOF
{
    "logs": {
        "logs_collected": {
            "files": {
                "collect_list": [
                    {
                        "file_path": "/var/log/${project_name}/*.log",
                        "log_group_name": "/aws/ec2/${project_name}",
                        "log_stream_name": "{instance_id}"
                    }
                ]
            }
        }
    },
    "metrics": {
        "namespace": "AWS/EC2/${project_name}",
        "metrics_collected": {
            "cpu": {
                "measurement": [
                    "cpu_usage_idle",
                    "cpu_usage_iowait",
                    "cpu_usage_user",
                    "cpu_usage_system"
                ],
                "metrics_collection_interval": 60
            },
            "disk": {
                "measurement": [
                    "used_percent"
                ],
                "metrics_collection_interval": 60,
                "resources": [
                    "*"
                ]
            },
            "mem": {
                "measurement": [
                    "mem_used_percent"
                ],
                "metrics_collection_interval": 60
            }
        }
    }
}
EOF

# Signal that user data script completed successfully
echo "User data script completed successfully" > /var/log/user-data.log 
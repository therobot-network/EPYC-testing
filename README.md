# EPYC-testing: Model Deployment Repository

A scalable architecture for deploying machine learning models on AWS EC2 instances with automated terminal access and deployment tools.

## 🚀 Quick Setup

### One-Command Installation
```bash
./install.sh
```

This will:
- ✅ Check prerequisites (Python, SSH, rsync)
- 🔑 Configure PEM key permissions
- 🐍 Set up Python virtual environment
- 📦 Install dependencies
- 🔧 Create SSH configuration
- 🚀 Test EC2 connection
- 📋 Create convenience commands

### EC2 Instance Information
Configuration is now loaded dynamically from `configs/ec2.yaml`:
- **Instance ID**: `i-07784f133e33f426c` (from config)
- **Public IP**: `52.53.198.86` (from config)
- **Private IP**: `172.31.5.154` (from config)
- **Instance Type**: `c6a.large` (from config)
- **Region**: `us-west-1` (from config)
- **Availability Zone**: `us-west-1c` (from config)

## 🛠️ Available Commands

After installation, use these convenient commands:

```bash
# Connect to EC2 instance
./ec2-connect

# Deploy entire project to EC2
./ec2-deploy  

# Quick sync changes during development
./ec2-sync

# Alternative SSH connection (after setup)
ssh epyc-testing
```

## 📁 Repository Structure

```
├── app/                    # Main application code
│   ├── api/               # API endpoints and routes
│   ├── models/            # Model loading and inference logic
│   ├── utils/             # Utility functions and helpers
│   └── config/            # Application configuration
├── scripts/               # EC2 automation scripts
│   ├── connect-ec2.sh     # Connect to EC2 instance
│   ├── deploy-to-ec2.sh   # Full deployment to EC2
│   └── sync-to-ec2.sh     # Quick sync for development
├── configs/               # Configuration files
├── logs/                  # Log files
├── venv/                  # Python virtual environment
├── griffin-connect.pem    # EC2 PEM key (keep secure!)
├── install.sh             # Main setup script
├── ec2-connect            # Convenience connection command
├── ec2-deploy             # Convenience deployment command
└── ec2-sync               # Convenience sync command
```

## 🔄 Development Workflow

### Initial Setup
1. **Clone and Install**:
   ```bash
   git clone <repository-url>
   cd EPYC-testing
   ./install.sh
   ```

2. **Deploy to EC2**:
   ```bash
   ./ec2-deploy
   ```

### Daily Development
1. **Make local changes** to your code
2. **Quick sync** to EC2:
   ```bash
   ./ec2-sync
   ```
3. **Connect to EC2** to test:
   ```bash
   ./ec2-connect
   ```

### Full Deployment
When you have major changes or want to reset the EC2 environment:
```bash
./ec2-deploy
```

## 🐍 Local Development

```bash
# Activate virtual environment
source venv/bin/activate

# Run locally
python app/main.py

# Run tests
pytest

# Format code
black app/
```

## 🐳 Docker Deployment

```bash
# Local Docker
docker-compose up --build

# On EC2 (after connecting)
cd ~/EPYC-testing
docker-compose up --build
```

## ⚙️ Configuration

The application now uses a dual configuration approach:

### EC2-Specific Configuration
The `configs/ec2.yaml` file contains EC2-specific settings that are automatically loaded:
- Instance details (ID, IPs, type, region)
- Connection settings (user, PEM key, SSH timeout)
- Performance optimizations (memory, CPU, batch sizes)
- Security settings (allowed hosts, CORS origins)
- Storage paths for the EC2 environment

### General Application Configuration
1. **Copy example config**:
   ```bash
   cp configs/config.example.yaml configs/config.yaml
   ```

2. **Update configuration** values for your environment

3. **Set environment variables** in `.env` file

The application will automatically use EC2-specific settings when available, falling back to general configuration defaults.

## 🌐 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Model inference  
- `GET /models` - List available models
- `POST /models/load` - Load a specific model

## 🔧 Environment Variables

- `MODEL_PATH`: Path to model files (auto-configured from EC2 config)
- `HOST`: Server host (default: 0.0.0.0, auto-configured from EC2 config)
- `PORT`: Port for the API server (default: 8000, auto-configured from EC2 config)
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR, auto-configured from EC2 config)
- `AWS_REGION`: AWS region for deployment (auto-configured from EC2 config: us-west-1)

## 🔐 Security Notes

- The `griffin-connect.pem` file contains your private key
- Never commit PEM files to version control
- The install script sets proper permissions (400) automatically
- SSH config uses `StrictHostKeyChecking no` for convenience

## 🚨 Troubleshooting

### Connection Issues
```bash
# Check EC2 instance status
aws ec2 describe-instances --instance-ids i-07784f133e33f426c

# Test connection manually
ssh -i griffin-connect.pem ec2-user@52.53.198.86

# Try alternative username
ssh -i griffin-connect.pem ubuntu@52.53.198.86
```

### Permission Issues
```bash
# Fix PEM key permissions
chmod 400 griffin-connect.pem

# Reinstall if needed
./install.sh
```

### Sync Issues
```bash
# Force full redeployment
./ec2-deploy

# Check rsync manually
rsync -avz --dry-run -e "ssh -i griffin-connect.pem" ./ ec2-user@52.53.198.86:~/EPYC-testing/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and test locally
4. Use `./ec2-sync` to test on EC2
5. Add tests and documentation
6. Submit a pull request

## 📄 License

MIT License 
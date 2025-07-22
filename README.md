# EPYC-testing: Large Language Model Deployment Repository

A scalable architecture for deploying large language models (specifically Llama 3.3 70B) on high-performance AWS EC2 instances with automated deployment and management tools.

## 🚀 Quick Setup

### One-Command Installation
```bash
./install.sh
```

This will:
- ✅ Check prerequisites (Python, SSH, rsync, PyYAML)
- 🔑 Configure PEM key permissions  
- 🐍 Set up Python virtual environment
- 📦 Install dependencies
- 🔧 Create SSH configuration
- 🚀 Test EC2 connection

### EC2 Instance Information
Configuration is loaded dynamically from `configs/ec2.yaml`:
- **Instance ID**: `i-00268dae9fd36421f` (from config)
- **Public IP**: `54.151.76.197` (from config)
- **Private IP**: `172.31.11.209` (from config)  
- **Instance Type**: `c6a.24xlarge` (96 vCPUs, 192GB RAM)
- **Region**: `us-west-1` (from config)
- **Availability Zone**: `us-west-1c` (from config)

## 🛠️ Available Scripts

Use these scripts for EC2 management and deployment:

```bash
# Connect to EC2 instance
./scripts/connect-ec2.sh

# Full deployment to EC2
./scripts/deploy-to-ec2.sh  

# Quick sync changes during development
./scripts/sync-to-ec2.sh

# Cost monitoring and control
./scripts/quick-cost-check.sh
./scripts/cost-control.sh

# Llama 3.3 model installation
./scripts/install_llama33.sh

# Terraform infrastructure management
./scripts/terraform-setup.sh
```

## 📁 Repository Structure

```
├── app/                    # Main application code
│   ├── api/               # FastAPI endpoints and routes
│   │   └── routes.py      # API routes (/health, /predict, /models, /chat)
│   ├── models/            # Model implementations
│   │   ├── manager.py     # Model loading and management
│   │   ├── llama33_model.py # Llama 3.3 70B implementation
│   │   ├── transformers_model.py # Generic transformers support
│   │   └── base.py        # Base model interface
│   ├── config/            # Application configuration
│   │   └── settings.py    # Dynamic settings with EC2 config integration
│   ├── utils/             # Utility functions
│   └── main.py            # FastAPI application entry point
├── scripts/               # EC2 automation and management scripts
│   ├── connect-ec2.sh     # SSH connection to EC2
│   ├── deploy-to-ec2.sh   # Full project deployment
│   ├── sync-to-ec2.sh     # Quick development sync
│   ├── cost-control.sh    # Interactive cost management
│   ├── quick-cost-check.sh # Quick cost status check
│   ├── install_llama33.sh # Llama 3.3 70B installation
│   ├── terraform-setup.sh # Terraform initialization
│   ├── fix_dependencies.sh # Dependency troubleshooting
│   └── fix_hf_token.sh    # HuggingFace token fixes
├── terraform/             # Infrastructure as Code
│   ├── main.tf            # Main Terraform configuration
│   ├── variables.tf       # Variable definitions
│   ├── outputs.tf         # Output definitions
│   └── import-existing.sh # Import existing resources
├── configs/               # Configuration files
│   ├── ec2.yaml          # EC2-specific configuration
│   └── config.example.yaml # General config template
├── docs/                  # Documentation
│   └── LLAMA33_SETUP.md   # Llama 3.3 setup guide
├── griffin-connect.pem    # EC2 PEM key (keep secure!)
├── install.sh             # Main setup script
├── docker-compose.yml     # Docker deployment configuration
├── Dockerfile             # Container configuration
└── requirements.txt       # Python dependencies
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
   ./scripts/deploy-to-ec2.sh
   ```

### Daily Development
1. **Make local changes** to your code
2. **Quick sync** to EC2:
   ```bash
   ./scripts/sync-to-ec2.sh
   ```
3. **Connect to EC2** to test:
   ```bash
   ./scripts/connect-ec2.sh
   ```

### Model Installation (Llama 3.3 70B)
```bash
# Install Llama 3.3 70B model
./scripts/install_llama33.sh
```

### Cost Management
```bash
# Quick cost check
./scripts/quick-cost-check.sh

# Interactive cost control
./scripts/cost-control.sh
```

## 🐍 Local Development

```bash
# Activate virtual environment
source venv/bin/activate

# Run locally (after model installation)
python app/main.py

# Run with specific host/port
HOST=0.0.0.0 PORT=8000 python app/main.py
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

The application uses a dual configuration approach:

### EC2-Specific Configuration
The `configs/ec2.yaml` file contains EC2-optimized settings:
- Instance details (ID, IPs, type, region)
- Connection settings (user, PEM key, SSH timeout)
- Performance optimizations (memory: 192GB, CPU: 96 cores)
- Security settings (allowed hosts, CORS origins)
- Storage paths for the EC2 environment

### General Application Configuration
1. **Copy example config**:
   ```bash
   cp configs/config.example.yaml configs/config.yaml
   ```

2. **Set environment variables** in `.env` file for sensitive data

The application automatically uses EC2-specific settings when available, falling back to general configuration defaults.

## 🌐 API Endpoints

### Core Endpoints
- `GET /api/v1/health` - Health check
- `POST /api/v1/predict` - Model inference
- `GET /api/v1/models` - List loaded models
- `POST /api/v1/models/load` - Load a specific model
- `DELETE /api/v1/models/{model_name}` - Unload a model

### Chat Interface (Llama 3.3)
- `POST /api/v1/chat` - Conversational interface with message history

### Example Usage
```bash
# Health check
curl http://54.151.76.197:8000/api/v1/health

# Chat with Llama 3.3
curl -X POST http://54.151.76.197:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [
      {"role": "user", "content": "Explain quantum computing"}
    ],
    "model_name": "llama33",
    "max_new_tokens": 512,
    "temperature": 0.7
  }'
```

## 🔧 Environment Variables

- `MODEL_PATH`: Path to model files (auto-configured from EC2 config)
- `HOST`: Server host (default: 0.0.0.0, auto-configured from EC2 config)
- `PORT`: Port for the API server (default: 8000, auto-configured from EC2 config)
- `LOG_LEVEL`: Logging level (auto-configured from EC2 config: INFO)
- `AWS_REGION`: AWS region (auto-configured from EC2 config: us-west-1)
- `HF_TOKEN`: HuggingFace token for model downloads

## 🦙 Llama 3.3 70B Features

### Model Specifications
- **Parameters**: 70 billion
- **Context Length**: 131,072 tokens (128K)
- **Languages**: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai
- **Architecture**: Transformer with Grouped Query Attention (GQA)
- **Optimization**: Flash Attention 2, BFloat16 precision

### Hardware Optimization (c6a.24xlarge)
- **Memory**: 192GB RAM utilization
- **CPU**: 96 vCPU cores
- **Multi-GPU**: Automatic tensor parallelism
- **Batch Processing**: Optimized for large context windows

## 💰 Cost Management

### Instance Costs (c6a.24xlarge)
- **Hourly**: ~$4.61
- **Daily**: ~$110.64
- **Monthly**: ~$3,317.76

### Cost Control Tools
```bash
# Monitor costs and stop/start instance
./scripts/cost-control.sh

# Quick status check
./scripts/quick-cost-check.sh
```

## 🏗️ Infrastructure as Code

### Terraform Management
```bash
# Setup Terraform
./scripts/terraform-setup.sh

# Navigate to terraform directory
cd terraform

# Plan changes
terraform plan

# Apply changes
terraform apply

# View outputs
terraform output
```

## 🔐 Security Notes

- The `griffin-connect.pem` file contains your private key
- Never commit PEM files to version control
- The install script sets proper permissions (400) automatically
- EC2 security group configured for SSH (22) and HTTP (8000) access
- CORS configured for specific origins in production

## 🚨 Troubleshooting

### Connection Issues
```bash
# Check EC2 instance status
aws ec2 describe-instances --instance-ids i-00268dae9fd36421f

# Test connection manually
ssh -i griffin-connect.pem ubuntu@54.151.76.197

# Use connection script
./scripts/connect-ec2.sh
```

### Model Issues
```bash
# Fix HuggingFace token permissions
./scripts/fix_hf_token.sh

# Fix dependency issues
./scripts/fix_dependencies.sh

# Reinstall Llama 3.3
./scripts/install_llama33.sh
```

### Cost Control
```bash
# Emergency stop instance
./scripts/cost-control.sh
# Choose option 6 for emergency stop

# Quick cost check
./scripts/quick-cost-check.sh
```

### Deployment Issues
```bash
# Force full redeployment
./scripts/deploy-to-ec2.sh

# Check sync manually
rsync -avz --dry-run -e "ssh -i griffin-connect.pem" ./ ubuntu@54.151.76.197:~/EPYC-testing/
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes and test locally
4. Use `./scripts/sync-to-ec2.sh` to test on EC2
5. Add tests and documentation
6. Submit a pull request

## 📄 License

MIT License 
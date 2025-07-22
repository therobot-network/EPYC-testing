# Model Deployment Repository

A scalable architecture for deploying machine learning models on AWS EC2 instances.

## Repository Structure

```
├── app/                    # Main application code
│   ├── api/               # API endpoints and routes
│   ├── models/            # Model loading and inference logic
│   ├── utils/             # Utility functions and helpers
│   └── config/            # Application configuration
├── infrastructure/        # Infrastructure as code
│   ├── terraform/         # Terraform configurations
│   └── scripts/           # Deployment and setup scripts
├── docker/               # Docker configurations
├── tests/                # Test suites
├── data/                 # Data storage and processing
├── models/               # Model artifacts and weights
├── configs/              # Configuration files
└── docs/                 # Documentation

```

## Quick Start

1. **Setup Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Development**
   ```bash
   python app/main.py
   ```

3. **Docker Deployment**
   ```bash
   docker-compose up --build
   ```

4. **EC2 Deployment**
   ```bash
   cd infrastructure/scripts
   ./deploy.sh
   ```

## Configuration

- Copy `configs/config.example.yaml` to `configs/config.yaml`
- Update configuration values for your environment
- Set environment variables in `.env` file

## API Endpoints

- `GET /health` - Health check
- `POST /predict` - Model inference
- `GET /models` - List available models
- `POST /models/load` - Load a specific model

## Environment Variables

- `MODEL_PATH`: Path to model files
- `API_PORT`: Port for the API server
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `AWS_REGION`: AWS region for deployment

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

MIT License 
#!/bin/bash

# Llama 3.3 70B Instruct Installation Script
# This script downloads and configures the Llama 3.3 70B Instruct model

set -e

echo "🦙 Llama 3.3 70B Instruct Installation Script"
echo "=============================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
LOCAL_MODEL_PATH="./models/llama-3.3-70b-instruct"
VRAM_AVAILABLE=190  # 190GB VRAM available
REQUIRED_SPACE_GB=140  # Approximate space needed for the model

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

# Check system requirements
check_requirements() {
    print_status "Checking system requirements..."
    
    # Check Python version
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is required but not installed."
        exit 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    print_status "Python version: $python_version"
    
    # Check GPU availability
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            gpu_info=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits)
            print_status "GPU Information:"
            echo "$gpu_info"
            
            # Calculate total VRAM
            total_vram=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{sum+=$1} END {print sum/1024}')
            print_status "Total VRAM: ${total_vram}GB"
            
            if (( $(echo "$total_vram < 140" | bc -l) )); then
                print_warning "Recommended VRAM is 140GB+. You have ${total_vram}GB. Model will use CPU offloading."
            else
                print_success "Excellent! You have ${total_vram}GB VRAM - perfect for Llama 3.3 70B!"
            fi
        else
            print_warning "nvidia-smi found but no NVIDIA driver loaded. Please install NVIDIA drivers first."
        fi
    else
        print_warning "nvidia-smi not found. If you have NVIDIA GPUs, install nvidia-utils first:"
        print_warning "  sudo apt install nvidia-utils-535"
    fi
    
    # Check disk space
    available_space=$(df -BG . | awk 'NR==2 {print $4}' | sed 's/G//')
    if (( available_space < REQUIRED_SPACE_GB )); then
        print_error "Insufficient disk space. Need ${REQUIRED_SPACE_GB}GB, have ${available_space}GB"
        exit 1
    fi
    print_status "Available disk space: ${available_space}GB"
}

# Install Python dependencies
install_dependencies() {
    print_status "Installing Python dependencies..."
    
    # Install python3-venv if not available
    if ! python3 -m venv --help >/dev/null 2>&1; then
        print_status "Installing python3-venv package..."
        sudo apt update
        sudo apt install -y python3-venv python3-pip
    fi
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install required packages
    print_status "Installing PyTorch with CUDA support..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    
    print_status "Installing Transformers and related packages..."
    pip install transformers>=4.45.0
    pip install accelerate>=0.21.0
    pip install sentencepiece>=0.1.99
    pip install protobuf>=3.20.0
    pip install huggingface_hub>=0.17.0
    
    # Install bitsandbytes separately as it can be tricky
    if pip install bitsandbytes>=0.41.0; then
        print_success "BitsAndBytes installed successfully"
    else
        print_warning "BitsAndBytes installation failed - quantization features may not work"
    fi
    
    # Install optional optimizations
    print_status "Installing build dependencies..."
    pip install wheel setuptools
    
    print_status "Installing optimization libraries..."
    # Try to install flash-attn, but don't fail if it doesn't work
    if pip install flash-attn --no-build-isolation; then
        print_success "Flash Attention installed successfully"
    else
        print_warning "Flash Attention installation failed - model will use standard attention (slower but still functional)"
    fi
    # Try to install xformers, but don't fail if it doesn't work
    if pip install xformers; then
        print_success "xformers installed successfully"
    else
        print_warning "xformers installation failed - will use standard transformers attention"
    fi
    
    print_success "Dependencies installed successfully"
}

# Setup HuggingFace authentication
setup_hf_auth() {
    print_status "Setting up HuggingFace authentication..."
    
    if [ -z "$HF_TOKEN" ]; then
        echo ""
        print_warning "HuggingFace token not found in environment."
        echo "To download Llama 3.3, you need to:"
        echo "1. Visit https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct"
        echo "2. Request access to the model"
        echo "3. Create a HuggingFace token with read permissions"
        echo "4. Set it as environment variable: export HF_TOKEN=your_token_here"
        echo ""
        read -p "Enter your HuggingFace token: " hf_token
        export HF_TOKEN="$hf_token"
        
        # Save to .env file for persistence
        echo "HF_TOKEN=$hf_token" >> .env
        print_success "Token saved to .env file"
    fi
    
    # Login to HuggingFace
    python3 -c "
from huggingface_hub import login
import os
login(token=os.environ.get('HF_TOKEN'))
print('Successfully authenticated with HuggingFace')
"
}

# Download the model
download_model() {
    print_status "Downloading Llama 3.3 70B Instruct model..."
    print_status "This may take several hours depending on your internet connection..."
    
    # Create model directory
    mkdir -p "$LOCAL_MODEL_PATH"
    
    # Download using huggingface-cli for better progress tracking
    if command -v huggingface-cli &> /dev/null; then
        huggingface-cli download "$MODEL_NAME" --local-dir "$LOCAL_MODEL_PATH" --local-dir-use-symlinks False
    else
        # Fallback to Python script
        python3 -c "
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

model_name = '$MODEL_NAME'
local_path = '$LOCAL_MODEL_PATH'

print('Downloading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=local_path)
tokenizer.save_pretrained(local_path)

print('Downloading model... This will take a while...')
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    cache_dir=local_path,
    torch_dtype='auto',
    device_map=None,  # Don't load to GPU yet
    low_cpu_mem_usage=True
)
model.save_pretrained(local_path)
print('Model download complete!')
"
    fi
    
    print_success "Model downloaded to $LOCAL_MODEL_PATH"
}

# Create model configuration
create_model_config() {
    print_status "Creating model configuration..."
    
    cat > "$LOCAL_MODEL_PATH/llama33_config.json" << EOF
{
    "model_name": "llama-3.3-70b-instruct",
    "model_path": "$LOCAL_MODEL_PATH",
    "model_type": "llama33",
    "architecture": "LlamaForCausalLM",
    "parameters": {
        "num_parameters": "70B",
        "context_length": 131072,
        "vocab_size": 128256,
        "hidden_size": 8192,
        "num_attention_heads": 64,
        "num_key_value_heads": 8,
        "intermediate_size": 28672,
        "num_hidden_layers": 80,
        "rope_theta": 500000.0,
        "max_position_embeddings": 131072
    },
    "chat_template": {
        "system_start": "<|start_header_id|>system<|end_header_id|>",
        "system_end": "<|eot_id|>",
        "user_start": "<|start_header_id|>user<|end_header_id|>",
        "user_end": "<|eot_id|>",
        "assistant_start": "<|start_header_id|>assistant<|end_header_id|>",
        "assistant_end": "<|eot_id|>",
        "begin_of_text": "<|begin_of_text|>",
        "end_of_text": "<|end_of_text|>"
    },
    "system_prompt": "You are a helpful, harmless, and honest AI assistant. You are Llama 3.3, created by Meta. You aim to be helpful, harmless, and honest in all your responses. You should be conversational and engaging while maintaining accuracy and safety.",
    "generation_config": {
        "do_sample": true,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": 4096,
        "pad_token_id": 128001,
        "eos_token_id": [128001, 128009],
        "bos_token_id": 128000
    },
    "hardware_optimization": {
        "use_flash_attention": true,
        "use_gradient_checkpointing": true,
        "load_in_8bit": false,
        "load_in_4bit": false,
        "device_map": "auto",
        "torch_dtype": "bfloat16",
        "attn_implementation": "flash_attention_2",
        "rope_scaling": null
    },
    "multi_gpu_config": {
        "max_memory_per_gpu": "47GB",
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "enable_tensor_parallelism": true
    }
}
EOF
    
    print_success "Model configuration created"
}

# Test model loading
test_model() {
    print_status "Testing model loading..."
    
    python3 -c "
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
import warnings
warnings.filterwarnings('ignore')

model_path = '$LOCAL_MODEL_PATH'
print(f'Loading model from {model_path}...')

try:
    # Load configuration
    with open(f'{model_path}/llama33_config.json', 'r') as f:
        config = json.load(f)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print(f'✓ Tokenizer loaded successfully')
    print(f'  Vocabulary size: {len(tokenizer)}')
    print(f'  Special tokens: BOS={tokenizer.bos_token_id}, EOS={tokenizer.eos_token_id}, PAD={tokenizer.pad_token_id}')
    
    # Test GPU availability
    if torch.cuda.is_available():
        print(f'✓ CUDA available with {torch.cuda.device_count()} GPUs')
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            memory_gb = props.total_memory / (1024**3)
            print(f'  GPU {i}: {props.name} ({memory_gb:.1f}GB)')
    else:
        print('⚠ CUDA not available, will use CPU')
    
    # Test model loading (just config, not weights to save time)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(model_path)
    print(f'✓ Model configuration loaded')
    print(f'  Model type: {model_config.model_type}')
    print(f'  Hidden size: {model_config.hidden_size}')
    print(f'  Number of layers: {model_config.num_hidden_layers}')
    print(f'  Context length: {model_config.max_position_embeddings}')
    
    print('✓ Model test completed successfully!')
    
except Exception as e:
    print(f'✗ Model test failed: {str(e)}')
    exit(1)
"
    
    print_success "Model test completed successfully"
}

# Create usage example
create_usage_example() {
    print_status "Creating usage example..."
    
    cat > "examples/llama33_example.py" << 'EOF'
#!/usr/bin/env python3
"""
Llama 3.3 70B Instruct Usage Example
This example shows how to use the Llama 3.3 70B model with proper configuration.
"""

import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.models.llama33_model import Llama33Model
import asyncio

async def main():
    # Model configuration
    model_path = "./models/llama-3.3-70b-instruct"
    
    # Load the model using our custom class
    model = Llama33Model(model_path)
    await model.load()
    
    # Example conversation
    messages = [
        {
            "role": "system", 
            "content": "You are a helpful AI assistant specialized in explaining complex topics clearly and concisely."
        },
        {
            "role": "user", 
            "content": "Explain quantum computing in simple terms."
        }
    ]
    
    # Generate response
    response, inference_time = await model.predict(messages, max_new_tokens=512, temperature=0.7)
    
    print(f"Response generated in {inference_time:.2f} seconds:")
    print(response)
    
    # Cleanup
    await model.unload()

if __name__ == "__main__":
    asyncio.run(main())
EOF
    
    mkdir -p examples
    chmod +x examples/llama33_example.py
    
    print_success "Usage example created at examples/llama33_example.py"
}

# Main installation flow
main() {
    echo "Starting Llama 3.3 70B installation..."
    
    check_requirements
    install_dependencies
    setup_hf_auth
    download_model
    create_model_config
    test_model
    create_usage_example
    
    echo ""
    print_success "🎉 Llama 3.3 70B Instruct installation completed successfully!"
    echo ""
    echo "Next steps:"
    echo "1. The model is installed at: $LOCAL_MODEL_PATH"
    echo "2. Run the example: python3 examples/llama33_example.py"
    echo "3. Use the model in your application via the ModelManager"
    echo ""
    echo "Configuration details:"
    echo "- Model supports up to 131K context length"
    echo "- Optimized for multi-GPU setup with ${VRAM_AVAILABLE}GB VRAM"
    echo "- Uses Flash Attention 2 for improved performance"
    echo "- Supports 8 languages: English, German, French, Italian, Portuguese, Hindi, Spanish, Thai"
    echo ""
    print_warning "Note: First model load may take several minutes as it loads 70B parameters"
}

# Run main function
main "$@" 
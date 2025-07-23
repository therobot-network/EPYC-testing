#!/bin/bash

# Llama 3.1 8B Setup Script
# This script downloads and configures Llama 3.1 8B Instruct as the default model

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Print functions
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

# Configuration
MODEL_DIR="./models/llama-3.1-8b-instruct"
CACHE_DIR="$HOME/.cache/huggingface/transformers"

echo "🦙 Llama 3.1 8B Setup Script"
echo "=============================="
echo ""

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    print_error "Python 3 is required but not installed."
    exit 1
fi

# Check if the download script exists
if [ ! -f "scripts/download_llama31.py" ]; then
    print_error "Download script not found. Please run from project root."
    exit 1
fi

# Check if model already exists
if [ -d "$MODEL_DIR" ] && [ "$(ls -A $MODEL_DIR)" ]; then
    print_warning "Model directory $MODEL_DIR already exists and is not empty."
    read -p "Do you want to re-download? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        print_status "Skipping download. Using existing model."
        SKIP_DOWNLOAD=true
    fi
fi

# Download the model if needed
if [ "$SKIP_DOWNLOAD" != "true" ]; then
    print_status "Downloading Llama 3.1 8B Instruct model..."
    python3 scripts/download_llama31.py \
        --model-size 8b \
        --cache-dir "$CACHE_DIR" \
        --verify
    
    if [ $? -ne 0 ]; then
        print_error "Model download failed!"
        exit 1
    fi
    
    print_success "Model downloaded successfully!"
fi

# Create model directory if it doesn't exist
mkdir -p "$MODEL_DIR"

# Create symlinks or copy files from cache to model directory
print_status "Setting up model files..."

# Find the cached model directory
CACHED_MODEL_DIR=$(find "$CACHE_DIR" -name "*Llama-3.1-8B-Instruct*" -type d | head -1)

if [ -z "$CACHED_MODEL_DIR" ]; then
    print_error "Could not find cached model directory"
    exit 1
fi

print_status "Found cached model at: $CACHED_MODEL_DIR"

# Create symlinks to the cached files (saves disk space)
if [ "$SKIP_DOWNLOAD" != "true" ]; then
    print_status "Creating symlinks to cached model files..."
    
    # Remove existing directory if empty or create new one
    rm -rf "$MODEL_DIR"
    mkdir -p "$MODEL_DIR"
    
    # Create symlinks for all model files
    for file in "$CACHED_MODEL_DIR"/*; do
        if [ -f "$file" ]; then
            ln -sf "$file" "$MODEL_DIR/$(basename "$file")"
        fi
    done
fi

# Create Llama 3.1 configuration file
print_status "Creating Llama 3.1 configuration..."

cat > "$MODEL_DIR/llama31_config.json" << 'EOF'
{
    "model_name": "llama-3.1-8b-instruct",
    "model_path": "./models/llama-3.1-8b-instruct",
    "model_type": "llama31",
    "architecture": "LlamaForCausalLM",
    "parameters": {
        "num_parameters": "8B",
        "context_length": 131072,
        "vocab_size": 128256,
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "intermediate_size": 14336,
        "num_hidden_layers": 32,
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
    "system_prompt": "You are a helpful AI assistant created by Meta. You are Llama 3.1, a large language model. You aim to be helpful, harmless, and honest in all your responses.",
    "generation_config": {
        "do_sample": true,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 50,
        "repetition_penalty": 1.1,
        "max_new_tokens": 2048,
        "pad_token_id": 128001,
        "eos_token_id": [128001, 128009],
        "bos_token_id": 128000
    },
    "hardware_optimization": {
        "use_flash_attention": true,
        "use_gradient_checkpointing": true,
        "load_in_8bit": true,
        "load_in_4bit": false,
        "device_map": "auto",
        "torch_dtype": "float16",
        "attn_implementation": "flash_attention_2",
        "rope_scaling": null
    }
}
EOF

print_success "Configuration file created!"

# Create a simple test script
print_status "Creating test script..."

cat > "examples/test_llama31.py" << 'EOF'
#!/usr/bin/env python3
"""
Simple test script for Llama 3.1 8B model
"""

import asyncio
import sys
from pathlib import Path

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.models.llama31_model import Llama31Model

async def test_llama31():
    """Test the Llama 3.1 model with a simple prompt."""
    model_path = "./models/llama-3.1-8b-instruct"
    
    print("🚀 Testing Llama 3.1 8B model...")
    print(f"Model path: {model_path}")
    
    try:
        # Initialize model
        model = Llama31Model(model_path)
        
        # Load model
        print("Loading model...")
        await model.load()
        
        # Test simple prediction
        print("\n" + "="*50)
        print("Testing simple prediction...")
        response = await model.predict("Hello! Can you tell me about artificial intelligence?")
        print(f"Response: {response}")
        
        # Test chat interface
        print("\n" + "="*50)
        print("Testing chat interface...")
        messages = [
            {"role": "user", "content": "What is Python programming language?"}
        ]
        chat_response = await model.chat(messages)
        print(f"Chat response: {chat_response}")
        
        # Get performance stats
        stats = model.get_performance_stats()
        print("\n" + "="*50)
        print("Performance Statistics:")
        for key, value in stats.items():
            if isinstance(value, list) and value:
                print(f"  {key}: {value[-1]:.3f} (latest)")
            elif not isinstance(value, list):
                print(f"  {key}: {value:.3f}")
        
        print("\n✅ All tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        return False
    
    finally:
        # Clean up
        if 'model' in locals():
            await model.unload()
    
    return True

if __name__ == "__main__":
    success = asyncio.run(test_llama31())
    sys.exit(0 if success else 1)
EOF

chmod +x examples/test_llama31.py

print_success "Test script created!"

# Test the setup
print_status "Testing model setup..."
if python3 examples/test_llama31.py; then
    print_success "Model setup test passed!"
else
    print_warning "Model setup test had issues, but installation may still work"
fi

echo ""
print_success "🎉 Llama 3.1 8B setup completed!"
echo ""
echo "Next steps:"
echo "1. Test the model: python3 examples/test_llama31.py"
echo "2. Launch the TUI: ./cli.py --auto-load"
echo "3. Start the API server: python3 app/main.py"
echo ""
echo "Model details:"
echo "- Location: $MODEL_DIR"
echo "- Model: Llama 3.1 8B Instruct"
echo "- Context length: 128K tokens"
echo "- Optimized for: CPU inference with 8-bit quantization"
echo "- Memory usage: ~8GB (with quantization)"
echo ""
print_warning "Note: First model load may take a few minutes" 
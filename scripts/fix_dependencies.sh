#!/bin/bash

# Quick fix for dependency issues
echo "🔧 Fixing dependency issues..."

# Make sure we're in the virtual environment
if [ -z "$VIRTUAL_ENV" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Install build dependencies first
echo "Installing build dependencies..."
pip install wheel setuptools

# Install the core packages that are working
echo "Installing working packages..."
pip install transformers>=4.45.0
pip install accelerate>=0.21.0
pip install sentencepiece>=0.1.99
pip install protobuf>=3.20.0
pip install huggingface_hub>=0.17.0

# Try bitsandbytes (optional for quantization)
echo "Attempting to install bitsandbytes..."
if pip install bitsandbytes>=0.41.0; then
    echo "✅ BitsAndBytes installed successfully"
else
    echo "⚠️  BitsAndBytes installation failed - quantization features may not work (but model will still run)"
fi

# Skip flash-attn for now - it's optional and causing issues
echo "⚠️  Skipping flash-attn installation (optional optimization)"
echo "⚠️  Skipping xformers installation (optional optimization)"

# Install additional required packages for the model
echo "Installing additional model dependencies..."
pip install loguru
pip install pydantic
pip install fastapi
pip install uvicorn

echo "✅ Dependencies fixed! You can now continue with the model download."
echo ""
echo "To continue installation, run:"
echo "  source venv/bin/activate"
echo "  huggingface-cli login"
echo "  huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./models/llama-3.3-70b-instruct" 
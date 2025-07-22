# Llama 3.3 70B Instruct Setup Guide

This guide will help you install and configure the Llama 3.3 70B Instruct model for optimal performance on your system with 190GB VRAM.

## 🚀 Quick Start

### Prerequisites

- **Hardware**: 140GB+ VRAM recommended (you have 190GB ✅)
- **Storage**: 140GB+ free disk space
- **OS**: Linux/macOS with NVIDIA GPUs
- **Python**: 3.8+ with CUDA support

### 1. Run the Installation Script

```bash
# Make sure you're in the project root
cd /path/to/EPYC-testing

# Run the installation script
./scripts/install_llama33.sh
```

The script will:
- ✅ Check system requirements
- ✅ Install required Python dependencies
- ✅ Setup HuggingFace authentication
- ✅ Download the Llama 3.3 70B model (~140GB)
- ✅ Configure optimizations for your hardware
- ✅ Test the installation

### 2. HuggingFace Access

You'll need to:
1. Visit [https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
2. Request access to the model (approval usually takes a few hours)
3. Create a HuggingFace token with read permissions
4. Provide the token when prompted by the install script

## 📋 Model Specifications

### Llama 3.3 70B Instruct Details

| Specification | Value |
|---------------|-------|
| **Parameters** | 70 billion |
| **Context Length** | 131,072 tokens (128K) |
| **Vocabulary Size** | 128,256 tokens |
| **Architecture** | Transformer with GQA |
| **Precision** | BFloat16 optimized |
| **Languages** | English, German, French, Italian, Portuguese, Hindi, Spanish, Thai |

### Performance Optimizations

- **Flash Attention 2**: Reduces memory usage and increases speed
- **Multi-GPU Support**: Automatically distributes across available GPUs
- **Gradient Checkpointing**: Memory optimization for large models
- **BFloat16 Precision**: Optimal balance of speed and accuracy

## 🔧 Configuration

### Hardware Configuration

The system automatically configures for your 190GB VRAM:

```json
{
  "multi_gpu_config": {
    "max_memory_per_gpu": "47GB",
    "tensor_parallel_size": 4,
    "pipeline_parallel_size": 1,
    "enable_tensor_parallelism": true
  },
  "hardware_optimization": {
    "use_flash_attention": true,
    "torch_dtype": "bfloat16",
    "device_map": "auto",
    "attn_implementation": "flash_attention_2"
  }
}
```

### System Prompt

The default system prompt is:

```
You are a helpful, harmless, and honest AI assistant. You are Llama 3.3, created by Meta. You aim to be helpful, harmless, and honest in all your responses. You should be conversational and engaging while maintaining accuracy and safety.
```

### Chat Template Format

Llama 3.3 uses a specific chat template:

```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_message}

<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_message}

<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{assistant_response}

<|eot_id|>
```

## 💻 Usage Examples

### 1. Using the Model Manager

```python
from app.models.manager import ModelManager
import asyncio

async def main():
    # Initialize model manager
    manager = ModelManager()
    
    # Load Llama 3.3
    await manager.load_model(
        model_name="llama33",
        model_path="./models/llama-3.3-70b-instruct",
        model_type="llama33"
    )
    
    # Make a prediction
    response, time_taken = await manager.predict(
        model_name="llama33",
        input_data="Explain quantum computing in simple terms",
        parameters={"max_new_tokens": 512, "temperature": 0.7}
    )
    
    print(f"Response: {response}")
    print(f"Generated in {time_taken:.2f} seconds")

asyncio.run(main())
```

### 2. Chat Interface

```python
from app.models.llama33_model import Llama33Model
import asyncio

async def main():
    model = Llama33Model("./models/llama-3.3-70b-instruct")
    await model.load()
    
    messages = [
        {
            "role": "system",
            "content": "You are a helpful coding assistant."
        },
        {
            "role": "user", 
            "content": "Write a Python function to calculate fibonacci numbers."
        }
    ]
    
    response = await model.chat(messages, max_new_tokens=1024)
    print(response)
    
    await model.unload()

asyncio.run(main())
```

### 3. REST API Usage

```bash
# Load the model
curl -X POST "http://localhost:8000/api/v1/models/load" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama33",
    "model_path": "./models/llama-3.3-70b-instruct"
  }'

# Chat with the model
curl -X POST "http://localhost:8000/api/v1/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "llama33",
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful AI assistant."
      },
      {
        "role": "user",
        "content": "What are the benefits of renewable energy?"
      }
    ],
    "max_new_tokens": 1024,
    "temperature": 0.7
  }'
```

## ⚡ Performance Tuning

### Memory Optimization

For your 190GB VRAM setup:

```python
# In llama33_config.json
{
  "hardware_optimization": {
    "load_in_8bit": false,    # Keep false for full precision
    "load_in_4bit": false,    # Keep false for best quality
    "use_flash_attention": true,
    "device_map": "auto"
  }
}
```

### Generation Parameters

Recommended settings for different use cases:

**Creative Writing:**
```json
{
  "temperature": 0.8,
  "top_p": 0.9,
  "top_k": 40,
  "repetition_penalty": 1.1
}
```

**Code Generation:**
```json
{
  "temperature": 0.2,
  "top_p": 0.95,
  "top_k": 50,
  "repetition_penalty": 1.05
}
```

**Analytical Tasks:**
```json
{
  "temperature": 0.3,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.0
}
```

## 🔍 Monitoring and Debugging

### GPU Memory Usage

```python
import torch

def check_gpu_memory():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            print(f"GPU {i}: {allocated:.1f}GB allocated, {cached:.1f}GB cached, {total:.1f}GB total")

check_gpu_memory()
```

### Performance Metrics

The model logs performance metrics:
- **Tokens per second**: Generation speed
- **Memory usage**: VRAM utilization
- **Load time**: Model initialization time

### Common Issues and Solutions

**Issue: Out of Memory Error**
```bash
# Solution: Enable gradient checkpointing
"hardware_optimization": {
  "use_gradient_checkpointing": true
}
```

**Issue: Slow Generation**
```bash
# Solution: Check Flash Attention installation
pip install flash-attn --no-build-isolation
```

**Issue: Model Loading Timeout**
```bash
# Solution: Increase timeout in settings
"inference_timeout": 300  # 5 minutes
```

## 📊 Benchmarks and Expected Performance

### Expected Performance on Your System (190GB VRAM)

| Metric | Expected Value |
|--------|----------------|
| **Load Time** | 3-5 minutes |
| **Generation Speed** | 15-25 tokens/second |
| **Memory Usage** | 140-150GB VRAM |
| **Context Length** | Up to 128K tokens |

### Quality Benchmarks

Llama 3.3 70B achieves:
- **MMLU**: 86.0% accuracy
- **HumanEval**: 88.4% pass@1
- **MATH**: 77.0% accuracy
- **GPQA**: 50.5% accuracy

## 🛠 Troubleshooting

### Installation Issues

1. **HuggingFace Access Denied**
   - Ensure you've been granted access to the model
   - Check your HF token has read permissions
   - Try logging out and back in: `huggingface-cli logout && huggingface-cli login`

2. **CUDA Out of Memory**
   - Your system has 190GB VRAM, so this shouldn't occur
   - If it does, try enabling quantization temporarily

3. **Flash Attention Installation Failed**
   - Install build dependencies: `pip install ninja`
   - Try: `pip install flash-attn --no-build-isolation --force-reinstall`

### Runtime Issues

1. **Slow Generation**
   - Check GPU utilization with `nvidia-smi`
   - Ensure Flash Attention is working
   - Verify multi-GPU distribution

2. **Poor Response Quality**
   - Adjust temperature (lower = more focused)
   - Check system prompt formatting
   - Ensure proper chat template usage

## 🔄 Updates and Maintenance

### Updating the Model

```bash
# Re-run the install script to update
./scripts/install_llama33.sh

# Or manually update
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct --local-dir ./models/llama-3.3-70b-instruct
```

### Monitoring

- Check logs in `./logs/` directory
- Monitor GPU usage with `nvidia-smi`
- Use the `/api/v1/models` endpoint to check model status

## 📚 Additional Resources

- [Llama 3.3 Model Card](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct)
- [Transformers Documentation](https://huggingface.co/docs/transformers)
- [Flash Attention](https://github.com/Dao-AILab/flash-attention)
- [Multi-GPU Setup Guide](https://huggingface.co/docs/accelerate/usage_guides/big_modeling)

## 🆘 Support

If you encounter issues:

1. Check the logs in `./logs/`
2. Review this documentation
3. Check GPU memory with `nvidia-smi`
4. Verify model files are complete
5. Test with a smaller model first

---

**Note**: First-time model loading may take 5-10 minutes as it loads 70 billion parameters. Subsequent loads will be faster due to caching. 
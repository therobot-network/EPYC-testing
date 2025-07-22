# EPYC-testing TUI Guide

## 🦙 LLaMA 3.3 70B Terminal User Interface

A beautiful, feature-rich terminal user interface for interacting with the LLaMA 3.3 70B model on your EPYC-testing deployment.

## ✨ Features

### 🎨 Beautiful Interface
- **Rich terminal UI** with colors, panels, and animations
- **Loading screens** with progress bars and system information
- **Real-time chat interface** with message history
- **Responsive design** that adapts to terminal size

### ⚡ Real-time Progress Tracking
- **Token-by-token generation progress** with live updates
- **Performance metrics** (tokens/second, elapsed time)
- **Second-by-second timestamping** of all interactions
- **Generation statistics** and completion times

### 🔧 System Integration
- **Automatic model discovery** from configured paths
- **EC2 optimization detection** and hardware information
- **Memory and CPU monitoring** during model loading
- **Seamless integration** with existing model manager

### 🎮 Interactive Controls
- **Keyboard shortcuts** for quick actions
- **Message history** with scrollable chat
- **Model reloading** and chat clearing
- **Graceful shutdown** handling

## 🚀 Quick Start

### 1. Install Dependencies

```bash
# Install the TUI libraries (already added to requirements.txt)
pip install rich>=13.7.0 textual>=0.45.0 click>=8.1.0
```

### 2. Launch the TUI

```bash
# From the project root
./cli.py

# Or with Python
python cli.py

# With options
./cli.py --model-path ./models/llama-3.3-70b-instruct --auto-load --debug
```

### 3. Available Options

```bash
Usage: cli.py [OPTIONS]

Options:
  -m, --model-path PATH   Path to the LLaMA model directory
  -n, --model-name TEXT   Name to assign to the loaded model (default: llama33)
  -a, --auto-load         Automatically load the model on startup
  -d, --debug             Enable debug mode
  --help                  Show this message and exit
```

## 🎯 Usage Examples

### Basic Usage

```bash
# Launch with auto-detection
./cli.py

# Launch with specific model path
./cli.py -m /path/to/llama-3.3-70b-instruct

# Auto-load model on startup
./cli.py --auto-load

# Debug mode for troubleshooting
./cli.py --debug
```

### EC2 Usage

```bash
# On your EC2 instance
cd ~/EPYC-testing
./cli.py --auto-load

# The TUI will automatically detect EC2 configuration and show:
# - Instance type (c6a.24xlarge)
# - Available CPU cores (96)
# - Memory (192GB)
# - AWS region
```

## 🖥️ Interface Overview

### Loading Screen

```
┌─ 🦙 Welcome ────────────────────────────────┐
│              EPYC-TESTING TUI                │
│         LLaMA 3.3 70B Terminal Interface     │
└──────────────────────────────────────────────┘

┌─ Model Loading ──────────────────────────────┐
│ 🦙 Loading LLaMA 3.3 70B                    │
│ 📁 ./models/llama-3.3-70b-instruct          │
│ 🏷️  llama33                                 │
└──────────────────────────────────────────────┘

┌─ System Information ─────────────────────────┐
│ CPU Cores    │ 96        │ Physical cores   │
│ Memory       │ 192 GB    │ 15% used        │
│ Disk Space   │ 1000 GB   │ 45% used        │
│ Model Size   │ ~140 GB   │ Estimated usage │
│ Instance     │ c6a.24xl  │ AWS EC2 instance│
│ Region       │ us-west-1 │ AWS region      │
└──────────────────────────────────────────────┘

┌─ Loading Progress ───────────────────────────┐
│ Stage: Loading Model Weights                 │
│ Progress: 60.0%                             │
│ Elapsed: 45.2s                              │
│ Details: This may take several minutes...   │
│                                             │
│ [████████████████████░░░░░░░░░░░░░░░░░░░░] 60.0% │
└──────────────────────────────────────────────┘

[12:34:56] Initializing model manager...
[12:35:01] Loading tokenizer from model path...
[12:35:15] Reading model configuration...
[12:35:20] Loading 70B parameters... This will take time.
[12:36:45] Applying hardware optimizations...
[12:36:50] Model loaded successfully!
```

### Chat Interface

```
EPYC-testing TUI - Chat - llama33                    12:34:56

┌─────────────────────────────────────────────────────────┐
│ ℹ️ System (12:34:56)                                   │
│ Chat interface ready. Start typing to chat with       │
│ LLaMA 3.3!                                            │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 👤 You (12:35:12)                                      │
│ Explain quantum computing in simple terms              │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ 🦙 LLaMA 3.3 (12:35:18)                               │
│ Quantum computing is like having a very special       │
│ computer that works with the weird rules of quantum   │
│ physics instead of regular physics...                 │
└─────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────┐
│ Type your message here...                    [Send]    │
└─────────────────────────────────────────────────────────┘

[12:35:18] Generating... 45 tokens (12.3 tok/s) - 3.7s
```

## ⌨️ Keyboard Shortcuts

| Shortcut | Action | Description |
|----------|--------|-------------|
| `Ctrl+C` | Quit | Exit the application |
| `Ctrl+L` | Clear Chat | Clear conversation history |
| `Ctrl+R` | Reload Model | Reload the model (useful after updates) |
| `Enter` | Send Message | Send the current message |
| `↑/↓` | Scroll | Navigate through chat history |
| `Tab` | Focus | Switch between input and send button |

## 🔧 Configuration

### Automatic Model Detection

The TUI automatically searches for models in these locations:

1. `./models/llama-3.3-70b-instruct` (local)
2. `{settings.model_path}/llama-3.3-70b-instruct` (configured path)
3. `/home/ubuntu/EPYC-testing/models/llama-3.3-70b-instruct` (EC2 path)

### EC2 Integration

When running on EC2, the TUI automatically detects and displays:

- **Instance Information**: Type, ID, region, availability zone
- **Performance Metrics**: CPU cores, memory, optimization settings
- **Hardware Capabilities**: SIMD support, memory bandwidth
- **Cost Information**: Instance pricing and usage estimates

### Generation Parameters

Default generation settings (can be modified in the code):

```python
parameters = {
    "max_new_tokens": 1024,      # Maximum response length
    "temperature": 0.7,          # Creativity (0.1-2.0)
    "top_p": 0.9,               # Nucleus sampling
    "top_k": 50,                # Top-K sampling
    "repetition_penalty": 1.1    # Avoid repetition
}
```

## 📊 Progress Tracking Features

### Real-time Generation Metrics

- **Token Count**: Live count of generated tokens
- **Speed Tracking**: Tokens per second calculation
- **Time Elapsed**: Precise timing from start to completion
- **Progress Estimation**: Visual progress bar based on expected length
- **Stage Information**: Current generation phase

### Timestamping

All interactions are timestamped with precision:

```
[12:35:18] User message received
[12:35:18] Generation started
[12:35:19] 10 tokens generated (15.2 tok/s)
[12:35:20] 25 tokens generated (14.8 tok/s)
[12:35:22] Generation completed - 156 tokens (13.9 tok/s)
```

### Performance Statistics

After each generation, detailed statistics are available:

- **Total tokens generated**
- **Average generation speed**
- **Response length (characters/words)**
- **Memory usage during generation**
- **Hardware utilization**

## 🐛 Troubleshooting

### Common Issues

#### Model Not Found
```
❌ No model found!
Please specify a model path with --model-path or ensure the model is in one of these locations:
• ./models/llama-3.3-70b-instruct
• /configured/path/llama-3.3-70b-instruct
• /home/ubuntu/EPYC-testing/models/llama-3.3-70b-instruct
```

**Solution**: Download the model using `./scripts/install_llama33.sh` or specify the correct path.

#### Memory Issues
```
Error loading model: CUDA out of memory
```

**Solutions**:
- Ensure you have enough system memory (192GB recommended)
- Check that no other processes are using excessive memory
- Try restarting the EC2 instance if memory is fragmented

#### Performance Issues
```
Generation very slow (< 1 tok/s)
```

**Solutions**:
- Check CPU utilization with `htop`
- Ensure EC2 instance is not throttled
- Verify model is using optimized settings (FP16, proper threading)

### Debug Mode

Enable debug mode for detailed error information:

```bash
./cli.py --debug
```

This provides:
- **Detailed error traces**
- **Model loading diagnostics**
- **Performance profiling**
- **Memory usage monitoring**

### Log Files

The TUI creates log files in `./logs/`:
- `tui.log` - General application logs
- `model.log` - Model loading and inference logs
- `performance.log` - Performance metrics and timing

## 🔄 Updates and Maintenance

### Updating the TUI

```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt

# Restart the TUI
./cli.py --auto-load
```

### Model Updates

To update the model:

```bash
# Reload model in running TUI
Ctrl+R

# Or restart with new model path
./cli.py -m /path/to/new/model --auto-load
```

## 🚀 Advanced Features

### Custom Model Configurations

You can customize generation parameters by modifying the chat interface:

```python
# In app/cli/tui.py, modify the send_message method
parameters = {
    "max_new_tokens": 2048,     # Longer responses
    "temperature": 0.3,         # More focused
    "top_p": 0.95,             # Slightly more diverse
    "repetition_penalty": 1.05  # Less repetition penalty
}
```

### Multiple Models

The TUI can be extended to support multiple models:

```bash
# Load different models
./cli.py -m ./models/model1 -n model1
./cli.py -m ./models/model2 -n model2
```

### Conversation Export

Chat history can be exported (future feature):

```python
# Export conversation to file
chat_interface.export_conversation("conversation_2024.json")
```

## 📈 Performance Optimization

### For c6a.24xlarge (96 vCPUs, 192GB RAM)

Optimal settings are automatically applied:

```yaml
performance:
  torch_threads: 96
  cpu_cores: 96
  batch_size: 32
  torch_dtype: "float16"
  low_cpu_mem_usage: true
  use_half_precision: true
```

### Memory Management

- **Model caching**: Keep model in memory between sessions
- **Gradient checkpointing**: Reduce memory usage during inference
- **Batch optimization**: Process multiple requests efficiently

### CPU Optimization

- **Thread allocation**: Use all 96 CPU cores effectively
- **SIMD instructions**: Leverage AVX2 for faster computation
- **Memory bandwidth**: Optimize for AMD EPYC architecture

## 🤝 Contributing

To contribute to the TUI:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/tui-enhancement`
3. **Make changes** to the TUI components in `app/cli/`
4. **Test thoroughly** on both local and EC2 environments
5. **Submit a pull request** with detailed description

### Code Structure

```
app/cli/
├── __init__.py          # Package initialization
├── main.py              # CLI entry point and argument parsing
├── tui.py               # Main TUI application and components
└── progress_tracker.py  # Real-time progress tracking
```

---

**Happy chatting with LLaMA 3.3! 🦙✨** 
# Simple Chat CLI Usage

No more curl commands! Use the simple CLI to chat with your models easily.

## Quick Start

### 1. Start the server (in one terminal)
```bash
python -m app.main
```

### 2. Chat with models (in another terminal)
```bash
# Ultra-simple - just specify the model
python chat.py llama31          # Chat with Llama 3.1
python chat.py llama2           # Chat with Llama 2 13B
python chat.py llama33          # Chat with Llama 3.3 70B

# Or start interactive mode
python chat.py                  # Choose model interactively
```

## Available Models

| Shortcut | Model | Description |
|----------|-------|-------------|
| `llama31` | Llama 3.1 8B | Fast and efficient, good for most tasks |
| `llama2` | Llama 2 13B | Reliable and well-tested, great balance |
| `llama33` | Llama 3.3 70B | Most capable, requires lots of VRAM |

## Interactive Commands

Once in the CLI, you can use these commands:

```bash
# Model management
load llama31        # Load Llama 3.1
load llama2         # Load Llama 2
unload llama31      # Unload a model

# Information
models              # Show available models
loaded              # Show currently loaded models
help                # Show help

# Chat management
clear               # Clear conversation history
quit                # Exit (or Ctrl+C)

# Generation parameters
set temperature 0.8    # Adjust creativity (0.1-1.0)
set max_tokens 512     # Limit response length
set top_p 0.9         # Nucleus sampling
set top_k 50          # Top-k sampling
```

## Examples

### Basic Chat
```bash
$ python chat.py llama31
✓ Connected to server at http://localhost:8000
✓ Llama 3.1 8B Instruct - Fast and efficient loaded successfully!

[llama31] > Hello! How are you? 
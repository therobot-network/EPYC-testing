#!/usr/bin/env python3
"""
Ultra-simple chat wrapper - the easiest way to chat with your models!

Usage:
  python chat.py llama31          # Load and chat with Llama 3.1
  python chat.py llama2           # Load and chat with Llama 2
  python chat.py                  # Interactive mode
"""

import sys
import subprocess
import os
from pathlib import Path

def main():
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    chat_cli_path = script_dir / "chat_cli.py"
    
    # Check if chat_cli.py exists
    if not chat_cli_path.exists():
        print("❌ chat_cli.py not found!")
        return 1
    
    # Build command
    cmd = [sys.executable, str(chat_cli_path)]
    
    # If a model is specified as argument, add it
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
        cmd.extend(["--model", model_name])
    
    # Add any additional arguments
    if len(sys.argv) > 2:
        cmd.extend(sys.argv[2:])
    
    # Run the CLI
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nGoodbye!")
    except Exception as e:
        print(f"Error running chat CLI: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 
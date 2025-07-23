#!/usr/bin/env python3
"""
Llama 3.1 Model Download Script

This script downloads the Llama 3.1 8B Instruct model from Hugging Face.
It handles authentication, model caching, and provides progress tracking.

Requirements:
- Hugging Face account with accepted Llama 3.1 license
- HF_TOKEN environment variable or manual token input
- Sufficient disk space (~16GB for 8B model)

Usage:
    python scripts/download_llama31.py [--model-size 8b] [--cache-dir /path/to/cache]
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import login, snapshot_download, HfApi
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
except ImportError as e:
    print(f"Error: Missing required packages. Please install with:")
    print("pip install transformers huggingface_hub torch")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configurations
LLAMA_MODELS = {
    "8b": "meta-llama/Llama-3.1-8B-Instruct",
    "70b": "meta-llama/Llama-3.1-70B-Instruct", 
    "405b": "meta-llama/Llama-3.1-405B-Instruct"
}

def get_hf_token() -> Optional[str]:
    """Get Hugging Face token from environment or user input."""
    token = os.getenv('HF_TOKEN')
    if token:
        logger.info("Found HF_TOKEN in environment variables")
        return token
    
    logger.warning("HF_TOKEN not found in environment variables")
    token = input("Please enter your Hugging Face token: ").strip()
    if not token:
        logger.error("No token provided. Cannot proceed without authentication.")
        return None
    return token

def check_license_acceptance(model_id: str, token: str) -> bool:
    """Check if user has accepted the Llama 3.1 license."""
    try:
        api = HfApi()
        # Try to access model info - this will fail if license not accepted
        model_info = api.model_info(model_id, token=token)
        logger.info(f"License check passed for {model_id}")
        return True
    except Exception as e:
        logger.error(f"License check failed: {e}")
        logger.error("Please visit https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct")
        logger.error("and accept the Llama 3.1 Community License Agreement")
        return False

def check_disk_space(cache_dir: Path, model_size: str) -> bool:
    """Check if sufficient disk space is available."""
    # Approximate model sizes in GB
    size_requirements = {
        "8b": 16,   # ~16GB for 8B model
        "70b": 140, # ~140GB for 70B model  
        "405b": 810 # ~810GB for 405B model
    }
    
    required_gb = size_requirements.get(model_size, 16)
    
    try:
        stat = os.statvfs(cache_dir)
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
        
        logger.info(f"Available disk space: {free_gb:.1f}GB")
        logger.info(f"Required disk space: {required_gb}GB")
        
        if free_gb < required_gb:
            logger.error(f"Insufficient disk space. Need {required_gb}GB, have {free_gb:.1f}GB")
            return False
        return True
    except Exception as e:
        logger.warning(f"Could not check disk space: {e}")
        return True  # Proceed anyway

def download_model(model_id: str, cache_dir: Path, token: str) -> bool:
    """Download the specified Llama model."""
    try:
        logger.info(f"Starting download of {model_id}")
        logger.info(f"Cache directory: {cache_dir}")
        
        # Download model files
        logger.info("Downloading model files...")
        snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            token=token,
            resume_download=True,
            local_files_only=False
        )
        
        logger.info("Model download completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False

def verify_model(model_id: str, cache_dir: Path) -> bool:
    """Verify the downloaded model can be loaded."""
    try:
        logger.info("Verifying model integrity...")
        
        # Try to load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_id, 
            cache_dir=cache_dir,
            local_files_only=True
        )
        logger.info("✓ Tokenizer loaded successfully")
        
        # Try to load model config (lightweight check)
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(
            model_id,
            cache_dir=cache_dir, 
            local_files_only=True
        )
        logger.info("✓ Model configuration loaded successfully")
        
        logger.info("Model verification completed!")
        return True
        
    except Exception as e:
        logger.error(f"Model verification failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download Llama 3.1 models from Hugging Face")
    parser.add_argument(
        "--model-size", 
        choices=["8b", "70b", "405b"], 
        default="8b",
        help="Model size to download (default: 8b)"
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path.home() / ".cache" / "huggingface" / "transformers",
        help="Directory to cache downloaded models"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify model integrity after download"
    )
    parser.add_argument(
        "--skip-space-check",
        action="store_true", 
        help="Skip disk space verification"
    )
    
    args = parser.parse_args()
    
    # Get model ID
    model_id = LLAMA_MODELS[args.model_size]
    logger.info(f"Selected model: {model_id}")
    
    # Create cache directory
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    
    # Check disk space
    if not args.skip_space_check:
        if not check_disk_space(args.cache_dir, args.model_size):
            sys.exit(1)
    
    # Get authentication token
    token = get_hf_token()
    if not token:
        sys.exit(1)
    
    # Login to Hugging Face
    try:
        login(token=token)
        logger.info("Successfully authenticated with Hugging Face")
    except Exception as e:
        logger.error(f"Authentication failed: {e}")
        sys.exit(1)
    
    # Check license acceptance
    if not check_license_acceptance(model_id, token):
        sys.exit(1)
    
    # Download model
    if not download_model(model_id, args.cache_dir, token):
        sys.exit(1)
    
    # Verify model if requested
    if args.verify:
        if not verify_model(model_id, args.cache_dir):
            logger.warning("Model verification failed, but download may still be usable")
    
    logger.info("="*50)
    logger.info("DOWNLOAD COMPLETED SUCCESSFULLY!")
    logger.info(f"Model: {model_id}")
    logger.info(f"Location: {args.cache_dir}")
    logger.info("="*50)
    
    # Print usage instructions
    print("\nTo use this model in your code:")
    print(f"""
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("{model_id}")
model = AutoModelForCausalLM.from_pretrained(
    "{model_id}",
    torch_dtype=torch.float16,
    device_map="auto"
)
""")

if __name__ == "__main__":
    main() 
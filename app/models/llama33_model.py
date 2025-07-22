"""
Llama 3.3 70B Instruct model implementation with optimized configuration.
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from loguru import logger
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    AutoConfig,
    BitsAndBytesConfig,
    GenerationConfig
)

from app.models.base import BaseModel


class Llama33Model(BaseModel):
    """Llama 3.3 70B Instruct model implementation with optimized configuration."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tokenizer = None
        self.config = None
        self.generation_config = None
        self.llama33_config = None
        self.chat_template = None
        
        # Load Llama 3.3 specific configuration
        self._load_llama33_config()
    
    def _load_llama33_config(self):
        """Load Llama 3.3 specific configuration."""
        config_path = Path(self.model_path) / "llama33_config.json"
        if config_path.exists():
            with open(config_path, 'r') as f:
                self.llama33_config = json.load(f)
            logger.info("Loaded Llama 3.3 configuration")
        else:
            # Default configuration if file doesn't exist
            self.llama33_config = self._get_default_config()
            logger.warning("Using default Llama 3.3 configuration")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default Llama 3.3 configuration."""
        return {
            "model_name": "llama-3.3-70b-instruct",
            "model_type": "llama33",
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
            "system_prompt": "You are a helpful, harmless, and honest AI assistant. You are Llama 3.3, created by Meta.",
            "generation_config": {
                "do_sample": True,
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
                "use_flash_attention": True,
                "torch_dtype": "bfloat16",
                "device_map": "auto",
                "attn_implementation": "flash_attention_2"
            }
        }
    
    async def load(self) -> None:
        """Load Llama 3.3 model with optimized configuration."""
        try:
            logger.info(f"Loading Llama 3.3 70B model from {self.model_path}")
            
            loop = asyncio.get_event_loop()
            
            # Load tokenizer first
            logger.info("Loading tokenizer...")
            self.tokenizer = await loop.run_in_executor(
                None,
                self._load_tokenizer
            )
            
            # Load model configuration
            logger.info("Loading model configuration...")
            # Load config with proper path handling
            if Path(self.model_path).exists():
                self.config = await loop.run_in_executor(
                    None,
                    lambda: AutoConfig.from_pretrained(self.model_path, local_files_only=True)
                )
            else:
                self.config = await loop.run_in_executor(
                    None,
                    lambda: AutoConfig.from_pretrained(self.model_path)
                )
            
            # Setup generation configuration
            self._setup_generation_config()
            
            # Load model with optimizations
            logger.info("Loading model with optimizations... This may take several minutes...")
            self.model = await loop.run_in_executor(
                None,
                self._load_model_optimized
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            self.is_loaded = True
            logger.info("Llama 3.3 70B model loaded successfully")
            
            # Log model info
            self._log_model_info()
            
        except Exception as e:
            logger.error(f"Failed to load Llama 3.3 model: {str(e)}")
            raise
    
    def _load_tokenizer(self):
        """Load tokenizer with proper configuration."""
        # Check if path exists locally
        if Path(self.model_path).exists():
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=False,
                local_files_only=True
            )
        else:
            # Fallback to HuggingFace repo
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                use_fast=True,
                trust_remote_code=False
            )
        
        # Ensure proper special tokens
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        return tokenizer
    
    def _load_model_optimized(self):
        """Load model with hardware optimizations."""
        from app.config.settings import get_settings
        
        settings = get_settings()
        hw_config = self.llama33_config.get("hardware_optimization", {})
        
        # Get EC2 performance config for CPU optimization
        ec2_config = settings._ec2_config.get('performance', {})
        
        # Setup torch dtype - prioritize EC2 config for CPU optimization
        dtype_name = ec2_config.get("torch_dtype", hw_config.get("torch_dtype", "float32"))
        
        # Handle half-precision optimization for AMD EPYC
        if not torch.cuda.is_available():
            if dtype_name == "float16":
                # float16 is supported on AMD EPYC for CPU inference
                logger.info("Using float16 (half-precision) for AMD EPYC CPU inference")
            elif dtype_name == "bfloat16":
                # bfloat16 may not work well on CPU, fallback to float16 for half-precision
                dtype_name = "float16"
                logger.info("Converting bfloat16 to float16 for AMD EPYC CPU inference")
            elif ec2_config.get("use_half_precision", False):
                # Force half-precision if explicitly enabled
                dtype_name = "float16"
                logger.info("Forcing float16 (half-precision) as requested in EC2 config")
        
        torch_dtype = getattr(torch, dtype_name)
        
        # Setup device map - force CPU for c6a instances
        if not torch.cuda.is_available():
            device_map = "cpu"
            logger.info("Using CPU device map (no CUDA available)")
        else:
            device_map = hw_config.get("device_map", "auto")
        
        # Setup attention implementation (GPU only)
        attn_implementation = None
        if torch.cuda.is_available() and hw_config.get("use_flash_attention", True):
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning("Flash Attention not available, using default attention")
        
        # Setup quantization - prioritize EC2 config for half-precision optimization
        quantization_config = None
        use_8bit = ec2_config.get("load_in_8bit", hw_config.get("load_in_8bit", False))
        use_4bit = ec2_config.get("load_in_4bit", hw_config.get("load_in_4bit", False))
        use_half_precision = ec2_config.get("use_half_precision", False)
        
        # For CPU inference with half-precision optimization
        if not torch.cuda.is_available():
            if use_half_precision:
                # Skip quantization for half-precision optimization
                logger.info("Using half-precision (FP16) optimization - skipping quantization")
                quantization_config = None
            elif use_8bit:
                logger.warning("8-bit quantization not optimal for CPU, using half-precision instead")
                quantization_config = None
            elif use_4bit:
                logger.warning("4-bit quantization disabled for half-precision optimization")
                quantization_config = None
        else:
            # GPU quantization settings (if CUDA available)
            if use_8bit:
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                logger.info("Using 8-bit quantization")
            elif use_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch_dtype,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
                logger.info("Using 4-bit quantization")
        
        # Apply AMD EPYC specific optimizations before model loading
        if not torch.cuda.is_available():
            self._apply_amd_epyc_optimizations(ec2_config)
        
        # Load model with proper path handling
        if Path(self.model_path).exists():
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                trust_remote_code=False,
                low_cpu_mem_usage=True,
                local_files_only=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                config=self.config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                quantization_config=quantization_config,
                attn_implementation=attn_implementation,
                trust_remote_code=False,
                low_cpu_mem_usage=True
            )
        
        return model
    
    def _apply_amd_epyc_optimizations(self, ec2_config: Dict[str, Any]):
        """Apply AMD EPYC specific optimizations for CPU inference."""
        import os
        
        # Set threading optimizations
        torch_threads = ec2_config.get("torch_threads", 96)
        mkl_threads = ec2_config.get("mkl_threads", 96)
        omp_threads = ec2_config.get("omp_num_threads", 96)
        
        # Apply PyTorch threading
        torch.set_num_threads(torch_threads)
        logger.info(f"Set PyTorch threads to {torch_threads}")
        
        # Apply MKL optimizations if available
        if ec2_config.get("use_mkl", True):
            try:
                import mkl
                mkl.set_num_threads(mkl_threads)
                logger.info(f"Set MKL threads to {mkl_threads}")
            except ImportError:
                logger.warning("MKL not available, using default threading")
        
        # Set OpenMP threads
        os.environ["OMP_NUM_THREADS"] = str(omp_threads)
        logger.info(f"Set OMP_NUM_THREADS to {omp_threads}")
        
        # Enable AMD EPYC specific optimizations
        if ec2_config.get("use_avx2", True):
            os.environ["TORCH_USE_AVX2"] = "1"
            logger.info("Enabled AVX2 SIMD instructions")
        
        if ec2_config.get("use_avx512", True):
            os.environ["TORCH_USE_AVX512"] = "1"
            logger.info("Enabled AVX-512 SIMD instructions")
        
        # Enable vectorization optimizations
        if ec2_config.get("vectorization", True):
            os.environ["TORCH_ENABLE_VECTORIZATION"] = "1"
            logger.info("Enabled vectorization optimizations")
        
        # Memory allocation optimizations for large models
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        logger.info("Applied memory allocation optimizations")
    
    def _setup_generation_config(self):
        """Setup generation configuration."""
        gen_config = self.llama33_config.get("generation_config", {})
        
        self.generation_config = GenerationConfig(
            do_sample=gen_config.get("do_sample", True),
            temperature=gen_config.get("temperature", 0.7),
            top_p=gen_config.get("top_p", 0.9),
            top_k=gen_config.get("top_k", 50),
            repetition_penalty=gen_config.get("repetition_penalty", 1.1),
            max_new_tokens=gen_config.get("max_new_tokens", 4096),
            pad_token_id=gen_config.get("pad_token_id", 128001),
            eos_token_id=gen_config.get("eos_token_id", [128001, 128009]),
            bos_token_id=gen_config.get("bos_token_id", 128000)
        )
    
    def _log_model_info(self):
        """Log model information."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            # Log CPU information for AMD EPYC
            import psutil
            cpu_info = psutil.cpu_count(logical=True)
            memory_gb = psutil.virtual_memory().total / (1024**3)
            logger.info(f"CPU cores: {cpu_info} vCPUs")
            logger.info(f"System memory: {memory_gb:.1f}GB")
            logger.info("Running on AMD EPYC c6a.24xlarge with FP16 optimization")
        
        # Log model parameters and precision
        if hasattr(self.model, 'num_parameters'):
            num_params = self.model.num_parameters() / 1e9
            logger.info(f"Model parameters: {num_params:.1f}B")
        
        # Log precision information
        if hasattr(self.model, 'dtype'):
            logger.info(f"Model precision: {self.model.dtype}")
        
        logger.info(f"Context length: {self.config.max_position_embeddings}")
        logger.info(f"Vocabulary size: {self.config.vocab_size}")
        
        # Log optimization status
        from app.config.settings import get_settings
        settings = get_settings()
        ec2_config = settings._ec2_config.get('performance', {})
        if ec2_config.get("use_half_precision", False):
            logger.info("Half-precision (FP16) optimization: ENABLED")
        if ec2_config.get("use_mkl", False):
            logger.info("Intel MKL optimization: ENABLED")
        if ec2_config.get("vectorization", False):
            logger.info("SIMD vectorization: ENABLED")
    
    def _format_chat_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages using Llama 3.3 chat template."""
        template = self.llama33_config["chat_template"]
        formatted_parts = [template["begin_of_text"]]
        
        for message in messages:
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_parts.extend([
                    template["system_start"],
                    "\n\n" + content + "\n\n",
                    template["system_end"]
                ])
            elif role == "user":
                formatted_parts.extend([
                    template["user_start"],
                    "\n\n" + content + "\n\n",
                    template["user_end"]
                ])
            elif role == "assistant":
                formatted_parts.extend([
                    template["assistant_start"],
                    "\n\n" + content + "\n\n",
                    template["assistant_end"]
                ])
        
        # Add assistant start for generation
        formatted_parts.append(template["assistant_start"] + "\n\n")
        
        return "".join(formatted_parts)
    
    async def predict(self, input_data: Any, **kwargs) -> Any:
        """Make prediction with Llama 3.3 model."""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Handle different input types
            if isinstance(input_data, str):
                # Simple text input
                prompt = input_data
            elif isinstance(input_data, list):
                # Chat messages format
                prompt = self._format_chat_messages(input_data)
            elif isinstance(input_data, dict):
                # Single message
                if "messages" in input_data:
                    prompt = self._format_chat_messages(input_data["messages"])
                else:
                    prompt = input_data.get("content", str(input_data))
            else:
                prompt = str(input_data)
            
            # Get generation parameters
            max_new_tokens = kwargs.get("max_new_tokens", self.generation_config.max_new_tokens)
            temperature = kwargs.get("temperature", self.generation_config.temperature)
            top_p = kwargs.get("top_p", self.generation_config.top_p)
            top_k = kwargs.get("top_k", self.generation_config.top_k)
            repetition_penalty = kwargs.get("repetition_penalty", self.generation_config.repetition_penalty)
            
            # Tokenize input
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                None,
                lambda: self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=min(len(prompt.split()) + max_new_tokens, 
                                 self.config.max_position_embeddings)
                )
            )
            
            # Move to appropriate device
            if hasattr(self.model, 'device'):
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
            elif torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Generate response
            start_time = time.time()
            
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.model.generate(
                        **inputs,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        top_k=top_k,
                        repetition_penalty=repetition_penalty,
                        do_sample=temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.generation_config.eos_token_id,
                        use_cache=True
                    )
                )
            
            generation_time = time.time() - start_time
            
            # Decode response
            response_ids = outputs[0][inputs["input_ids"].shape[-1]:]
            response = self.tokenizer.decode(response_ids, skip_special_tokens=True)
            
            # Clean up response
            response = response.strip()
            
            # Log generation stats
            num_tokens = len(response_ids)
            tokens_per_second = num_tokens / generation_time if generation_time > 0 else 0
            logger.info(f"Generated {num_tokens} tokens in {generation_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
            
            return response
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    async def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Chat interface for conversational use."""
        # Add default system message if not present
        if not messages or messages[0]["role"] != "system":
            system_prompt = self.llama33_config.get("system_prompt", 
                "You are a helpful, harmless, and honest AI assistant.")
            messages = [{"role": "system", "content": system_prompt}] + messages
        
        return await self.predict(messages, **kwargs)
    
    def get_chat_template(self) -> Dict[str, str]:
        """Get the chat template configuration."""
        return self.llama33_config.get("chat_template", {})
    
    def get_supported_languages(self) -> List[str]:
        """Get list of supported languages."""
        return [
            "English", "German", "French", "Italian", 
            "Portuguese", "Hindi", "Spanish", "Thai"
        ]
    
    async def unload(self) -> None:
        """Unload Llama 3.3 model and free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.config is not None:
            del self.config
            self.config = None
        
        if self.generation_config is not None:
            del self.generation_config
            self.generation_config = None
        
        self.is_loaded = False
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        logger.info("Llama 3.3 model unloaded and GPU memory cleared")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        info = super().get_model_info()
        
        if self.llama33_config:
            from app.config.settings import get_settings
            settings = get_settings()
            ec2_config = settings._ec2_config.get('performance', {})
            
            info.update({
                "model_name": self.llama33_config.get("model_name", "llama-3.3-70b-instruct"),
                "model_type": "llama33",
                "context_length": 131072,
                "supported_languages": self.get_supported_languages(),
                "chat_template_available": True,
                "flash_attention_enabled": self.llama33_config.get("hardware_optimization", {}).get("use_flash_attention", False),
                "multi_gpu_optimized": True,
                "half_precision_enabled": ec2_config.get("use_half_precision", False),
                "precision_type": ec2_config.get("torch_dtype", "float32"),
                "amd_epyc_optimized": not torch.cuda.is_available(),
                "mkl_enabled": ec2_config.get("use_mkl", False),
                "vectorization_enabled": ec2_config.get("vectorization", False),
                "avx2_enabled": ec2_config.get("use_avx2", False),
                "avx512_enabled": ec2_config.get("use_avx512", False)
            })
        
        return info 
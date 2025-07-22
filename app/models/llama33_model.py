"""
Llama 3.3 70B Instruct model implementation with optimized configuration.
"""

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
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
from app.optimizations.matrix_ops import get_matrix_ops


class Llama33Model(BaseModel):
    """Llama 3.3 70B Instruct model implementation with optimized configuration."""
    
    def __init__(self, model_path: str = None, device: str = "cpu", quantization_config: Optional[Dict] = None):
        """
        Initialize LLaMA 3.3 70B model.
        
        Args:
            model_path: Path to model files
            device: Device to run on ("cpu" or "cuda")
            quantization_config: Configuration for quantization
        """
        self.model_path = model_path
        self.device = device
        self.quantization_config = quantization_config
        self.model = None
        self.tokenizer = None
        self.config = None
        self.is_quantized = quantization_config is not None
        self.quantizer = None
        
        # Performance tracking
        self.performance_stats = {
            'inference_times': [],
            'memory_usage': [],
            'tokens_per_second': []
        }
        
        logger.info(f"Initializing LLaMA 3.3 70B model on {device}")
        if self.is_quantized:
            logger.info(f"Quantization enabled: {quantization_config}")
        
        # Initialize quantizer if needed
        if self.is_quantized:
            from ..optimizations.quantization import LLaMAQuantizer, create_quantization_config
            
            if isinstance(quantization_config, dict):
                quant_config = create_quantization_config(**quantization_config)
            else:
                quant_config = quantization_config
            
            self.quantizer = LLaMAQuantizer(quant_config)
        
        # Initialize assembly optimizations
        self.matrix_ops = get_matrix_ops()
        if self.matrix_ops.use_optimizations:
            logger.info("🚀 Assembly optimizations enabled for LLaMA model")
        else:
            logger.warning("⚠️ Assembly optimizations not available, using fallback")
        
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
    
    async def load(self, use_quantization: bool = None):
        """
        Load the model and tokenizer with optional quantization.
        
        Args:
            use_quantization: Override quantization setting
        """
        if use_quantization is not None:
            self.is_quantized = use_quantization
        
        try:
            logger.info("🔄 Loading LLaMA 3.3 70B model...")
            
            # Load tokenizer
            await self._load_tokenizer()
            
            # Load model
            if self.is_quantized:
                await self._load_quantized_model()
            else:
                await self._load_standard_model()
            
            # Setup generation config
            self._setup_generation_config()
            
            logger.info("✅ Model loaded successfully")
            
            # Print model info
            self._print_model_info()
            
        except Exception as e:
            logger.error(f"❌ Failed to load model: {e}")
            raise
    
    async def _load_standard_model(self):
        """Load standard (non-quantized) model."""
        from transformers import AutoModelForCausalLM
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        if self.device == "cpu":
            self.model = self.model.to(self.device)
    
    async def _load_quantized_model(self):
        """Load model with quantization."""
        from transformers import AutoModelForCausalLM
        from ..optimizations.quantization import estimate_model_size, print_quantization_summary
        
        logger.info("Loading model for quantization...")
        
        # Load original model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float32,  # Load in FP32 for quantization
            device_map=None,  # Load on CPU first
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
        
        # Estimate original model size
        original_size = estimate_model_size(self.model)
        
        # Apply quantization
        logger.info("Applying quantization...")
        self.model = self.quantizer.quantize_model(self.model)
        
        # Print quantization summary
        print_quantization_summary(original_size, self.quantizer.config.bits)
        
        # Move to target device
        if self.device != "cpu":
            self.model = self.model.to(self.device)
    
    def _print_model_info(self):
        """Print model information."""
        if self.model is None:
            return
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info("📊 Model Information")
        logger.info("=" * 40)
        logger.info(f"Model: LLaMA 3.3 70B Instruct")
        logger.info(f"Device: {self.device}")
        logger.info(f"Quantized: {self.is_quantized}")
        if self.is_quantized:
            logger.info(f"Quantization: {self.quantizer.config.bits}-bit")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        # Memory usage
        if torch.cuda.is_available() and self.device == "cuda":
            memory_allocated = torch.cuda.memory_allocated() / 1024**3
            logger.info(f"GPU memory allocated: {memory_allocated:.2f} GB")
    
    async def predict(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
        use_cache: bool = True
    ) -> str:
        """
        Generate text with optimized inference.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            do_sample: Whether to use sampling
            use_cache: Whether to use KV cache
            
        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        
        start_time = time.time()
        
        try:
            # Tokenize input
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
            attention_mask = inputs["attention_mask"].to(self.device)
            
            # Generate with optimized settings
            with torch.no_grad():
                if self.is_quantized:
                    outputs = self._quantized_generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        use_cache=use_cache
                    )
                else:
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        temperature=temperature,
                        top_p=top_p,
                        do_sample=do_sample,
                        use_cache=use_cache,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][len(input_ids[0]):], 
                skip_special_tokens=True
            )
            
            # Track performance
            inference_time = time.time() - start_time
            tokens_generated = len(outputs[0]) - len(input_ids[0])
            tokens_per_second = tokens_generated / inference_time if inference_time > 0 else 0
            
            self.performance_stats['inference_times'].append(inference_time)
            self.performance_stats['tokens_per_second'].append(tokens_per_second)
            
            logger.info(f"Generated {tokens_generated} tokens in {inference_time:.2f}s ({tokens_per_second:.1f} tokens/s)")
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
    
    def _quantized_generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        do_sample: bool,
        use_cache: bool
    ) -> torch.Tensor:
        """Optimized generation for quantized models."""
        
        # Use model's generate method with quantization optimizations
        return self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            use_cache=use_cache,
            pad_token_id=self.tokenizer.eos_token_id,
            # Additional optimizations for CPU inference
            num_beams=1,  # Disable beam search for speed
            early_stopping=False,
            length_penalty=1.0
        )
    
    def save_quantized_model(self, save_path: str):
        """Save quantized model to disk."""
        if not self.is_quantized or self.quantizer is None:
            raise RuntimeError("Model is not quantized")
        
        self.quantizer.save_quantized_model(self.model, save_path)
        
        # Also save tokenizer
        tokenizer_path = os.path.join(os.path.dirname(save_path), "tokenizer")
        self.tokenizer.save_pretrained(tokenizer_path)
        
        logger.info(f"Quantized model and tokenizer saved to {save_path}")
    
    def load_quantized_model(self, model_path: str):
        """Load pre-quantized model from disk."""
        if self.quantizer is None:
            raise RuntimeError("Quantizer not initialized")
        
        # Load tokenizer
        tokenizer_path = os.path.join(os.path.dirname(model_path), "tokenizer")
        if os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Load quantized model
        self.model = self.quantizer.load_quantized_model(self.model, model_path)
        self.is_quantized = True
        
        logger.info(f"Quantized model loaded from {model_path}")
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics."""
        if not self.performance_stats['inference_times']:
            return {}
        
        return {
            'avg_inference_time': np.mean(self.performance_stats['inference_times']),
            'avg_tokens_per_second': np.mean(self.performance_stats['tokens_per_second']),
            'total_inferences': len(self.performance_stats['inference_times']),
            'min_tokens_per_second': np.min(self.performance_stats['tokens_per_second']),
            'max_tokens_per_second': np.max(self.performance_stats['tokens_per_second'])
        }
    
    def reset_performance_stats(self):
        """Reset performance tracking."""
        self.performance_stats = {
            'inference_times': [],
            'memory_usage': [],
            'tokens_per_second': []
        }
    
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
        
        self.tokenizer = tokenizer
    
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
    
    def _apply_optimization_hooks(self):
        """Apply assembly optimization hooks to model layers."""
        if not hasattr(self, '_hooks_applied'):
            try:
                # Hook into linear layers for matrix multiplication optimization
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        # Replace forward method with optimized version
                        original_forward = module.forward
                        
                        def optimized_forward(x, original_fn=original_forward, layer_name=name):
                            # Use our assembly-optimized matrix operations
                            if self.matrix_ops.use_optimizations and x.dim() == 2:
                                try:
                                    return self.matrix_ops.linear_forward(x, module.weight, module.bias)
                                except Exception as e:
                                    logger.warning(f"Assembly optimization failed for {layer_name}, falling back: {e}")
                                    return original_fn(x)
                            else:
                                return original_fn(x)
                        
                        module.forward = optimized_forward
                
                self._hooks_applied = True
                logger.info("🔧 Applied assembly optimization hooks to model layers")
                
            except Exception as e:
                logger.error(f"Failed to apply optimization hooks: {e}")
    
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
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
            self.config = await loop.run_in_executor(
                None,
                AutoConfig.from_pretrained,
                self.model_path
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
        hw_config = self.llama33_config.get("hardware_optimization", {})
        
        # Setup torch dtype
        torch_dtype = getattr(torch, hw_config.get("torch_dtype", "bfloat16"))
        
        # Setup device map for multi-GPU
        device_map = hw_config.get("device_map", "auto")
        
        # Setup attention implementation
        attn_implementation = None
        if hw_config.get("use_flash_attention", True):
            try:
                import flash_attn
                attn_implementation = "flash_attention_2"
                logger.info("Using Flash Attention 2")
            except ImportError:
                logger.warning("Flash Attention not available, using default attention")
        
        # Setup quantization if needed
        quantization_config = None
        if hw_config.get("load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            logger.info("Using 8-bit quantization")
        elif hw_config.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            logger.info("Using 4-bit quantization")
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            config=self.config,
            torch_dtype=torch_dtype,
            device_map=device_map,
            quantization_config=quantization_config,
            attn_implementation=attn_implementation,
            trust_remote_code=False,
            low_cpu_mem_usage=True,
            use_cache=True
        )
        
        return model
    
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
            bos_token_id=gen_config.get("bos_token_id", 128000),
            use_cache=True
        )
    
    def _log_model_info(self):
        """Log model information."""
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / (1024**3)
                logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        
        # Log model parameters
        if hasattr(self.model, 'num_parameters'):
            num_params = self.model.num_parameters() / 1e9
            logger.info(f"Model parameters: {num_params:.1f}B")
        
        logger.info(f"Context length: {self.config.max_position_embeddings}")
        logger.info(f"Vocabulary size: {self.config.vocab_size}")
    
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
            info.update({
                "model_name": self.llama33_config.get("model_name", "llama-3.3-70b-instruct"),
                "model_type": "llama33",
                "context_length": 131072,
                "supported_languages": self.get_supported_languages(),
                "chat_template_available": True,
                "flash_attention_enabled": self.llama33_config.get("hardware_optimization", {}).get("use_flash_attention", False),
                "multi_gpu_optimized": True
            })
        
        return info 
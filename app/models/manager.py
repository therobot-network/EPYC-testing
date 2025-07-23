"""
Model manager for loading, managing, and serving ML models.
"""

import asyncio
import time
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from loguru import logger

from app.config.settings import get_settings
from app.models.base import BaseModel
from app.models.pytorch_model import PyTorchModel
from app.models.transformers_model import TransformersModel
from app.models.llama33_model import Llama33Model
from app.models.llama31_model import Llama31Model
from app.models.llama2_13b_model import Llama2_13BModel


class ModelManager:
    """Manages multiple models and their lifecycle."""
    
    def __init__(self):
        self.models: Dict[str, BaseModel] = {}
        self.model_info: Dict[str, Dict[str, Any]] = {}
        self.settings = get_settings()
        self._lock = asyncio.Lock()
        
        # Apply EC2 performance optimizations
        self._apply_performance_optimizations()
    
    def _apply_performance_optimizations(self):
        """Apply EC2-specific performance optimizations."""
        # Set PyTorch thread count based on EC2 configuration
        if self.settings.torch_threads:
            torch.set_num_threads(self.settings.torch_threads)
            logger.info(f"Set PyTorch threads to {self.settings.torch_threads}")
        
        # Log performance configuration
        if self.settings.ec2_instance_type:
            logger.info(f"Running on EC2 instance type: {self.settings.ec2_instance_type}")
            logger.info(f"CPU cores available: {self.settings.cpu_cores}")
            logger.info(f"Max models in memory: {self.settings.max_models_in_memory}")
            logger.info(f"Max batch size: {self.settings.max_batch_size}")
    
    async def load_model(
        self, 
        model_name: str, 
        model_path: str, 
        model_type: str = "auto",
        force_reload: bool = False
    ) -> None:
        """Load a model into memory."""
        async with self._lock:
            if model_name in self.models and not force_reload:
                logger.info(f"Model '{model_name}' already loaded")
                return
            
            # Check memory limits
            if len(self.models) >= self.settings.max_models_in_memory:
                await self._evict_oldest_model()
            
            try:
                logger.info(f"Loading model '{model_name}' from {model_path}")
                
                # Determine model type and create appropriate model instance
                if model_type == "auto":
                    model_type = self._detect_model_type(model_path)
                
                model = self._create_model(model_type, model_path)
                await model.load()
                
                self.models[model_name] = model
                self.model_info[model_name] = {
                    "status": "loaded",
                    "loaded_at": datetime.now().isoformat(),
                    "model_path": model_path,
                    "model_type": model_type,
                    "memory_usage": self._get_model_memory_usage(model),
                    "ec2_optimized": True if self.settings.ec2_instance_type else False
                }
                
                logger.info(f"Successfully loaded model '{model_name}'")
                
            except Exception as e:
                logger.error(f"Failed to load model '{model_name}': {str(e)}")
                raise ValueError(f"Failed to load model: {str(e)}")
    
    async def unload_model(self, model_name: str) -> None:
        """Unload a model from memory."""
        async with self._lock:
            if model_name not in self.models:
                raise ValueError(f"Model '{model_name}' not found")
            
            try:
                logger.info(f"Unloading model '{model_name}'")
                
                model = self.models.pop(model_name)
                await model.unload()
                
                self.model_info.pop(model_name, None)
                
                # Force garbage collection
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                logger.info(f"Successfully unloaded model '{model_name}'")
                
            except Exception as e:
                logger.error(f"Failed to unload model '{model_name}': {str(e)}")
                raise ValueError(f"Failed to unload model: {str(e)}")
    
    async def predict(
        self, 
        model_name: str, 
        input_data: Any, 
        parameters: Optional[Dict[str, Any]] = None
    ) -> Tuple[Any, float]:
        """Make a prediction using the specified model."""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not loaded")
        
        model = self.models[model_name]
        parameters = parameters or {}
        
        start_time = time.time()
        try:
            prediction = await model.predict(input_data, **parameters)
            inference_time = time.time() - start_time
            
            logger.info(f"Prediction completed for model '{model_name}' in {inference_time:.3f}s")
            return prediction, inference_time
            
        except Exception as e:
            logger.error(f"Prediction failed for model '{model_name}': {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    async def list_models(self) -> Dict[str, Dict[str, Any]]:
        """List all loaded models and their information."""
        return self.model_info.copy()
    
    async def cleanup(self) -> None:
        """Clean up all loaded models."""
        logger.info("Cleaning up all models...")
        
        for model_name in list(self.models.keys()):
            try:
                await self.unload_model(model_name)
            except Exception as e:
                logger.error(f"Error cleaning up model '{model_name}': {str(e)}")
        
        logger.info("Model cleanup completed")
    
    def _detect_model_type(self, model_path: str) -> str:
        """Detect model type based on path and contents."""
        # Check for HuggingFace model identifiers first
        if "meta-llama/Llama-3.1" in model_path:
            return "llama31"
        elif "meta-llama/Llama-3.3" in model_path:
            return "llama33"
        elif "meta-llama/Llama-2" in model_path:
            return "llama2"
        
        # Check for local path indicators
        if "llama-3.1" in model_path.lower() or "llama31" in model_path.lower():
            return "llama31"
        elif "llama-3.1-8b" in model_path.lower():
            return "llama31"
        # Check for Llama 3.1 config file
        elif Path(model_path).joinpath("llama31_config.json").exists():
            return "llama31"
        # Check for Llama 3.3 specific indicators
        elif "llama-3.3" in model_path.lower() or "llama33" in model_path.lower():
            return "llama33"
        elif "llama-3.3-70b" in model_path.lower():
            return "llama33"
        # Check for Llama 3.3 config file
        elif Path(model_path).joinpath("llama33_config.json").exists():
            return "llama33"
        # Check for Llama 2 specific indicators
        elif "llama-2" in model_path.lower() or "llama2" in model_path.lower():
            return "llama2"
        elif "llama-2-13b" in model_path.lower():
            return "llama2"
        # Check for Llama 2 config file
        elif Path(model_path).joinpath("llama2_config.json").exists():
            return "llama2"
        # Simple heuristics - can be extended
        elif "transformers" in model_path.lower() or any(
            file in model_path for file in ["config.json", "pytorch_model.bin"]
        ):
            return "transformers"
        elif model_path.endswith((".pt", ".pth")):
            return "pytorch"
        else:
            return "llama31"  # Default to Llama 3.1
    
    def _create_model(self, model_type: str, model_path: str) -> BaseModel:
        """Create a model instance based on type."""
        if model_type == "llama31":
            return Llama31Model(model_path)
        elif model_type == "llama33":
            return Llama33Model(model_path)
        elif model_type == "llama2":
            return Llama2_13BModel(model_path)
        elif model_type == "transformers":
            return TransformersModel(model_path)
        elif model_type == "pytorch":
            return PyTorchModel(model_path)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _get_model_memory_usage(self, model: BaseModel) -> Optional[float]:
        """Get approximate memory usage of a model in MB."""
        try:
            if hasattr(model, 'model') and hasattr(model.model, 'parameters'):
                param_size = sum(p.numel() * p.element_size() for p in model.model.parameters())
                return param_size / (1024 * 1024)  # Convert to MB
        except Exception as e:
            logger.warning(f"Could not calculate memory usage: {str(e)}")
        return None
    
    async def _evict_oldest_model(self) -> None:
        """Evict the oldest loaded model to make room for a new one."""
        if not self.models:
            return
        
        # Find the oldest model based on loaded_at timestamp
        oldest_model = min(
            self.model_info.items(),
            key=lambda x: x[1].get("loaded_at", "")
        )
        
        logger.info(f"Evicting oldest model: {oldest_model[0]}")
        await self.unload_model(oldest_model[0]) 
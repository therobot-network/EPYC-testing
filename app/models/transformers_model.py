"""
Transformers model implementation.
"""

import asyncio
from typing import Any, Dict, List, Union

import torch
from loguru import logger
from transformers import AutoModel, AutoTokenizer, AutoConfig

from app.models.base import BaseModel


class TransformersModel(BaseModel):
    """Transformers model implementation."""
    
    def __init__(self, model_path: str):
        super().__init__(model_path)
        self.tokenizer = None
        self.config = None
    
    async def load(self) -> None:
        """Load Transformers model from path."""
        try:
            logger.info(f"Loading Transformers model from {self.model_path}")
            
            loop = asyncio.get_event_loop()
            
            # Load config, tokenizer, and model
            self.config = await loop.run_in_executor(
                None,
                AutoConfig.from_pretrained,
                self.model_path
            )
            
            self.tokenizer = await loop.run_in_executor(
                None,
                AutoTokenizer.from_pretrained,
                self.model_path
            )
            
            self.model = await loop.run_in_executor(
                None,
                AutoModel.from_pretrained,
                self.model_path
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            
            self.is_loaded = True
            logger.info("Transformers model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load Transformers model: {str(e)}")
            raise
    
    async def predict(self, input_data: Any, **kwargs) -> Any:
        """Make prediction with Transformers model."""
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Handle different input types
            if isinstance(input_data, str):
                texts = [input_data]
            elif isinstance(input_data, list):
                texts = input_data
            else:
                raise ValueError("Input must be string or list of strings")
            
            # Tokenize input
            loop = asyncio.get_event_loop()
            inputs = await loop.run_in_executor(
                None,
                lambda: self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=kwargs.get("max_length", 512)
                )
            )
            
            # Move to GPU if model is on GPU
            if next(self.model.parameters()).is_cuda:
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # Make prediction
            with torch.no_grad():
                outputs = await loop.run_in_executor(
                    None,
                    lambda: self.model(**inputs)
                )
            
            # Extract embeddings or logits
            if hasattr(outputs, 'last_hidden_state'):
                # For models that output hidden states
                embeddings = outputs.last_hidden_state
                # Mean pooling
                attention_mask = inputs.get('attention_mask')
                if attention_mask is not None:
                    mask_expanded = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
                    sum_embeddings = torch.sum(embeddings * mask_expanded, 1)
                    sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
                    result = sum_embeddings / sum_mask
                else:
                    result = embeddings.mean(dim=1)
            elif hasattr(outputs, 'logits'):
                # For classification models
                result = outputs.logits
            else:
                # Fallback to first output
                result = outputs[0]
            
            # Convert to CPU and return
            if result.is_cuda:
                result = result.cpu()
            
            return result.numpy().tolist()
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    async def unload(self) -> None:
        """Unload Transformers model and free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        if self.config is not None:
            del self.config
            self.config = None
        
        self.is_loaded = False
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Transformers model unloaded") 
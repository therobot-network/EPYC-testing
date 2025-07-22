"""
PyTorch model implementation.
"""

import asyncio
from typing import Any

import torch
from loguru import logger

from app.models.base import BaseModel


class PyTorchModel(BaseModel):
    """PyTorch model implementation."""
    
    async def load(self) -> None:
        """Load PyTorch model from file."""
        try:
            logger.info(f"Loading PyTorch model from {self.model_path}")
            
            # Load model in a thread to avoid blocking
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                None, 
                torch.load, 
                self.model_path, 
                {"map_location": "cpu"}
            )
            
            # Set model to evaluation mode
            if hasattr(self.model, 'eval'):
                self.model.eval()
            
            # Move to GPU if available
            if torch.cuda.is_available():
                self.model = self.model.cuda()
                logger.info("Model moved to GPU")
            
            self.is_loaded = True
            logger.info("PyTorch model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load PyTorch model: {str(e)}")
            raise
    
    async def predict(self, input_data: Any, **kwargs) -> Any:
        """Make prediction with PyTorch model."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Model not loaded")
        
        try:
            # Convert input to tensor if needed
            if not isinstance(input_data, torch.Tensor):
                input_tensor = torch.tensor(input_data, dtype=torch.float32)
            else:
                input_tensor = input_data
            
            # Move to GPU if model is on GPU
            if next(self.model.parameters()).is_cuda:
                input_tensor = input_tensor.cuda()
            
            # Add batch dimension if needed
            if input_tensor.dim() == 1:
                input_tensor = input_tensor.unsqueeze(0)
            
            # Make prediction
            loop = asyncio.get_event_loop()
            with torch.no_grad():
                prediction = await loop.run_in_executor(
                    None,
                    lambda: self.model(input_tensor)
                )
            
            # Convert back to CPU and numpy if needed
            if prediction.is_cuda:
                prediction = prediction.cpu()
            
            return prediction.numpy().tolist()
            
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            raise
    
    async def unload(self) -> None:
        """Unload PyTorch model and free resources."""
        if self.model is not None:
            del self.model
            self.model = None
        
        self.is_loaded = False
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("PyTorch model unloaded") 
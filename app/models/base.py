"""
Base model class for all model implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseModel(ABC):
    """Abstract base class for all model implementations."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.is_loaded = False
    
    @abstractmethod
    async def load(self) -> None:
        """Load the model from the specified path."""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Any, **kwargs) -> Any:
        """Make a prediction using the loaded model."""
        pass
    
    @abstractmethod
    async def unload(self) -> None:
        """Unload the model and free resources."""
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model."""
        return {
            "model_path": self.model_path,
            "is_loaded": self.is_loaded,
            "model_type": self.__class__.__name__
        } 
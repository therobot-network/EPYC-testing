"""
API routes for model deployment service.
"""

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from loguru import logger
from pydantic import BaseModel

from app.config.settings import get_settings
from app.models.manager import ModelManager


router = APIRouter()


class PredictionRequest(BaseModel):
    """Request model for predictions."""
    input_data: Any
    model_name: Optional[str] = "default"
    parameters: Optional[Dict[str, Any]] = {}


class ChatRequest(BaseModel):
    """Request model for chat conversations."""
    messages: List[Dict[str, str]]
    model_name: Optional[str] = "default"
    max_new_tokens: Optional[int] = 1024
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9
    top_k: Optional[int] = 50


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    prediction: Any
    model_name: str
    inference_time: float


class ModelInfo(BaseModel):
    """Model information."""
    name: str
    status: str
    loaded_at: Optional[str] = None
    memory_usage: Optional[float] = None


class LoadModelRequest(BaseModel):
    """Request model for loading a model."""
    model_name: str
    model_path: str
    force_reload: bool = False


def get_model_manager(request: Request) -> ModelManager:
    """Get model manager from application state."""
    return request.app.state.model_manager


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "model-deployment",
        "version": "1.0.0"
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict(
    request: PredictionRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Make predictions using the specified model."""
    try:
        logger.info(f"Prediction request for model: {request.model_name}")
        
        prediction, inference_time = await model_manager.predict(
            model_name=request.model_name,
            input_data=request.input_data,
            parameters=request.parameters
        )
        
        return PredictionResponse(
            prediction=prediction,
            model_name=request.model_name,
            inference_time=inference_time
        )
        
    except ValueError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during prediction"
        )


@router.get("/models", response_model=List[ModelInfo])
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """List all loaded models."""
    try:
        models = await model_manager.list_models()
        return [
            ModelInfo(
                name=name,
                status=info["status"],
                loaded_at=info.get("loaded_at"),
                memory_usage=info.get("memory_usage")
            )
            for name, info in models.items()
        ]
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error retrieving model list"
        )


@router.post("/models/load")
async def load_model(
    request: LoadModelRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Load a new model."""
    try:
        logger.info(f"Loading model: {request.model_name} from {request.model_path}")
        
        await model_manager.load_model(
            model_name=request.model_name,
            model_path=request.model_path,
            force_reload=request.force_reload
        )
        
        return {
            "message": f"Model '{request.model_name}' loaded successfully",
            "model_name": request.model_name,
            "model_path": request.model_path
        }
        
    except ValueError as e:
        logger.error(f"Model loading error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error loading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during model loading"
        )


@router.post("/chat")
async def chat(
    request: ChatRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Chat with a model using conversation format."""
    try:
        logger.info(f"Chat request for model: {request.model_name}")
        
        if request.model_name not in model_manager.models:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model '{request.model_name}' not found or not loaded"
            )
        
        model = model_manager.models[request.model_name]
        
        # Check if model supports chat interface
        if hasattr(model, 'chat'):
            response = await model.chat(
                messages=request.messages,
                max_new_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k
            )
        else:
            # Fallback to regular prediction
            response, _ = await model_manager.predict(
                model_name=request.model_name,
                input_data=request.messages,
                parameters={
                    "max_new_tokens": request.max_new_tokens,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "top_k": request.top_k
                }
            )
        
        return {
            "response": response,
            "model_name": request.model_name,
            "model_type": model_manager.model_info[request.model_name].get("model_type", "unknown")
        }
        
    except ValueError as e:
        logger.error(f"Chat error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during chat: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during chat"
        )


@router.delete("/models/{model_name}")
async def unload_model(
    model_name: str,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """Unload a model from memory."""
    try:
        await model_manager.unload_model(model_name)
        return {
            "message": f"Model '{model_name}' unloaded successfully",
            "model_name": model_name
        }
    except ValueError as e:
        logger.error(f"Model unloading error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error unloading model: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error during model unloading"
        ) 
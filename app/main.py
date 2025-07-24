"""
Main application entry point for the model deployment service.
"""

import os
from contextlib import asynccontextmanager
from pathlib import Path

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(".env")
    if env_path.exists():
        load_dotenv(env_path)
        print(f"✅ Loaded environment variables from {env_path}")
    else:
        print("⚠️ No .env file found")
except ImportError:
    print("⚠️ python-dotenv not installed, loading environment variables manually")
    # Manual .env loading as fallback
    env_path = Path(".env")
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
        print(f"✅ Manually loaded environment variables from {env_path}")

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.config.settings import get_settings
from app.models.manager import ModelManager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting model deployment service...")
    
    # Initialize model manager
    model_manager = ModelManager()
    app.state.model_manager = model_manager
    
    # Load default model if specified
    settings = get_settings()
    if settings.default_model_path:
        logger.info(f"Loading default model from {settings.default_model_path}")
        await model_manager.load_model("default", settings.default_model_path)
    
    logger.info("Service startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down model deployment service...")
    if hasattr(app.state, 'model_manager'):
        await app.state.model_manager.cleanup()
    logger.info("Service shutdown complete")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()
    
    app = FastAPI(
        title="Model Deployment Service",
        description="A scalable service for deploying and serving ML models",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include API routes
    app.include_router(router, prefix="/api/v1")
    
    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    
    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level.lower(),
    ) 
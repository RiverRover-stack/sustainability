"""
FastAPI Main Application

File Responsibility:
    Entry point for REST API serving ML predictions and recommendations.

Inputs:
    - HTTP requests with consumption data

Outputs:
    - JSON responses with predictions, recommendations, carbon data

Assumptions:
    - Models are trained and saved in models/ directory
    - Running on localhost:8000 by default

Failure Modes:
    - 404 if models not found
    - 500 on prediction errors
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import routes
from .routes import predict, recommend, carbon


# Lifespan handler for startup/shutdown
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown logic.
    
    Purpose: Load models on startup, cleanup on shutdown.
    """
    # Startup: Load models
    print("🚀 Starting Energy Predictor API...")
    predict.load_models()
    print("✓ Models loaded successfully")
    
    yield  # App is running
    
    # Shutdown: Cleanup
    print("👋 Shutting down API...")


# Create FastAPI app
app = FastAPI(
    title="Smart AI Energy Predictor API",
    description="""
    REST API for energy consumption prediction and optimization.
    
    **Features:**
    - 🔮 Predict energy consumption
    - 💡 Get optimization recommendations
    - 🌍 Calculate carbon footprint
    - 📊 Explain predictions with SHAP
    
    **SDG 7 - Affordable and Clean Energy**
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predict.router, prefix="/api/v1", tags=["Prediction"])
app.include_router(recommend.router, prefix="/api/v1", tags=["Recommendations"])
app.include_router(carbon.router, prefix="/api/v1", tags=["Carbon"])


@app.get("/", tags=["Health"])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Smart AI Energy Predictor API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "sdg": "SDG 7 - Affordable and Clean Energy"
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Health check endpoint for container orchestration."""
    models_loaded = predict.are_models_loaded()
    return {
        "status": "healthy" if models_loaded else "degraded",
        "models_loaded": models_loaded,
        "message": "API is operational"
    }


# Error handlers
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all uncaught exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "tip": "Check server logs for details"
        }
    )

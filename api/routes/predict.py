"""
Prediction API Routes

File Responsibility:
    Handle prediction endpoints for energy consumption forecasting.

Inputs:
    - HTTP requests with prediction parameters

Outputs:
    - JSON predictions with optional SHAP explanations

Assumptions:
    - Models are loaded at startup
    - Feature engineering matches training

Failure Modes:
    - 503 if models not loaded
    - 400 on invalid input
"""

import os
import pickle
from datetime import datetime
from typing import Dict, Optional

from fastapi import APIRouter, HTTPException

# Import schemas
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.schemas import PredictionRequest, PredictionResponse

# Router
router = APIRouter()

# Model storage
_models: Dict = {}
_scaler = None


def load_models():
    """
    Load trained models from disk.
    
    Purpose: Initialize models at API startup.
    
    Side effects: Populates _models dictionary
    """
    global _models, _scaler
    
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    
    model_files = {
        'linear_regression': 'linear_regression.pkl',
        'random_forest': 'random_forest.pkl',
        'xgboost': 'xgboost_model.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    _models[name] = pickle.load(f)
                print(f"  ✓ Loaded {name}")
            except Exception as e:
                print(f"  ✗ Failed to load {name}: {e}")


def are_models_loaded() -> bool:
    """Check if any models are loaded."""
    return len(_models) > 0


def _prepare_features(request: PredictionRequest) -> list:
    """
    Prepare features for prediction.
    
    Purpose: Transform request into model input format.
    
    Note: This is a simplified version. Full version should match
    the feature engineering from preprocessing.py
    """
    import numpy as np
    
    # Get or infer values
    hour = request.hour if request.hour is not None else datetime.now().hour
    
    # Basic features (simplified - full version needs more)
    features = [
        request.temperature,
        request.humidity,
        request.occupancy,
        hour,
        1 if request.is_weekend else 0,
        np.sin(2 * np.pi * hour / 24),  # Cyclical hour encoding
        np.cos(2 * np.pi * hour / 24),
    ]
    
    # Pad to match expected feature count (adjust based on your model)
    # This is a simplified approach - production should match exact features
    while len(features) < 20:  # Typical feature count
        features.append(0)
    
    return features


@router.post(
    "/predict",
    response_model=PredictionResponse,
    summary="Predict energy consumption",
    description="Get energy consumption prediction for given conditions"
)
async def predict(request: PredictionRequest):
    """
    Predict energy consumption.
    
    Purpose: Main prediction endpoint.
    
    Inputs:
        request: PredictionRequest with temperature, humidity, etc.
    
    Outputs:
        PredictionResponse with predicted kWh
    """
    if not are_models_loaded():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Please try again later."
        )
    
    try:
        import numpy as np
        
        # Prepare features
        features = _prepare_features(request)
        X = np.array([features])
        
        # Use best model (XGBoost if available)
        model_name = 'xgboost' if 'xgboost' in _models else list(_models.keys())[0]
        model = _models[model_name]
        
        # Predict
        prediction = float(model.predict(X)[0])
        
        return PredictionResponse(
            predicted_kwh=round(prediction, 4),
            model_used=model_name,
            confidence="high" if prediction > 0 else "unknown",
            timestamp=datetime.now(),
            explanation=None  # Add SHAP in production
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.get(
    "/models",
    summary="List available models",
    description="Get list of loaded models and their status"
)
async def list_models():
    """List available prediction models."""
    return {
        "models": list(_models.keys()),
        "count": len(_models),
        "default": "xgboost" if "xgboost" in _models else (list(_models.keys())[0] if _models else None)
    }

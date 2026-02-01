"""
Pydantic Schemas for API Request/Response Validation

File Responsibility:
    Define data models for API input validation and output serialization.

Inputs:
    N/A (schema definitions)

Outputs:
    Validated data models

Assumptions:
    - Pydantic v2 syntax
    - All numeric values use reasonable ranges

Failure Modes:
    - ValidationError on invalid input
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# ============================================
# Prediction Schemas
# ============================================

class PredictionRequest(BaseModel):
    """
    Request model for energy consumption prediction.
    
    Purpose: Validate input for prediction endpoint.
    """
    timestamp: Optional[datetime] = Field(None, description="Prediction timestamp")
    temperature: float = Field(..., ge=-10, le=50, description="Temperature in Celsius")
    humidity: float = Field(60, ge=0, le=100, description="Humidity percentage")
    occupancy: float = Field(0.7, ge=0, le=1, description="Occupancy level (0-1)")
    hour: Optional[int] = Field(None, ge=0, le=23, description="Hour of day")
    is_weekend: bool = Field(False, description="Is weekend day")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "temperature": 32.5,
                "humidity": 65,
                "occupancy": 0.8,
                "hour": 14,
                "is_weekend": False
            }]
        }
    }


class PredictionResponse(BaseModel):
    """
    Response model for energy consumption prediction.
    
    Purpose: Structured prediction output with optional explanation.
    """
    predicted_kwh: float = Field(..., description="Predicted consumption in kWh")
    model_used: str = Field(..., description="Model used for prediction")
    confidence: Optional[str] = Field(None, description="Confidence level")
    timestamp: datetime = Field(default_factory=datetime.now)
    explanation: Optional[Dict[str, float]] = Field(None, description="SHAP feature contributions")


# ============================================
# Recommendation Schemas
# ============================================

class RecommendationItem(BaseModel):
    """Single recommendation item."""
    title: str
    description: str
    priority: str = Field(..., pattern="^(high|medium|low)$")
    category: str
    potential_savings: str
    action_items: List[str] = []


class RecommendationResponse(BaseModel):
    """
    Response model for optimization recommendations.
    
    Purpose: Return prioritized list of energy-saving tips.
    """
    recommendations: List[RecommendationItem]
    total_count: int
    estimated_monthly_savings: Optional[str] = None


# ============================================
# Carbon Schemas
# ============================================

class CarbonRequest(BaseModel):
    """
    Request model for carbon footprint calculation.
    
    Purpose: Validate input for carbon endpoint.
    """
    monthly_kwh: float = Field(..., ge=0, le=10000, description="Monthly consumption in kWh")
    emission_factor: float = Field(0.82, description="kg CO2 per kWh (default: India grid)")
    
    model_config = {
        "json_schema_extra": {
            "examples": [{
                "monthly_kwh": 300,
                "emission_factor": 0.82
            }]
        }
    }


class CarbonEquivalents(BaseModel):
    """Environmental equivalents for carbon emissions."""
    car_km: float = Field(..., description="Equivalent car travel in km")
    flight_km: float = Field(..., description="Equivalent flight distance in km")
    trees_needed: float = Field(..., description="Trees needed to offset annually")


class CarbonResponse(BaseModel):
    """
    Response model for carbon footprint calculation.
    
    Purpose: Return CO2 emissions with context.
    """
    co2_kg: float = Field(..., description="Monthly CO2 emissions in kg")
    co2_annual_kg: float = Field(..., description="Annual CO2 emissions in kg")
    emission_factor: float = Field(..., description="Emission factor used")
    equivalents: CarbonEquivalents
    benchmark_comparison: Dict[str, Any]


# ============================================
# Health Schemas
# ============================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., pattern="^(healthy|degraded|unhealthy)$")
    models_loaded: bool
    message: str

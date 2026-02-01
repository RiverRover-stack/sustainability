"""
Recommendation API Routes

File Responsibility:
    Handle recommendation endpoints for energy optimization tips.

Inputs:
    - Monthly consumption data

Outputs:
    - Prioritized list of recommendations

Assumptions:
    - Recommendation logic matches dashboard

Failure Modes:
    - 400 on invalid consumption values
"""

import os
import sys
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

# Import schemas
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.schemas import RecommendationResponse, RecommendationItem

# Router
router = APIRouter()


@router.get(
    "/recommend",
    response_model=RecommendationResponse,
    summary="Get optimization recommendations",
    description="Get personalized energy-saving recommendations based on consumption"
)
async def get_recommendations(
    monthly_kwh: float = Query(..., ge=0, le=10000, description="Monthly consumption in kWh"),
    include_solar: bool = Query(True, description="Include solar panel recommendations"),
    max_results: int = Query(5, ge=1, le=20, description="Maximum recommendations to return")
):
    """
    Get personalized energy optimization recommendations.
    
    Purpose: Provide actionable energy-saving tips.
    
    Inputs:
        monthly_kwh: Monthly energy consumption
        include_solar: Whether to include solar recommendations
        max_results: Maximum number of recommendations
    
    Outputs:
        RecommendationResponse with prioritized tips
    """
    recommendations = []
    
    # High consumption recommendations
    if monthly_kwh > 400:
        recommendations.append(RecommendationItem(
            title="High Consumption Alert",
            description=f"Your consumption of {monthly_kwh:.0f} kWh/month is above average. Consider an energy audit.",
            priority="high",
            category="consumption",
            potential_savings="20-30% reduction possible",
            action_items=[
                "Schedule professional energy audit",
                "Review appliance usage patterns",
                "Check for phantom loads"
            ]
        ))
    
    # AC optimization (common for high consumers)
    if monthly_kwh > 200:
        recommendations.append(RecommendationItem(
            title="AC Temperature Optimization",
            description="Set AC thermostat to 24-25°C instead of 18-20°C. Each degree saves 3-5% energy.",
            priority="high",
            category="appliance",
            potential_savings="₹500-1000/month",
            action_items=[
                "Set AC to 24°C minimum",
                "Use ceiling fans with AC",
                "Clean AC filters monthly"
            ]
        ))
    
    # Off-peak usage
    recommendations.append(RecommendationItem(
        title="Shift to Off-Peak Hours",
        description="Run washing machines, dishwashers, and water heaters during 11 PM - 5 AM for lower demand.",
        priority="medium",
        category="behavior",
        potential_savings="5-10% on applicable loads",
        action_items=[
            "Set timer for washing machine",
            "Preheat water during off-peak",
            "Charge devices overnight"
        ]
    ))
    
    # LED lighting
    recommendations.append(RecommendationItem(
        title="Switch to LED Lighting",
        description="Replace all bulbs with LED. LEDs use 75% less energy and last 25x longer.",
        priority="medium",
        category="appliance",
        potential_savings="₹200-400/month on lighting",
        action_items=[
            "Replace CFL/incandescent with LED",
            "Use 9W LED instead of 60W bulb",
            "Install motion sensors in common areas"
        ]
    ))
    
    # Solar recommendation
    if include_solar and monthly_kwh > 300:
        monthly_bill = monthly_kwh * 7  # Approximate ₹7/kWh average
        solar_capacity = min(10, monthly_kwh / 120)  # Rough sizing
        
        recommendations.append(RecommendationItem(
            title="Consider Rooftop Solar",
            description=f"With {monthly_kwh:.0f} kWh/month consumption, a {solar_capacity:.1f}kW system could cover 70-80% of needs.",
            priority="medium",
            category="investment",
            potential_savings=f"₹{monthly_bill*0.7:.0f}/month after payback",
            action_items=[
                f"Get quote for {solar_capacity:.1f}kW system",
                "Check PM Surya Ghar subsidy eligibility",
                "Evaluate net metering options"
            ]
        ))
    
    # Standby power
    recommendations.append(RecommendationItem(
        title="Eliminate Standby Power",
        description="Devices on standby consume 5-10% of household energy. Use power strips with switches.",
        priority="low",
        category="behavior",
        potential_savings="₹100-200/month",
        action_items=[
            "Unplug chargers when not in use",
            "Use smart power strips",
            "Turn off set-top boxes at night"
        ]
    ))
    
    # Sort by priority
    priority_order = {"high": 0, "medium": 1, "low": 2}
    recommendations.sort(key=lambda x: priority_order.get(x.priority, 3))
    
    # Limit results
    recommendations = recommendations[:max_results]
    
    # Calculate estimated savings
    monthly_savings = monthly_kwh * 0.15 * 7  # 15% reduction at ₹7/kWh
    
    return RecommendationResponse(
        recommendations=recommendations,
        total_count=len(recommendations),
        estimated_monthly_savings=f"₹{monthly_savings:.0f}"
    )


@router.get(
    "/quick-tips",
    summary="Get quick energy tips",
    description="Get short, actionable energy-saving tips"
)
async def quick_tips():
    """Get a list of quick energy-saving tips."""
    return {
        "tips": [
            "🌡️ Set AC to 24°C - saves 15% energy",
            "💡 Switch to LED - 75% less electricity",
            "⏰ Run appliances during off-peak hours",
            "🔌 Unplug chargers when not in use",
            "🌀 Clean AC filters monthly",
            "☀️ Use natural light during day",
            "🚿 Use solar water heater",
            "⭐ Buy 5-star rated appliances"
        ]
    }

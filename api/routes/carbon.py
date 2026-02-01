"""
Carbon API Routes

File Responsibility:
    Handle carbon footprint calculation endpoints.

Inputs:
    - Monthly consumption in kWh
    - Optional emission factor

Outputs:
    - CO2 emissions with equivalents and benchmarks

Assumptions:
    - Default emission factor is India grid (0.82 kg/kWh)

Failure Modes:
    - 400 on invalid consumption values
"""

from typing import Optional

from fastapi import APIRouter, Query

# Import schemas
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from api.schemas import CarbonRequest, CarbonResponse, CarbonEquivalents

# Router
router = APIRouter()

# Constants
INDIA_EMISSION_FACTOR = 0.82  # kg CO2 per kWh
INDIAN_HOUSEHOLD_AVG = 150  # kWh per month
CAR_KM_PER_KG_CO2 = 6.5  # km per kg CO2
FLIGHT_KM_PER_KG_CO2 = 12  # km per kg CO2
TREE_ABSORPTION_KG_YEAR = 22  # kg CO2 per tree per year


def _calculate_equivalents(co2_kg: float) -> CarbonEquivalents:
    """
    Calculate environmental equivalents.
    
    Purpose: Make carbon numbers tangible.
    
    Inputs:
        co2_kg: Monthly CO2 emissions in kg
    
    Outputs:
        CarbonEquivalents with car km, flight km, trees needed
    """
    return CarbonEquivalents(
        car_km=round(co2_kg * CAR_KM_PER_KG_CO2, 1),
        flight_km=round(co2_kg * FLIGHT_KM_PER_KG_CO2, 1),
        trees_needed=round((co2_kg * 12) / TREE_ABSORPTION_KG_YEAR, 1)  # Annual
    )


def _get_benchmark_comparison(monthly_kwh: float) -> dict:
    """
    Compare consumption to benchmarks.
    
    Purpose: Provide context for consumption level.
    """
    vs_avg_percent = ((monthly_kwh - INDIAN_HOUSEHOLD_AVG) / INDIAN_HOUSEHOLD_AVG) * 100
    
    if monthly_kwh < INDIAN_HOUSEHOLD_AVG * 0.7:
        status = "excellent"
        message = "Great job! You're well below average."
    elif monthly_kwh < INDIAN_HOUSEHOLD_AVG:
        status = "good"
        message = "You're below average. Keep it up!"
    elif monthly_kwh < INDIAN_HOUSEHOLD_AVG * 1.3:
        status = "average"
        message = "Room for improvement. Try our recommendations."
    else:
        status = "high"
        message = "Above average. Significant savings potential."
    
    return {
        "status": status,
        "vs_benchmark_percent": round(vs_avg_percent, 1),
        "benchmark_kwh": INDIAN_HOUSEHOLD_AVG,
        "message": message
    }


@router.post(
    "/carbon",
    response_model=CarbonResponse,
    summary="Calculate carbon footprint",
    description="Calculate CO2 emissions from electricity consumption"
)
async def calculate_carbon(request: CarbonRequest):
    """
    Calculate carbon footprint from electricity consumption.
    
    Purpose: Convert kWh to CO2 with context.
    
    Inputs:
        request: CarbonRequest with monthly_kwh and optional emission_factor
    
    Outputs:
        CarbonResponse with emissions, equivalents, benchmarks
    """
    monthly_co2 = request.monthly_kwh * request.emission_factor
    annual_co2 = monthly_co2 * 12
    
    return CarbonResponse(
        co2_kg=round(monthly_co2, 2),
        co2_annual_kg=round(annual_co2, 2),
        emission_factor=request.emission_factor,
        equivalents=_calculate_equivalents(monthly_co2),
        benchmark_comparison=_get_benchmark_comparison(request.monthly_kwh)
    )


@router.get(
    "/carbon/quick",
    summary="Quick carbon calculation",
    description="Calculate carbon footprint with query parameters"
)
async def quick_carbon(
    kwh: float = Query(..., ge=0, le=10000, description="Monthly consumption in kWh")
):
    """Quick carbon calculation with defaults."""
    co2_kg = kwh * INDIA_EMISSION_FACTOR
    
    return {
        "monthly_kwh": kwh,
        "monthly_co2_kg": round(co2_kg, 2),
        "annual_co2_kg": round(co2_kg * 12, 2),
        "emission_factor": INDIA_EMISSION_FACTOR,
        "car_km_equivalent": round(co2_kg * CAR_KM_PER_KG_CO2, 1),
        "trees_to_offset": round((co2_kg * 12) / TREE_ABSORPTION_KG_YEAR, 1)
    }


@router.get(
    "/carbon/factors",
    summary="Get emission factors",
    description="Get emission factors for different regions"
)
async def emission_factors():
    """Get emission factors for reference."""
    return {
        "india": {
            "factor": 0.82,
            "source": "Central Electricity Authority, 2023",
            "unit": "kg CO2 per kWh"
        },
        "global_average": {
            "factor": 0.50,
            "source": "IEA World Average",
            "unit": "kg CO2 per kWh"
        },
        "renewable": {
            "factor": 0.05,
            "source": "Typical solar/wind average",
            "unit": "kg CO2 per kWh"
        }
    }


@router.get(
    "/carbon/reduction",
    summary="Calculate reduction impact",
    description="Calculate impact of reducing consumption"
)
async def reduction_impact(
    current_kwh: float = Query(..., ge=0, description="Current monthly kWh"),
    reduction_percent: float = Query(10, ge=0, le=100, description="Reduction percentage")
):
    """Calculate the impact of reducing consumption."""
    reduced_kwh = current_kwh * (1 - reduction_percent / 100)
    saved_kwh = current_kwh - reduced_kwh
    saved_co2 = saved_kwh * INDIA_EMISSION_FACTOR
    saved_money = saved_kwh * 7  # Approximate ₹7/kWh
    
    return {
        "current_kwh": current_kwh,
        "reduction_percent": reduction_percent,
        "new_consumption_kwh": round(reduced_kwh, 1),
        "monthly_savings": {
            "kwh": round(saved_kwh, 1),
            "co2_kg": round(saved_co2, 2),
            "rupees": round(saved_money, 0)
        },
        "annual_savings": {
            "kwh": round(saved_kwh * 12, 1),
            "co2_kg": round(saved_co2 * 12, 2),
            "rupees": round(saved_money * 12, 0)
        },
        "trees_equivalent": round(saved_co2 * 12 / TREE_ABSORPTION_KG_YEAR, 1)
    }

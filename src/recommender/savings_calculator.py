"""
Savings Calculator Module

File Responsibility:
    Calculates potential energy and cost savings from
    various efficiency interventions and behavioral changes.

Inputs:
    - Current monthly consumption in kWh

Outputs:
    - Savings scenarios with costs and payback periods

Assumptions:
    - Average electricity rate is ₹5.5/kWh
    - Implementation costs are approximate

Failure Modes:
    - Negative consumption produces incorrect results
"""

from typing import Dict, List


# Appliance energy ratings (kWh per hour of use)
APPLIANCE_RATINGS = {
    'ac': 1.5,
    'heater': 2.0,
    'refrigerator': 0.15,
    'washing_machine': 0.5,
    'water_heater': 2.5,
    'lighting': 0.1,
    'tv': 0.15,
    'computer': 0.2,
    'microwave': 1.2,
    'fan': 0.07
}

# Efficient alternatives with savings potential
EFFICIENT_ALTERNATIVES = {
    'ac': ('Inverter AC', 0.8, '5-star rated inverter AC can save up to 40% energy'),
    'heater': ('Solar Water Heater', 0.0, 'Solar heater can eliminate electricity usage'),
    'lighting': ('LED Bulbs', 0.02, 'LED bulbs use 80% less energy than incandescent'),
    'refrigerator': ('5-Star Refrigerator', 0.08, 'Energy-efficient refrigerators save 30-40%'),
    'fan': ('BLDC Fan', 0.03, 'BLDC fans consume 60% less electricity')
}

# Quick energy saving tips
QUICK_TIPS = [
    "💡 Turn off lights when leaving a room",
    "🌡️ Set AC to 24°C - each degree lower increases consumption by 6%",
    "🔌 Unplug chargers and appliances when not in use",
    "🧊 Don't keep the refrigerator door open for long",
    "👕 Wash clothes in cold water when possible",
    "☀️ Dry clothes in sunlight instead of using a dryer",
    "🪟 Use natural light during daytime",
    "🌬️ Clean AC filters monthly for optimal efficiency",
    "⏰ Use timers and smart plugs for automatic control",
    "📊 Monitor your energy usage regularly"
]


def get_quick_tips() -> List[str]:
    """Return general quick energy-saving tips."""
    return QUICK_TIPS.copy()


def estimate_bill(kwh: float, rate: float = 5.5) -> float:
    """
    Estimate electricity bill from consumption.
    
    Purpose: Simple bill estimation for savings calculations.
    
    Inputs:
        kwh: Consumption in kilowatt-hours
        rate: Average rate in ₹/kWh (default: 5.5)
        
    Outputs:
        Estimated bill amount in ₹
        
    Side effects: None
    """
    return kwh * rate


def calculate_savings_potential(current_monthly_kwh: float) -> Dict:
    """
    Calculate potential savings from various interventions.
    
    Purpose: Show ROI for efficiency investments.
    
    Inputs:
        current_monthly_kwh: Current monthly consumption
        
    Outputs:
        Dictionary of savings scenarios with costs and payback
        
    Side effects: None
    """
    current_cost = estimate_bill(current_monthly_kwh)
    
    scenarios = {
        'led_lighting': {
            'action': 'Switch to LED lighting',
            'savings_percent': 0.10,
            'implementation_cost': 2000,
        },
        'efficient_ac': {
            'action': 'Upgrade to 5-star inverter AC',
            'savings_percent': 0.20,
            'implementation_cost': 40000,
        },
        'solar_3kw': {
            'action': 'Install 3kW solar system',
            'savings_percent': 0.50,
            'implementation_cost': 150000,
        },
        'behavioral_changes': {
            'action': 'Behavioral changes (no cost)',
            'savings_percent': 0.15,
            'implementation_cost': 0,
        }
    }
    
    for key, scenario in scenarios.items():
        monthly_savings = current_cost * scenario['savings_percent']
        scenario['monthly_savings'] = round(monthly_savings, 2)
        scenario['annual_savings'] = round(monthly_savings * 12, 2)
        
        if scenario['implementation_cost'] > 0 and monthly_savings > 0:
            scenario['payback_months'] = round(
                scenario['implementation_cost'] / monthly_savings, 1
            )
        else:
            scenario['payback_months'] = 0
    
    return scenarios


def get_appliance_rating(appliance: str) -> float:
    """Get energy rating for an appliance in kWh/hour."""
    return APPLIANCE_RATINGS.get(appliance.lower(), 0.0)


def get_efficient_alternative(appliance: str) -> tuple:
    """Get efficient alternative for an appliance."""
    return EFFICIENT_ALTERNATIVES.get(appliance.lower(), (None, None, None))

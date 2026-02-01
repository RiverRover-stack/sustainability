"""Recommender package - Energy optimization recommendations."""

from .engine import EnergyRecommender, get_recommendations_for_dashboard
from .pattern_analyzer import (
    analyze_consumption_pattern, identify_high_consumption_periods,
    get_seasonal_consumption, PEAK_HOURS, OFF_PEAK_HOURS
)
from .savings_calculator import (
    calculate_savings_potential, get_quick_tips, estimate_bill,
    APPLIANCE_RATINGS, EFFICIENT_ALTERNATIVES
)

__all__ = [
    'EnergyRecommender', 'get_recommendations_for_dashboard',
    'analyze_consumption_pattern', 'identify_high_consumption_periods',
    'get_seasonal_consumption', 'PEAK_HOURS', 'OFF_PEAK_HOURS',
    'calculate_savings_potential', 'get_quick_tips', 'estimate_bill',
    'APPLIANCE_RATINGS', 'EFFICIENT_ALTERNATIVES'
]

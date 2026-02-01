"""Carbon reduction goals and impact calculations."""

from typing import Dict, List
try:
    from .emission_factors import BENCHMARKS, get_emission_factor
except ImportError:
    from emission_factors import BENCHMARKS, get_emission_factor


def calculate_reduction_impact(current_kwh: float, reduction_percent: float, emission_factor: float = None) -> Dict:
    """Calculate CO2 savings from consumption reduction."""
    if emission_factor is None:
        emission_factor = get_emission_factor('india')
    
    reduction_factor = reduction_percent / 100
    reduced_kwh = current_kwh * (1 - reduction_factor)
    current_co2 = current_kwh * emission_factor
    reduced_co2 = reduced_kwh * emission_factor
    saved_co2 = current_co2 - reduced_co2
    
    return {
        'current_kwh': round(current_kwh, 2),
        'reduced_kwh': round(reduced_kwh, 2),
        'kwh_saved': round(current_kwh - reduced_kwh, 2),
        'current_co2_kg': round(current_co2, 2),
        'reduced_co2_kg': round(reduced_co2, 2),
        'co2_saved_kg': round(saved_co2, 2),
        'reduction_percent': reduction_percent,
        'trees_equivalent': round(saved_co2 * 12 / BENCHMARKS['tree_annual_absorption'], 1)
    }


def get_carbon_goal(current_monthly_kwh: float, target_reduction: float = 20, emission_factor: float = None) -> Dict:
    """Set a carbon reduction goal with milestones."""
    if emission_factor is None:
        emission_factor = get_emission_factor('india')
    
    current_co2 = current_monthly_kwh * emission_factor
    target_co2 = current_co2 * (1 - target_reduction / 100)
    
    milestones = []
    for percent in [25, 50, 75, 100]:
        reduction_so_far = (current_co2 - target_co2) * percent / 100
        milestone_co2 = current_co2 - reduction_so_far
        milestones.append({
            'progress_percent': percent,
            'co2_kg': round(milestone_co2, 2),
            'kwh': round(milestone_co2 / emission_factor, 2)
        })
    
    return {
        'current': {'kwh': round(current_monthly_kwh, 2), 'co2_kg': round(current_co2, 2)},
        'target': {
            'kwh': round(current_monthly_kwh * (1 - target_reduction / 100), 2),
            'co2_kg': round(target_co2, 2),
            'reduction_percent': target_reduction
        },
        'milestones': milestones,
        'annual_impact': {
            'co2_saved_kg': round((current_co2 - target_co2) * 12, 2),
            'trees_equivalent': round((current_co2 - target_co2) * 12 / BENCHMARKS['tree_annual_absorption'], 1)
        }
    }


def get_benchmark_status(co2_kg: float) -> str:
    """Get status relative to Indian household benchmark."""
    benchmark = BENCHMARKS['indian_household_monthly']
    percent_diff = (co2_kg / benchmark - 1) * 100
    
    if percent_diff <= -20: return 'excellent'
    elif percent_diff <= 0: return 'good'
    elif percent_diff <= 20: return 'average'
    else: return 'high'

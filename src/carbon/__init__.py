"""Carbon package - Carbon footprint calculations."""

from .carbon_calculator import CarbonCalculator, calculate_carbon_footprint
from .emission_factors import EMISSION_FACTORS, BENCHMARKS, get_emission_factor, calculate_equivalents
from .carbon_goals import calculate_reduction_impact, get_carbon_goal, get_benchmark_status

__all__ = [
    'CarbonCalculator', 'calculate_carbon_footprint', 'EMISSION_FACTORS',
    'BENCHMARKS', 'get_emission_factor', 'calculate_equivalents',
    'calculate_reduction_impact', 'get_carbon_goal', 'get_benchmark_status'
]

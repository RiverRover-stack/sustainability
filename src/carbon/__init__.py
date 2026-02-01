"""Carbon package - Carbon footprint calculations."""

import sys, os
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_pkg_dir)
if _pkg_dir not in sys.path: sys.path.insert(0, _pkg_dir)
if _src_dir not in sys.path: sys.path.insert(0, _src_dir)

from carbon_calculator import CarbonCalculator, calculate_carbon_footprint
from emission_factors import EMISSION_FACTORS, BENCHMARKS, get_emission_factor, calculate_equivalents
from carbon_goals import calculate_reduction_impact, get_carbon_goal, get_benchmark_status

__all__ = [
    'CarbonCalculator', 'calculate_carbon_footprint', 'EMISSION_FACTORS',
    'BENCHMARKS', 'get_emission_factor', 'calculate_equivalents',
    'calculate_reduction_impact', 'get_carbon_goal', 'get_benchmark_status'
]

"""Emission factors and benchmarks for CO2 calculations."""

from typing import Dict

# Emission factors (kg CO2 per kWh) by source
EMISSION_FACTORS: Dict[str, float] = {
    'india': 0.82, 'coal': 0.91, 'gas': 0.45, 'solar': 0.05,
    'wind': 0.01, 'nuclear': 0.02, 'hydro': 0.02,
}

# Comparison benchmarks
BENCHMARKS: Dict[str, float] = {
    'indian_household_monthly': 150,
    'car_km': 0.12, 'flight_km': 0.255, 'tree_annual_absorption': 22,
    'bus_km': 0.089, 'train_km': 0.041,
}


def get_emission_factor(source: str = 'india') -> float:
    """Get emission factor for energy source (kg CO2/kWh)."""
    return EMISSION_FACTORS.get(source.lower(), EMISSION_FACTORS['india'])


def get_benchmark(key: str) -> float:
    """Get benchmark value by key."""
    return BENCHMARKS.get(key, 0.0)


def calculate_equivalents(co2_kg: float) -> Dict:
    """Convert CO2 emissions to relatable metrics."""
    return {
        'car_km': round(co2_kg / BENCHMARKS['car_km'], 1),
        'flight_km': round(co2_kg / BENCHMARKS['flight_km'], 1),
        'bus_km': round(co2_kg / BENCHMARKS['bus_km'], 1),
        'train_km': round(co2_kg / BENCHMARKS['train_km'], 1),
    }

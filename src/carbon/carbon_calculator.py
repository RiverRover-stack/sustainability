"""Carbon footprint calculator - Main interface for CO2 calculations."""

from typing import Dict
import pandas as pd

try:
    from emission_factors import EMISSION_FACTORS, BENCHMARKS, get_emission_factor, calculate_equivalents
    from carbon_goals import calculate_reduction_impact, get_carbon_goal, get_benchmark_status
except ImportError:
    from carbon.emission_factors import EMISSION_FACTORS, BENCHMARKS, get_emission_factor, calculate_equivalents
    from carbon.carbon_goals import calculate_reduction_impact, get_carbon_goal, get_benchmark_status


class CarbonCalculator:
    """Calculate carbon footprint from electricity consumption."""
    
    EMISSION_FACTORS = EMISSION_FACTORS
    BENCHMARKS = BENCHMARKS
    
    def __init__(self, emission_factor: float = None, country: str = 'india'):
        self.emission_factor = emission_factor or get_emission_factor(country)
    
    def calculate_emissions(self, kwh: float) -> float:
        """Calculate CO2 emissions (kg) from kWh."""
        return kwh * self.emission_factor
    
    def calculate_monthly_emissions(self, df: pd.DataFrame, consumption_col: str = 'consumption_kwh') -> pd.DataFrame:
        """Aggregate emissions by month."""
        df = df.copy()
        df['co2_kg'] = df[consumption_col] * self.emission_factor
        monthly = df.groupby(pd.Grouper(key='timestamp', freq='M')).agg({
            consumption_col: 'sum', 'co2_kg': 'sum'
        }).reset_index()
        monthly.columns = ['month', 'total_kwh', 'total_co2_kg']
        monthly['month_name'] = monthly['month'].dt.strftime('%B %Y')
        return monthly
    
    def get_emission_summary(self, total_kwh: float, period: str = 'month') -> Dict:
        """Get comprehensive emission summary with comparisons."""
        co2_kg = self.calculate_emissions(total_kwh)
        
        if period == 'day': annual_co2 = co2_kg * 365
        elif period == 'month': annual_co2 = co2_kg * 12
        else: annual_co2 = co2_kg
        
        equivalents = calculate_equivalents(co2_kg)
        trees_needed = annual_co2 / BENCHMARKS['tree_annual_absorption']
        benchmark_monthly = BENCHMARKS['indian_household_monthly']
        
        if period == 'month':
            benchmark_comparison = (co2_kg / benchmark_monthly - 1) * 100
            status = get_benchmark_status(co2_kg)
        else:
            benchmark_comparison, status = None, None
        
        return {
            'consumption_kwh': round(total_kwh, 2),
            'co2_kg': round(co2_kg, 2),
            'co2_tonnes': round(co2_kg / 1000, 3),
            'emission_factor': self.emission_factor,
            'period': period,
            'equivalents': {**equivalents, 'trees_needed': round(trees_needed, 1)},
            'benchmark_comparison': {
                'benchmark_kg': benchmark_monthly,
                'vs_benchmark_percent': round(benchmark_comparison, 1) if benchmark_comparison else None,
                'status': status
            }
        }
    
    def calculate_reduction_impact(self, current_kwh: float, reduction_percent: float) -> Dict:
        return calculate_reduction_impact(current_kwh, reduction_percent, self.emission_factor)
    
    def get_carbon_goal(self, current_monthly_kwh: float, target_reduction: float = 20) -> Dict:
        return get_carbon_goal(current_monthly_kwh, target_reduction, self.emission_factor)
    
    def format_for_display(self, co2_kg: float) -> str:
        if co2_kg >= 1000: return f"{co2_kg / 1000:.2f} tonnes CO₂"
        return f"{co2_kg:.1f} kg CO₂"


def calculate_carbon_footprint(df: pd.DataFrame) -> Dict:
    """Complete carbon analysis for dashboard."""
    calc = CarbonCalculator()
    total_kwh = df['consumption_kwh'].sum()
    monthly_emissions = calc.calculate_monthly_emissions(df)
    last_month_kwh = monthly_emissions['total_kwh'].iloc[-1] if len(monthly_emissions) > 0 else 0
    
    return {
        'summary': calc.get_emission_summary(last_month_kwh, period='month'),
        'yearly': calc.get_emission_summary(total_kwh, period='year'),
        'monthly_breakdown': monthly_emissions.to_dict('records'),
        'reduction_scenarios': [calc.calculate_reduction_impact(last_month_kwh, p) for p in [10, 20, 30]],
        'calculator': calc
    }


if __name__ == "__main__":
    calc = CarbonCalculator()
    summary = calc.get_emission_summary(350, period='month')
    print(f"Consumption: {summary['consumption_kwh']} kWh, Emissions: {summary['co2_kg']} kg CO₂")
    print(f"Status: {summary['benchmark_comparison']['status']}")

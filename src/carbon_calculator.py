"""
Smart AI Energy Consumption Predictor
Carbon Calculator Module

Calculates CO2 emissions from electricity consumption using India-specific emission factors.
"""

from typing import Dict, List, Optional
from datetime import datetime
import pandas as pd


class CarbonCalculator:
    """Calculate carbon footprint from electricity consumption."""
    
    # Emission factors (kg CO2 per kWh)
    EMISSION_FACTORS = {
        'india': 0.82,  # India grid average
        'coal': 0.91,   # Coal power
        'gas': 0.45,    # Natural gas
        'solar': 0.05,  # Solar (lifecycle)
        'wind': 0.01,   # Wind (lifecycle)
        'nuclear': 0.02 # Nuclear (lifecycle)
    }
    
    # Comparison benchmarks
    BENCHMARKS = {
        'indian_household_monthly': 150,  # kg CO2 per month (average)
        'car_km': 0.12,  # kg CO2 per km driven
        'flight_km': 0.255,  # kg CO2 per passenger km
        'tree_annual_absorption': 22  # kg CO2 absorbed per tree per year
    }
    
    def __init__(self, emission_factor: float = None, country: str = 'india'):
        """
        Initialize carbon calculator.
        
        Args:
            emission_factor: Custom emission factor (kg CO2/kWh)
            country: Country for default emission factor
        """
        if emission_factor:
            self.emission_factor = emission_factor
        else:
            self.emission_factor = self.EMISSION_FACTORS.get(country, 0.82)
    
    def calculate_emissions(self, kwh: float) -> float:
        """
        Calculate CO2 emissions from electricity consumption.
        
        Args:
            kwh: Energy consumption in kilowatt-hours
            
        Returns:
            CO2 emissions in kg
        """
        return kwh * self.emission_factor
    
    def calculate_monthly_emissions(self, df: pd.DataFrame, 
                                    consumption_col: str = 'consumption_kwh') -> pd.DataFrame:
        """
        Calculate monthly carbon emissions from consumption data.
        
        Args:
            df: DataFrame with timestamp and consumption columns
            consumption_col: Name of consumption column
            
        Returns:
            DataFrame with monthly emissions
        """
        df = df.copy()
        
        # Calculate emissions for each row
        df['co2_kg'] = df[consumption_col] * self.emission_factor
        
        # Aggregate by month
        monthly = df.groupby(pd.Grouper(key='timestamp', freq='M')).agg({
            consumption_col: 'sum',
            'co2_kg': 'sum'
        }).reset_index()
        
        monthly.columns = ['month', 'total_kwh', 'total_co2_kg']
        monthly['month_name'] = monthly['month'].dt.strftime('%B %Y')
        
        return monthly
    
    def get_emission_summary(self, total_kwh: float, period: str = 'month') -> Dict:
        """
        Get comprehensive emission summary with comparisons.
        
        Args:
            total_kwh: Total energy consumption in kWh
            period: Time period ('day', 'month', 'year')
            
        Returns:
            Dictionary with emission details and comparisons
        """
        co2_kg = self.calculate_emissions(total_kwh)
        
        # Annualize for comparisons
        if period == 'day':
            annual_co2 = co2_kg * 365
        elif period == 'month':
            annual_co2 = co2_kg * 12
        else:
            annual_co2 = co2_kg
        
        # Calculate equivalents
        car_km = co2_kg / self.BENCHMARKS['car_km']
        flight_km = co2_kg / self.BENCHMARKS['flight_km']
        trees_needed = annual_co2 / self.BENCHMARKS['tree_annual_absorption']
        
        # Comparison with benchmark
        benchmark_monthly = self.BENCHMARKS['indian_household_monthly']
        if period == 'month':
            benchmark_comparison = (co2_kg / benchmark_monthly - 1) * 100
        else:
            benchmark_comparison = None
        
        return {
            'consumption_kwh': round(total_kwh, 2),
            'co2_kg': round(co2_kg, 2),
            'co2_tonnes': round(co2_kg / 1000, 3),
            'emission_factor': self.emission_factor,
            'period': period,
            'equivalents': {
                'car_km': round(car_km, 1),
                'car_description': f"Equivalent to driving {car_km:.0f} km by car",
                'flight_km': round(flight_km, 1),
                'flight_description': f"Equivalent to {flight_km:.0f} km of air travel",
                'trees_needed': round(trees_needed, 1),
                'trees_description': f"{trees_needed:.0f} trees needed to offset annual emissions"
            },
            'benchmark_comparison': {
                'benchmark_kg': benchmark_monthly,
                'vs_benchmark_percent': round(benchmark_comparison, 1) if benchmark_comparison else None,
                'status': self._get_status(benchmark_comparison) if benchmark_comparison else None
            }
        }
    
    def _get_status(self, percent_diff: float) -> str:
        """Get status based on comparison with benchmark."""
        if percent_diff <= -20:
            return 'excellent'
        elif percent_diff <= 0:
            return 'good'
        elif percent_diff <= 20:
            return 'average'
        else:
            return 'high'
    
    def calculate_reduction_impact(self, current_kwh: float, 
                                   reduction_percent: float) -> Dict:
        """
        Calculate impact of reducing energy consumption.
        
        Args:
            current_kwh: Current consumption in kWh
            reduction_percent: Target reduction percentage (0-100)
            
        Returns:
            Dictionary with reduction impact details
        """
        reduction_factor = reduction_percent / 100
        reduced_kwh = current_kwh * (1 - reduction_factor)
        
        current_co2 = self.calculate_emissions(current_kwh)
        reduced_co2 = self.calculate_emissions(reduced_kwh)
        saved_co2 = current_co2 - reduced_co2
        
        return {
            'current_kwh': round(current_kwh, 2),
            'reduced_kwh': round(reduced_kwh, 2),
            'kwh_saved': round(current_kwh - reduced_kwh, 2),
            'current_co2_kg': round(current_co2, 2),
            'reduced_co2_kg': round(reduced_co2, 2),
            'co2_saved_kg': round(saved_co2, 2),
            'reduction_percent': reduction_percent,
            'trees_equivalent': round(saved_co2 * 12 / self.BENCHMARKS['tree_annual_absorption'], 1)
        }
    
    def get_carbon_goal(self, current_monthly_kwh: float, 
                       target_reduction: float = 20) -> Dict:
        """
        Set a carbon reduction goal and track progress.
        
        Args:
            current_monthly_kwh: Current monthly consumption
            target_reduction: Target reduction percentage
            
        Returns:
            Goal details with milestones
        """
        current_co2 = self.calculate_emissions(current_monthly_kwh)
        target_co2 = current_co2 * (1 - target_reduction / 100)
        
        milestones = []
        for percent in [25, 50, 75, 100]:
            reduction_so_far = (current_co2 - target_co2) * percent / 100
            milestone_co2 = current_co2 - reduction_so_far
            milestones.append({
                'progress_percent': percent,
                'co2_kg': round(milestone_co2, 2),
                'kwh': round(milestone_co2 / self.emission_factor, 2)
            })
        
        return {
            'current': {
                'kwh': round(current_monthly_kwh, 2),
                'co2_kg': round(current_co2, 2)
            },
            'target': {
                'kwh': round(current_monthly_kwh * (1 - target_reduction / 100), 2),
                'co2_kg': round(target_co2, 2),
                'reduction_percent': target_reduction
            },
            'milestones': milestones,
            'annual_impact': {
                'co2_saved_kg': round((current_co2 - target_co2) * 12, 2),
                'trees_equivalent': round((current_co2 - target_co2) * 12 / 
                                         self.BENCHMARKS['tree_annual_absorption'], 1)
            }
        }
    
    def format_for_display(self, co2_kg: float) -> str:
        """Format CO2 value for display with appropriate units."""
        if co2_kg >= 1000:
            return f"{co2_kg / 1000:.2f} tonnes CO₂"
        else:
            return f"{co2_kg:.1f} kg CO₂"


def calculate_carbon_footprint(df: pd.DataFrame) -> Dict:
    """
    Wrapper function for dashboard to calculate carbon footprint.
    
    Args:
        df: Energy consumption DataFrame
        
    Returns:
        Dictionary with all carbon metrics
    """
    calc = CarbonCalculator()
    
    # Total consumption
    total_kwh = df['consumption_kwh'].sum()
    
    # Monthly breakdown
    monthly_emissions = calc.calculate_monthly_emissions(df)
    
    # Current month summary
    last_month_kwh = monthly_emissions['total_kwh'].iloc[-1] if len(monthly_emissions) > 0 else 0
    summary = calc.get_emission_summary(last_month_kwh, period='month')
    
    # Yearly projection
    yearly_kwh = total_kwh
    yearly_summary = calc.get_emission_summary(yearly_kwh, period='year')
    
    # Reduction impact scenarios
    reduction_scenarios = []
    for percent in [10, 20, 30]:
        impact = calc.calculate_reduction_impact(last_month_kwh, percent)
        reduction_scenarios.append(impact)
    
    return {
        'summary': summary,
        'yearly': yearly_summary,
        'monthly_breakdown': monthly_emissions.to_dict('records'),
        'reduction_scenarios': reduction_scenarios,
        'calculator': calc
    }


if __name__ == "__main__":
    # Example usage
    calc = CarbonCalculator()
    
    # Monthly consumption example
    monthly_kwh = 350
    
    print("="*60)
    print("CARBON FOOTPRINT ANALYSIS")
    print("="*60)
    
    summary = calc.get_emission_summary(monthly_kwh, period='month')
    
    print(f"\nMonthly Consumption: {summary['consumption_kwh']} kWh")
    print(f"Carbon Emissions: {summary['co2_kg']} kg CO₂")
    print(f"Emission Factor: {summary['emission_factor']} kg CO₂/kWh (India grid)")
    
    print(f"\n📊 Equivalents:")
    for key, value in summary['equivalents'].items():
        if 'description' in key:
            print(f"  • {value}")
    
    benchmark = summary['benchmark_comparison']
    print(f"\n📈 Benchmark Comparison:")
    print(f"  Indian Household Average: {benchmark['benchmark_kg']} kg CO₂/month")
    print(f"  Your Status: {benchmark['status'].upper()} ({benchmark['vs_benchmark_percent']:+.1f}% vs average)")
    
    # Reduction impact
    print(f"\n🌱 If you reduce consumption by 20%:")
    impact = calc.calculate_reduction_impact(monthly_kwh, 20)
    print(f"  CO₂ Saved: {impact['co2_saved_kg']} kg/month")
    print(f"  Annual Equivalent: {impact['trees_equivalent']} trees planted")

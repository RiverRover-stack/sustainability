"""
Recommendation Engine Module

File Responsibility:
    Generates personalized energy optimization recommendations
    based on consumption analysis.

Inputs:
    - DataFrame with energy consumption data

Outputs:
    - List of prioritized recommendations with action items

Assumptions:
    - Data has standard columns (timestamp, consumption_kwh, hour, etc.)
    - Recommendations are tailored for Indian context

Failure Modes:
    - Missing columns may skip certain recommendations
    - Empty DataFrame returns empty recommendations list
"""

import pandas as pd
from typing import Dict, List, Optional

try:
    from pattern_analyzer import (
        analyze_consumption_pattern, identify_high_consumption_periods, PEAK_HOURS
    )
    from savings_calculator import (
        get_quick_tips, calculate_savings_potential, APPLIANCE_RATINGS, EFFICIENT_ALTERNATIVES
    )
except ImportError:
    from recommender.pattern_analyzer import (
        analyze_consumption_pattern, identify_high_consumption_periods, PEAK_HOURS
    )
    from recommender.savings_calculator import (
        get_quick_tips, calculate_savings_potential, APPLIANCE_RATINGS, EFFICIENT_ALTERNATIVES
    )


class EnergyRecommender:
    """
    AI-powered energy optimization recommendation engine.
    
    Purpose: Generate personalized energy-saving recommendations.
    """
    
    def __init__(self):
        self.peak_hours = PEAK_HOURS
        self.appliance_ratings = APPLIANCE_RATINGS
        self.efficient_alternatives = EFFICIENT_ALTERNATIVES
    
    def analyze_consumption_pattern(self, df: pd.DataFrame) -> Dict:
        """Delegate to pattern_analyzer module."""
        return analyze_consumption_pattern(df)
    
    def identify_high_consumption_periods(
        self, 
        df: pd.DataFrame, 
        threshold_percentile: float = 90
    ) -> List[Dict]:
        """Delegate to pattern_analyzer module."""
        return identify_high_consumption_periods(df, threshold_percentile)
    
    def generate_recommendations(
        self, 
        df: pd.DataFrame, 
        preferences: Optional[Dict] = None
    ) -> List[Dict]:
        """Generate personalized energy optimization recommendations."""
        recommendations = []
        analysis = self.analyze_consumption_pattern(df)
        
        # Peak hour recommendation
        if analysis['peak_ratio'] > 1.3:
            recommendations.append(self._peak_usage_recommendation(analysis))
        
        # AC optimization
        ac_ratio = self._calculate_ac_ratio(df)
        if ac_ratio > 0.3:
            recommendations.append(self._ac_optimization_recommendation())
        
        # Weekend optimization
        if analysis['weekend_avg'] > analysis['weekday_avg'] * 1.3:
            recommendations.append(self._weekend_recommendation(analysis))
        
        # Evening lighting
        if self._has_high_evening_lighting(df):
            recommendations.append(self._lighting_recommendation())
        
        # Solar recommendation
        if analysis['daily_avg'] > 5:
            recommendations.append(self._solar_recommendation(analysis))
        
        # Standby power
        night_consumption = df[df['hour'].isin([1, 2, 3, 4])]['consumption_kwh'].mean()
        if night_consumption > 0.3:
            recommendations.append(self._standby_recommendation(night_consumption))
        
        # Seasonal recommendation
        if self._has_high_summer_consumption(df):
            recommendations.append(self._seasonal_recommendation())
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def _calculate_ac_ratio(self, df: pd.DataFrame) -> float:
        """Calculate AC contribution to total consumption."""
        if 'ac_running' not in df.columns:
            return 0.0
        ac_hours = df[df['ac_running'] == 1]['consumption_kwh'].sum()
        total = df['consumption_kwh'].sum()
        return ac_hours / total if total > 0 else 0.0
    
    def _has_high_evening_lighting(self, df: pd.DataFrame) -> bool:
        """Check if evening lighting usage is high."""
        if 'lighting' not in df.columns:
            return False
        evening_df = df[df['hour'].isin([18, 19, 20, 21, 22])]
        return evening_df['lighting'].mean() > 0.8
    
    def _has_high_summer_consumption(self, df: pd.DataFrame) -> bool:
        """Check if summer consumption is significantly higher."""
        hot = df[df['month'].isin([4, 5, 6])]['consumption_kwh'].mean()
        cold = df[df['month'].isin([12, 1, 2])]['consumption_kwh'].mean()
        return hot > cold * 1.5 if cold > 0 else False
    
    def _peak_usage_recommendation(self, analysis: Dict) -> Dict:
        return {
            'category': 'Peak Usage',
            'priority': 'high',
            'title': 'Shift Heavy Appliances to Off-Peak Hours',
            'description': f"Peak consumption is {analysis['peak_ratio']:.1f}x off-peak. "
                          "Run heavy appliances during 11 PM - 6 AM.",
            'potential_savings': f"Up to ₹{int(analysis['daily_avg'] * 30 * 0.15)} per month",
            'action_items': [
                'Schedule washing machine for early morning or late night',
                'Use timer features on appliances',
                'Avoid running AC on maximum during peak hours'
            ]
        }
    
    def _ac_optimization_recommendation(self) -> Dict:
        return {
            'category': 'Air Conditioning',
            'priority': 'high',
            'title': 'Optimize AC Usage',
            'description': 'AC accounts for over 30% of energy consumption.',
            'potential_savings': 'Up to 25% reduction in cooling costs',
            'action_items': [
                'Set AC to 24-26°C (each degree lower = 6% more consumption)',
                'Use ceiling fans with AC to distribute cool air',
                'Clean AC filters monthly',
                'Consider upgrading to 5-star inverter AC'
            ]
        }
    
    def _weekend_recommendation(self, analysis: Dict) -> Dict:
        savings = int((analysis['weekend_avg'] - analysis['weekday_avg']) * 8 * 4 * 5)
        return {
            'category': 'Weekend Usage',
            'priority': 'medium',
            'title': 'Weekend Consumption is High',
            'description': f"Weekend avg ({analysis['weekend_avg']:.2f} kWh/hr) > weekday.",
            'potential_savings': f"₹{savings} per month",
            'action_items': [
                'Turn off gaming consoles and TVs when not in use',
                'Plan cooking to reduce appliance cycles'
            ]
        }
    
    def _lighting_recommendation(self) -> Dict:
        return {
            'category': 'Lighting',
            'priority': 'medium',
            'title': 'Switch to LED Lighting',
            'description': 'High evening lighting usage detected.',
            'potential_savings': '80% reduction in lighting costs',
            'action_items': [
                'Replace incandescent with LED bulbs',
                'Install motion sensors in low-traffic areas',
                'Maximize natural daylight'
            ]
        }
    
    def _solar_recommendation(self, analysis: Dict) -> Dict:
        capacity = max(2, int(analysis['daily_avg'] / 4))
        return {
            'category': 'Renewable Energy',
            'priority': 'medium',
            'title': 'Consider Solar Panel Installation',
            'description': f"Daily avg of {analysis['daily_avg']:.1f} kWh suits solar.",
            'potential_savings': '50-70% reduction in annual costs',
            'action_items': [
                'Get quotes from solar installers',
                'Check PM Surya Ghar Yojana subsidy (up to ₹78,000)',
                'Consider net metering'
            ],
            'solar_estimate': {
                'recommended_capacity': f"{capacity} kW",
                'estimated_cost': f"₹{capacity * 50000:,}",
                'payback_period': '4-6 years'
            }
        }
    
    def _standby_recommendation(self, night_consumption: float) -> Dict:
        return {
            'category': 'Standby Power',
            'priority': 'low',
            'title': 'Reduce Phantom/Standby Loads',
            'description': f"Night base load of {night_consumption:.2f} kWh suggests standby.",
            'potential_savings': '₹200-500 per month',
            'action_items': [
                'Use power strips with switches',
                'Unplug chargers when not in use',
                'Turn off entertainment systems completely'
            ]
        }
    
    def _seasonal_recommendation(self) -> Dict:
        return {
            'category': 'Seasonal',
            'priority': 'medium',
            'title': 'Summer Cooling Optimization',
            'description': 'Summer consumption is significantly higher.',
            'potential_savings': '15-25% reduction in summer bills',
            'action_items': [
                'Use curtains/blinds to block sunlight',
                'Improve home insulation',
                'Service AC before summer'
            ]
        }
    
    def get_quick_tips(self) -> List[str]:
        """Return quick energy-saving tips."""
        return get_quick_tips()
    
    def calculate_savings_potential(self, current_monthly_kwh: float) -> Dict:
        """Calculate potential savings from interventions."""
        return calculate_savings_potential(current_monthly_kwh)


def get_recommendations_for_dashboard(df: pd.DataFrame) -> Dict:
    """Wrapper for dashboard integration."""
    recommender = EnergyRecommender()
    return {
        'recommendations': recommender.generate_recommendations(df),
        'quick_tips': recommender.get_quick_tips(),
        'analysis': recommender.analyze_consumption_pattern(df)
    }


if __name__ == "__main__":
    import os
    from data.preprocessing import load_data
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "energy_data.csv")
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        recommender = EnergyRecommender()
        
        print("=" * 50)
        print("RECOMMENDER TEST")
        print("=" * 50)
        
        recs = recommender.generate_recommendations(df)
        print(f"\nGenerated {len(recs)} recommendations:")
        for rec in recs[:3]:
            print(f"  [{rec['priority']}] {rec['title']}")
        
        print("\n✓ Recommender working correctly!")
    else:
        print(f"Data file not found: {data_path}")

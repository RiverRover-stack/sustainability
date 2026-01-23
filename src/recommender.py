"""
Smart AI Energy Consumption Predictor
Recommendation Engine Module

Provides AI-powered optimization recommendations based on energy usage patterns.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime


class EnergyRecommender:
    """AI-powered energy optimization recommendation engine."""
    
    def __init__(self):
        # Peak hours (6-10 AM and 6-10 PM)
        self.peak_hours = list(range(6, 10)) + list(range(18, 22))
        
        # Off-peak hours (11 PM - 6 AM)
        self.off_peak_hours = list(range(23, 24)) + list(range(0, 6))
        
        # Appliance energy ratings (kWh per hour of use)
        self.appliance_ratings = {
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
        
        # Efficient alternatives
        self.efficient_alternatives = {
            'ac': ('Inverter AC', 0.8, '5-star rated inverter AC can save up to 40% energy'),
            'heater': ('Solar Water Heater', 0.0, 'Solar heater can eliminate electricity usage'),
            'lighting': ('LED Bulbs', 0.02, 'LED bulbs use 80% less energy than incandescent'),
            'refrigerator': ('5-Star Refrigerator', 0.08, 'Energy-efficient refrigerators save 30-40%'),
            'fan': ('BLDC Fan', 0.03, 'BLDC fans consume 60% less electricity')
        }
    
    def analyze_consumption_pattern(self, df: pd.DataFrame) -> Dict:
        """Analyze overall consumption patterns."""
        analysis = {}
        
        # Daily average consumption
        daily_consumption = df.groupby(df['timestamp'].dt.date)['consumption_kwh'].sum()
        analysis['daily_avg'] = daily_consumption.mean()
        analysis['daily_max'] = daily_consumption.max()
        analysis['daily_min'] = daily_consumption.min()
        
        # Monthly consumption
        monthly = df.groupby(df['timestamp'].dt.month)['consumption_kwh'].sum()
        analysis['monthly_avg'] = monthly.mean()
        analysis['highest_month'] = monthly.idxmax()
        
        # Peak vs off-peak consumption
        df_temp = df.copy()
        df_temp['is_peak'] = df_temp['hour'].isin(self.peak_hours)
        analysis['peak_consumption'] = df_temp[df_temp['is_peak']]['consumption_kwh'].mean()
        analysis['offpeak_consumption'] = df_temp[~df_temp['is_peak']]['consumption_kwh'].mean()
        analysis['peak_ratio'] = analysis['peak_consumption'] / analysis['offpeak_consumption']
        
        # Weekend vs weekday
        analysis['weekend_avg'] = df[df['is_weekend'] == 1]['consumption_kwh'].mean()
        analysis['weekday_avg'] = df[df['is_weekend'] == 0]['consumption_kwh'].mean()
        
        return analysis
    
    def identify_high_consumption_periods(self, df: pd.DataFrame, threshold_percentile: float = 90) -> List[Dict]:
        """Identify periods of unusually high consumption."""
        threshold = df['consumption_kwh'].quantile(threshold_percentile / 100)
        high_periods = df[df['consumption_kwh'] > threshold].copy()
        
        periods = []
        for _, row in high_periods.iterrows():
            periods.append({
                'timestamp': row['timestamp'],
                'consumption': row['consumption_kwh'],
                'temperature': row.get('temperature', None),
                'ac_running': row.get('ac_running', 0),
                'hour': row['hour'],
                'is_weekend': row['is_weekend']
            })
        
        return periods
    
    def generate_recommendations(self, df: pd.DataFrame, preferences: Optional[Dict] = None) -> List[Dict]:
        """Generate personalized energy optimization recommendations."""
        recommendations = []
        analysis = self.analyze_consumption_pattern(df)
        high_periods = self.identify_high_consumption_periods(df)
        
        # 1. Peak hour recommendations
        if analysis['peak_ratio'] > 1.3:
            recommendations.append({
                'category': 'Peak Usage',
                'priority': 'high',
                'title': 'Shift Heavy Appliances to Off-Peak Hours',
                'description': f"Your peak hour consumption is {analysis['peak_ratio']:.1f}x higher than off-peak. "
                              f"Consider running washing machines, dishwashers, and water heaters during "
                              f"off-peak hours (11 PM - 6 AM) to reduce bills.",
                'potential_savings': f"Up to ₹{int(analysis['daily_avg'] * 30 * 0.15)} per month",
                'action_items': [
                    'Schedule washing machine for early morning or late night',
                    'Use timer features on appliances',
                    'Avoid running AC on maximum during peak hours'
                ]
            })
        
        # 2. AC optimization
        ac_hours = df[df['ac_running'] == 1]['consumption_kwh'].sum()
        total_consumption = df['consumption_kwh'].sum()
        if ac_hours / total_consumption > 0.3:
            recommendations.append({
                'category': 'Air Conditioning',
                'priority': 'high',
                'title': 'Optimize AC Usage',
                'description': 'Air conditioning accounts for over 30% of your energy consumption.',
                'potential_savings': 'Up to 25% reduction in cooling costs',
                'action_items': [
                    'Set AC temperature to 24-26°C (each degree lower increases consumption by 6%)',
                    'Use ceiling fans along with AC to distribute cool air',
                    'Ensure windows and doors are sealed properly',
                    'Clean AC filters monthly for optimal efficiency',
                    'Consider upgrading to 5-star inverter AC'
                ]
            })
        
        # 3. Weekend optimization
        if analysis['weekend_avg'] > analysis['weekday_avg'] * 1.3:
            recommendations.append({
                'category': 'Weekend Usage',
                'priority': 'medium',
                'title': 'Weekend Consumption is High',
                'description': f"Weekend consumption ({analysis['weekend_avg']:.2f} kWh/hr) is significantly "
                              f"higher than weekdays ({analysis['weekday_avg']:.2f} kWh/hr).",
                'potential_savings': f"₹{int((analysis['weekend_avg'] - analysis['weekday_avg']) * 8 * 4 * 5)} per month",
                'action_items': [
                    'Avoid leaving TVs and computers on standby',
                    'Plan cooking activities to reduce repeat use of appliances',
                    'Consider outdoor activities to reduce indoor energy use'
                ]
            })
        
        # 4. Evening lighting
        evening_df = df[df['hour'].isin([18, 19, 20, 21, 22])]
        if evening_df['lighting'].mean() > 0.8:
            recommendations.append({
                'category': 'Lighting',
                'priority': 'medium',
                'title': 'Switch to LED Lighting',
                'description': 'High evening lighting usage detected.',
                'potential_savings': '80% reduction in lighting electricity costs',
                'action_items': [
                    'Replace incandescent bulbs with LED bulbs',
                    'Install motion sensors in low-traffic areas',
                    'Maximize natural daylight during daytime',
                    'Use task lighting instead of room-wide lighting'
                ]
            })
        
        # 5. Solar adoption recommendation
        if analysis['daily_avg'] > 5:  # Average > 5 kWh/day
            recommendations.append({
                'category': 'Renewable Energy',
                'priority': 'medium',
                'title': 'Consider Solar Panel Installation',
                'description': f"With an average daily consumption of {analysis['daily_avg']:.1f} kWh, "
                              f"a rooftop solar system could significantly reduce your electricity bills.",
                'potential_savings': 'Up to 50-70% reduction in annual electricity costs',
                'action_items': [
                    'Contact local solar installation providers for assessment',
                    'Check government subsidies (PM Surya Ghar Yojana offers up to ₹78,000)',
                    'Consider net metering to sell excess power back to grid',
                    'Start with a 3-5 kW system for typical households'
                ],
                'solar_estimate': {
                    'recommended_capacity': f"{max(2, int(analysis['daily_avg'] / 4))} kW",
                    'estimated_cost': f"₹{max(2, int(analysis['daily_avg'] / 4)) * 50000:,}",
                    'payback_period': '4-6 years'
                }
            })
        
        # 6. Standby power
        night_consumption = df[df['hour'].isin([1, 2, 3, 4])]['consumption_kwh'].mean()
        if night_consumption > 0.3:
            recommendations.append({
                'category': 'Standby Power',
                'priority': 'low',
                'title': 'Reduce Phantom/Standby Loads',
                'description': f"Night-time base load of {night_consumption:.2f} kWh suggests appliances on standby.",
                'potential_savings': '₹200-500 per month',
                'action_items': [
                    'Use power strips with switches for electronics',
                    'Unplug chargers when not in use',
                    'Turn off entertainment systems completely',
                    'Check for old appliances with high standby consumption'
                ]
            })
        
        # 7. Seasonal recommendations
        hot_months = df[df['month'].isin([4, 5, 6])]['consumption_kwh'].mean()
        cold_months = df[df['month'].isin([12, 1, 2])]['consumption_kwh'].mean()
        
        if hot_months > cold_months * 1.5:
            recommendations.append({
                'category': 'Seasonal',
                'priority': 'medium',
                'title': 'Summer Cooling Optimization',
                'description': 'Your summer consumption is significantly higher due to cooling needs.',
                'potential_savings': '15-25% reduction in summer bills',
                'action_items': [
                    'Use curtains/blinds to block direct sunlight',
                    'Improve home insulation',
                    'Service AC before summer starts',
                    'Consider evaporative coolers for dry climates'
                ]
            })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        recommendations.sort(key=lambda x: priority_order[x['priority']])
        
        return recommendations
    
    def get_quick_tips(self) -> List[str]:
        """Return general quick energy-saving tips."""
        return [
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
    
    def calculate_savings_potential(self, current_monthly_kwh: float) -> Dict:
        """Calculate potential savings from various interventions."""
        current_cost = self._estimate_bill(current_monthly_kwh)
        
        scenarios = {
            'led_lighting': {
                'action': 'Switch to LED lighting',
                'savings_percent': 0.10,
                'implementation_cost': 2000,
                'payback_months': 0
            },
            'efficient_ac': {
                'action': 'Upgrade to 5-star inverter AC',
                'savings_percent': 0.20,
                'implementation_cost': 40000,
                'payback_months': 0
            },
            'solar_3kw': {
                'action': 'Install 3kW solar system',
                'savings_percent': 0.50,
                'implementation_cost': 150000,
                'payback_months': 0
            },
            'behavioral_changes': {
                'action': 'Behavioral changes (no cost)',
                'savings_percent': 0.15,
                'implementation_cost': 0,
                'payback_months': 0
            }
        }
        
        for key, scenario in scenarios.items():
            monthly_savings = current_cost * scenario['savings_percent']
            scenario['monthly_savings'] = round(monthly_savings, 2)
            scenario['annual_savings'] = round(monthly_savings * 12, 2)
            if scenario['implementation_cost'] > 0:
                scenario['payback_months'] = round(scenario['implementation_cost'] / monthly_savings, 1)
        
        return scenarios
    
    def _estimate_bill(self, kwh: float) -> float:
        """Simple bill estimation based on average tariff."""
        avg_rate = 5.5  # Average ₹ per kWh
        return kwh * avg_rate


def get_recommendations_for_dashboard(df: pd.DataFrame) -> Dict:
    """Wrapper function to get recommendations formatted for dashboard."""
    recommender = EnergyRecommender()
    
    return {
        'recommendations': recommender.generate_recommendations(df),
        'quick_tips': recommender.get_quick_tips(),
        'analysis': recommender.analyze_consumption_pattern(df)
    }


if __name__ == "__main__":
    import os
    from preprocessing import load_data
    
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "energy_data.csv")
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        recommender = EnergyRecommender()
        
        print("="*60)
        print("ENERGY CONSUMPTION ANALYSIS")
        print("="*60)
        
        analysis = recommender.analyze_consumption_pattern(df)
        print(f"\nDaily Average: {analysis['daily_avg']:.2f} kWh")
        print(f"Peak Hour Consumption: {analysis['peak_consumption']:.2f} kWh/hr")
        print(f"Off-Peak Consumption: {analysis['offpeak_consumption']:.2f} kWh/hr")
        print(f"Peak Ratio: {analysis['peak_ratio']:.2f}x")
        
        print("\n" + "="*60)
        print("RECOMMENDATIONS")
        print("="*60)
        
        recommendations = recommender.generate_recommendations(df)
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. [{rec['priority'].upper()}] {rec['title']}")
            print(f"   Category: {rec['category']}")
            print(f"   {rec['description']}")
            print(f"   Potential Savings: {rec['potential_savings']}")
    else:
        print(f"Data file not found: {data_path}")

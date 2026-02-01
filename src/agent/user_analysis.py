"""
User Consumption Analysis Module

File Responsibility:
    Computes summary statistics and insights from user's
    consumption DataFrame for personalized advice.

Inputs:
    - DataFrame with consumption data

Outputs:
    - Summary statistics dictionary
    - Insights and recommendations list

Assumptions:
    - Data has standard columns (timestamp, consumption_kwh, hour, is_weekend)
    - Data spans at least a few days

Failure Modes:
    - Missing columns raise KeyError
    - Empty DataFrame returns None
"""

import pandas as pd
from typing import Dict, List, Optional


def compute_user_summary(df: pd.DataFrame) -> Optional[Dict]:
    """
    Compute summary statistics from user's consumption data.
    
    Purpose: Extract key metrics for personalized responses.
    
    Inputs:
        df: DataFrame with consumption data
        
    Outputs:
        Dictionary with aggregate metrics or None if empty
        
    Side effects: None
    """
    if df is None or len(df) == 0:
        return None
    
    # Calculate key metrics
    total_kwh = df['consumption_kwh'].sum()
    daily_avg = df.groupby(df['timestamp'].dt.date)['consumption_kwh'].sum().mean()
    
    # Monthly breakdown
    monthly = df.groupby(df['timestamp'].dt.month)['consumption_kwh'].sum()
    highest_month = monthly.idxmax()
    lowest_month = monthly.idxmin()
    
    # Peak hours analysis
    hourly = df.groupby('hour')['consumption_kwh'].mean()
    peak_hour = hourly.idxmax()
    
    # Weekend vs weekday
    weekend_avg = df[df['is_weekend'] == 1]['consumption_kwh'].mean()
    weekday_avg = df[df['is_weekend'] == 0]['consumption_kwh'].mean()
    
    # Carbon footprint (India grid factor)
    carbon_kg = total_kwh * 0.82
    
    return {
        'total_kwh': round(total_kwh, 2),
        'daily_avg_kwh': round(daily_avg, 2),
        'monthly_avg_kwh': round(total_kwh / 12, 2),
        'highest_month': highest_month,
        'lowest_month': lowest_month,
        'peak_hour': peak_hour,
        'weekend_avg': round(weekend_avg, 3),
        'weekday_avg': round(weekday_avg, 3),
        'carbon_kg': round(carbon_kg, 2),
        'data_days': len(df['timestamp'].dt.date.unique())
    }


def format_user_context(summary: Optional[Dict]) -> str:
    """
    Format user summary as context string for LLM prompts.
    
    Purpose: Create readable context for AI prompts.
    
    Inputs:
        summary: Summary dictionary from compute_user_summary
        
    Outputs:
        Formatted context string
        
    Side effects: None
    """
    if summary is None:
        return "No user consumption data available."
    
    s = summary
    return f"""User's Energy Consumption Summary:
- Total consumption: {s['total_kwh']} kWh over {s['data_days']} days
- Daily average: {s['daily_avg_kwh']} kWh
- Monthly average: {s['monthly_avg_kwh']} kWh
- Highest consumption month: Month {s['highest_month']}
- Peak usage hour: {s['peak_hour']}:00
- Weekend average: {s['weekend_avg']} kWh/hr (Weekday: {s['weekday_avg']} kWh/hr)
- Annual carbon footprint: {s['carbon_kg']} kg CO₂"""


def generate_insights(summary: Dict) -> Dict:
    """
    Generate insights and recommendations from user summary.
    
    Purpose: Autonomous analysis of consumption patterns.
    
    Inputs:
        summary: Summary dictionary
        
    Outputs:
        Dictionary with insights and recommendations lists
        
    Side effects: None
    """
    if summary is None:
        return {'insights': [], 'recommendations': []}
    
    s = summary
    insights = []
    recommendations = []
    
    # Weekend vs weekday pattern
    if s['weekend_avg'] > s['weekday_avg'] * 1.3:
        insights.append({
            'type': 'pattern',
            'finding': 'Weekend consumption is significantly higher than weekdays',
            'detail': f"Weekend: {s['weekend_avg']:.3f} kWh/hr vs Weekday: {s['weekday_avg']:.3f} kWh/hr"
        })
        recommendations.append("Consider energy-conscious activities on weekends")
    
    # Evening peak
    if s['peak_hour'] in [18, 19, 20, 21]:
        insights.append({
            'type': 'peak',
            'finding': f"Peak consumption occurs in the evening at {s['peak_hour']}:00",
            'detail': "Evening peak hours often have higher tariff rates"
        })
        recommendations.append("Shift some activities to off-peak hours (post 11 PM)")
    
    # High daily usage
    if s['daily_avg_kwh'] > 15:
        insights.append({
            'type': 'high_usage',
            'finding': 'Daily consumption is above typical household average',
            'detail': f"Your daily average: {s['daily_avg_kwh']} kWh (typical: 10-15 kWh)"
        })
        recommendations.append("Consider energy audit to identify high-consumption appliances")
    
    # High carbon footprint
    if s['carbon_kg'] > 2000:
        insights.append({
            'type': 'carbon',
            'finding': 'Annual carbon footprint exceeds 2 tonnes CO₂',
            'detail': f"Your footprint: {s['carbon_kg']} kg CO₂"
        })
        recommendations.append("Consider solar installation to reduce carbon footprint")
    
    # Solar recommendation for high consumers
    if s['daily_avg_kwh'] > 10:
        recommended_solar = max(2, int(s['daily_avg_kwh'] / 4))
        recommendations.append(
            f"A {recommended_solar}kW solar system could offset most of your consumption"
        )
    
    return {'insights': insights, 'recommendations': recommendations}

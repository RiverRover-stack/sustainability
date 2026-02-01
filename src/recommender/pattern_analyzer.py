"""
Pattern Analyzer Module

File Responsibility:
    Analyzes consumption patterns to identify usage trends,
    peak hours, and high consumption periods.

Inputs:
    - DataFrame with timestamp, consumption_kwh, hour, is_weekend columns

Outputs:
    - Dictionary with consumption pattern analysis
    - List of high consumption periods

Assumptions:
    - Data is hourly granularity
    - Peak hours are 6-10 AM and 6-10 PM
    - Off-peak hours are 11 PM - 6 AM

Failure Modes:
    - Missing columns will raise KeyError
    - Empty DataFrame returns empty/zero values
"""

import pandas as pd
from typing import Dict, List


# Peak and off-peak hour definitions
PEAK_HOURS = list(range(6, 10)) + list(range(18, 22))
OFF_PEAK_HOURS = list(range(23, 24)) + list(range(0, 6))


def analyze_consumption_pattern(df: pd.DataFrame) -> Dict:
    """
    Analyze overall consumption patterns.
    
    Purpose: Extract key metrics from consumption data.
    
    Inputs:
        df: DataFrame with timestamp, consumption_kwh, hour, is_weekend
        
    Outputs:
        Dictionary with daily, monthly, peak/off-peak, and weekend metrics
        
    Side effects: None
    """
    analysis = {}
    
    # Daily consumption statistics
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
    df_temp['is_peak'] = df_temp['hour'].isin(PEAK_HOURS)
    analysis['peak_consumption'] = df_temp[df_temp['is_peak']]['consumption_kwh'].mean()
    analysis['offpeak_consumption'] = df_temp[~df_temp['is_peak']]['consumption_kwh'].mean()
    
    # Avoid division by zero
    if analysis['offpeak_consumption'] > 0:
        analysis['peak_ratio'] = analysis['peak_consumption'] / analysis['offpeak_consumption']
    else:
        analysis['peak_ratio'] = 1.0
    
    # Weekend vs weekday
    analysis['weekend_avg'] = df[df['is_weekend'] == 1]['consumption_kwh'].mean()
    analysis['weekday_avg'] = df[df['is_weekend'] == 0]['consumption_kwh'].mean()
    
    return analysis


def identify_high_consumption_periods(
    df: pd.DataFrame, 
    threshold_percentile: float = 90
) -> List[Dict]:
    """
    Identify periods of unusually high consumption.
    
    Purpose: Flag anomalous consumption for targeted recommendations.
    
    Inputs:
        df: DataFrame with consumption data
        threshold_percentile: Percentile above which consumption is "high"
        
    Outputs:
        List of high consumption period records
        
    Side effects: None
    """
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


def get_seasonal_consumption(df: pd.DataFrame) -> Dict:
    """
    Get seasonal consumption breakdown.
    
    Purpose: Identify seasonal patterns for recommendations.
    
    Inputs:
        df: DataFrame with month and consumption columns
        
    Outputs:
        Dictionary with seasonal averages
        
    Side effects: None
    """
    seasons = {
        'summer': [4, 5, 6],
        'monsoon': [7, 8, 9],
        'post_monsoon': [10, 11],
        'winter': [12, 1, 2, 3]
    }
    
    result = {}
    for season, months in seasons.items():
        season_df = df[df['month'].isin(months)]
        result[season] = season_df['consumption_kwh'].mean() if len(season_df) > 0 else 0
    
    return result

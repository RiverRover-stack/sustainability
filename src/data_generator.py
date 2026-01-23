"""
Smart AI Energy Consumption Predictor
Data Generator Module

Generates 1 year of synthetic smart meter data with realistic patterns
for training energy consumption prediction models.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def generate_data(start_date: str = "2025-01-01", days: int = 365, save_path: str = None) -> pd.DataFrame:
    """
    Generate synthetic energy consumption data with weather and occupancy features.
    
    Args:
        start_date: Start date for data generation (YYYY-MM-DD)
        days: Number of days to generate
        save_path: Optional path to save CSV file
        
    Returns:
        DataFrame with hourly energy consumption data
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate hourly timestamps
    start = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [start + timedelta(hours=i) for i in range(days * 24)]
    
    data = []
    
    for ts in timestamps:
        hour = ts.hour
        day_of_week = ts.weekday()
        month = ts.month
        is_weekend = 1 if day_of_week >= 5 else 0
        
        # Seasonal temperature patterns (India climate)
        if month in [3, 4, 5]:  # Summer
            base_temp = 35 + np.random.normal(0, 3)
        elif month in [6, 7, 8, 9]:  # Monsoon
            base_temp = 28 + np.random.normal(0, 2)
        elif month in [10, 11]:  # Post-monsoon
            base_temp = 25 + np.random.normal(0, 2)
        else:  # Winter
            base_temp = 18 + np.random.normal(0, 3)
        
        # Daily temperature variation
        temp_variation = 5 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else -3
        temperature = base_temp + temp_variation + np.random.normal(0, 1)
        
        # Humidity patterns
        if month in [6, 7, 8, 9]:  # Monsoon high humidity
            humidity = 75 + np.random.normal(0, 10)
        else:
            humidity = 50 + np.random.normal(0, 15)
        humidity = np.clip(humidity, 20, 100)
        
        # Occupancy patterns (1 = occupied, 0.5 = partial, 0 = empty)
        if is_weekend:
            if 8 <= hour <= 22:
                occupancy = 0.9 + np.random.normal(0, 0.1)
            else:
                occupancy = 0.3 + np.random.normal(0, 0.1)
        else:
            if 8 <= hour <= 18:  # Work/school hours
                occupancy = 0.2 + np.random.normal(0, 0.1)
            elif 18 <= hour <= 23:  # Evening at home
                occupancy = 0.9 + np.random.normal(0, 0.1)
            else:
                occupancy = 0.3 + np.random.normal(0, 0.1)
        occupancy = np.clip(occupancy, 0, 1)
        
        # Base consumption (kWh)
        base_consumption = 0.5
        
        # Time-of-day patterns
        if 6 <= hour <= 9:  # Morning peak
            time_factor = 1.5
        elif 18 <= hour <= 22:  # Evening peak
            time_factor = 2.0
        elif 0 <= hour <= 5:  # Night low
            time_factor = 0.4
        else:
            time_factor = 1.0
        
        # Weather impact (AC/Heating)
        if temperature > 30:
            weather_factor = 1 + (temperature - 30) * 0.1  # AC usage
        elif temperature < 15:
            weather_factor = 1 + (15 - temperature) * 0.08  # Heating
        else:
            weather_factor = 1.0
        
        # Occupancy impact
        occupancy_factor = 0.5 + occupancy * 1.0
        
        # Weekend factor
        weekend_factor = 1.2 if is_weekend else 1.0
        
        # Calculate consumption
        consumption = (base_consumption * time_factor * weather_factor * 
                      occupancy_factor * weekend_factor)
        consumption += np.random.normal(0, 0.1)  # Random noise
        consumption = max(0.1, consumption)  # Minimum consumption
        
        # Appliance-specific simulation
        appliance_usage = {
            'ac_running': 1 if temperature > 28 and occupancy > 0.5 else 0,
            'lighting': 1 if (hour >= 18 or hour <= 6) and occupancy > 0.3 else 0,
            'cooking': 1 if hour in [7, 8, 12, 13, 19, 20] else 0,
            'entertainment': 1 if (18 <= hour <= 23) and is_weekend else 0
        }
        
        data.append({
            'timestamp': ts,
            'consumption_kwh': round(consumption, 3),
            'temperature': round(temperature, 1),
            'humidity': round(humidity, 1),
            'occupancy': round(occupancy, 2),
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'is_weekend': is_weekend,
            'ac_running': appliance_usage['ac_running'],
            'lighting': appliance_usage['lighting'],
            'cooking': appliance_usage['cooking'],
            'entertainment': appliance_usage['entertainment']
        })
    
    df = pd.DataFrame(data)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    return df


def get_tariff_slabs():
    """
    Returns electricity tariff slabs (based on typical Indian domestic rates).
    """
    return [
        {'min_units': 0, 'max_units': 100, 'rate': 3.0},
        {'min_units': 101, 'max_units': 200, 'rate': 4.5},
        {'min_units': 201, 'max_units': 300, 'rate': 6.0},
        {'min_units': 301, 'max_units': 500, 'rate': 7.5},
        {'min_units': 501, 'max_units': float('inf'), 'rate': 8.5}
    ]


def calculate_bill(total_kwh: float, tariff_slabs: list = None) -> dict:
    """
    Calculate electricity bill based on consumption and tariff slabs.
    
    Args:
        total_kwh: Total energy consumption in kWh
        tariff_slabs: List of tariff dictionaries
        
    Returns:
        Dictionary with bill breakdown
    """
    if tariff_slabs is None:
        tariff_slabs = get_tariff_slabs()
    
    remaining = total_kwh
    total_bill = 0
    breakdown = []
    
    for slab in tariff_slabs:
        if remaining <= 0:
            break
            
        slab_range = slab['max_units'] - slab['min_units']
        if slab['max_units'] == float('inf'):
            units_in_slab = remaining
        else:
            units_in_slab = min(remaining, slab_range)
        
        slab_cost = units_in_slab * slab['rate']
        total_bill += slab_cost
        remaining -= units_in_slab
        
        breakdown.append({
            'slab': f"{slab['min_units']}-{slab['max_units']} units",
            'units': round(units_in_slab, 2),
            'rate': slab['rate'],
            'cost': round(slab_cost, 2)
        })
    
    return {
        'total_units': round(total_kwh, 2),
        'total_bill': round(total_bill, 2),
        'breakdown': breakdown
    }


if __name__ == "__main__":
    # Generate and save data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "energy_data.csv")
    df = generate_data(save_path=data_path)
    print(f"Generated {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"\nConsumption statistics:")
    print(df['consumption_kwh'].describe())

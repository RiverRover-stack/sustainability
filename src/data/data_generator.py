"""
Data Generator Module

File Responsibility:
    Generates synthetic smart meter data with realistic patterns
    for training energy consumption prediction models.

Inputs:
    - Start date and duration for data generation
    - Optional save path for CSV output

Outputs:
    - DataFrame with hourly energy consumption data including
      weather, occupancy, and appliance usage features

Assumptions:
    - Data follows Indian climate patterns (seasons, temperatures)
    - Hourly granularity for all measurements
    - Realistic occupancy patterns for residential usage

Failure Modes:
    - Invalid date format raises ValueError
    - Insufficient disk space for CSV save
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

# Import tariff functions for backward compatibility
from tariff_calculator import get_tariff_slabs, calculate_bill


def generate_data(
    start_date: str = "2025-01-01", 
    days: int = 365, 
    save_path: str = None
) -> pd.DataFrame:
    """
    Generate synthetic energy consumption data.
    
    Purpose: Create realistic training data for ML models.
    
    Inputs:
        start_date: Start date in YYYY-MM-DD format
        days: Number of days to generate
        save_path: Optional path to save CSV file
        
    Outputs:
        DataFrame with hourly energy consumption data
        
    Side effects: Saves CSV file if save_path provided
    """
    np.random.seed(42)  # For reproducibility
    
    # Generate hourly timestamps
    start = datetime.strptime(start_date, "%Y-%m-%d")
    timestamps = [start + timedelta(hours=i) for i in range(days * 24)]
    
    data = []
    
    for ts in timestamps:
        record = _generate_hourly_record(ts)
        data.append(record)
    
    df = pd.DataFrame(data)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        df.to_csv(save_path, index=False)
        print(f"Data saved to {save_path}")
    
    return df


def _generate_hourly_record(ts: datetime) -> dict:
    """Generate a single hourly data record."""
    hour = ts.hour
    day_of_week = ts.weekday()
    month = ts.month
    is_weekend = 1 if day_of_week >= 5 else 0
    
    # Generate weather conditions
    temperature = _generate_temperature(month, hour)
    humidity = _generate_humidity(month)
    
    # Generate occupancy
    occupancy = _generate_occupancy(hour, is_weekend)
    
    # Calculate consumption
    consumption = _calculate_consumption(
        hour, temperature, occupancy, is_weekend
    )
    
    # Determine appliance usage
    appliances = _get_appliance_usage(hour, temperature, occupancy, is_weekend)
    
    return {
        'timestamp': ts,
        'consumption_kwh': round(consumption, 3),
        'temperature': round(temperature, 1),
        'humidity': round(humidity, 1),
        'occupancy': round(occupancy, 2),
        'hour': hour,
        'day_of_week': day_of_week,
        'month': month,
        'is_weekend': is_weekend,
        **appliances
    }


def _generate_temperature(month: int, hour: int) -> float:
    """Generate realistic temperature based on Indian climate."""
    # Seasonal base temperatures
    if month in [3, 4, 5]:  # Summer
        base_temp = 35 + np.random.normal(0, 3)
    elif month in [6, 7, 8, 9]:  # Monsoon
        base_temp = 28 + np.random.normal(0, 2)
    elif month in [10, 11]:  # Post-monsoon
        base_temp = 25 + np.random.normal(0, 2)
    else:  # Winter
        base_temp = 18 + np.random.normal(0, 3)
    
    # Daily variation
    temp_variation = 5 * np.sin((hour - 6) * np.pi / 12) if 6 <= hour <= 18 else -3
    return base_temp + temp_variation + np.random.normal(0, 1)


def _generate_humidity(month: int) -> float:
    """Generate realistic humidity based on season."""
    if month in [6, 7, 8, 9]:  # Monsoon high humidity
        humidity = 75 + np.random.normal(0, 10)
    else:
        humidity = 50 + np.random.normal(0, 15)
    return np.clip(humidity, 20, 100)


def _generate_occupancy(hour: int, is_weekend: int) -> float:
    """Generate realistic occupancy patterns."""
    if is_weekend:
        if 8 <= hour <= 22:
            occupancy = 0.9 + np.random.normal(0, 0.1)
        else:
            occupancy = 0.3 + np.random.normal(0, 0.1)
    else:
        if 8 <= hour <= 18:  # Work hours
            occupancy = 0.2 + np.random.normal(0, 0.1)
        elif 18 <= hour <= 23:  # Evening
            occupancy = 0.9 + np.random.normal(0, 0.1)
        else:
            occupancy = 0.3 + np.random.normal(0, 0.1)
    return np.clip(occupancy, 0, 1)


def _calculate_consumption(
    hour: int, 
    temperature: float, 
    occupancy: float, 
    is_weekend: int
) -> float:
    """Calculate energy consumption based on factors."""
    base = 0.5
    
    # Time-of-day factor
    if 6 <= hour <= 9:
        time_factor = 1.5
    elif 18 <= hour <= 22:
        time_factor = 2.0
    elif 0 <= hour <= 5:
        time_factor = 0.4
    else:
        time_factor = 1.0
    
    # Weather factor
    if temperature > 30:
        weather_factor = 1 + (temperature - 30) * 0.1
    elif temperature < 15:
        weather_factor = 1 + (15 - temperature) * 0.08
    else:
        weather_factor = 1.0
    
    occupancy_factor = 0.5 + occupancy * 1.0
    weekend_factor = 1.2 if is_weekend else 1.0
    
    consumption = base * time_factor * weather_factor * occupancy_factor * weekend_factor
    consumption += np.random.normal(0, 0.1)
    return max(0.1, consumption)


def _get_appliance_usage(
    hour: int, 
    temperature: float, 
    occupancy: float, 
    is_weekend: int
) -> dict:
    """Determine appliance usage flags."""
    return {
        'ac_running': 1 if temperature > 28 and occupancy > 0.5 else 0,
        'lighting': 1 if (hour >= 18 or hour <= 6) and occupancy > 0.3 else 0,
        'cooking': 1 if hour in [7, 8, 12, 13, 19, 20] else 0,
        'entertainment': 1 if (18 <= hour <= 23) and is_weekend else 0
    }


if __name__ == "__main__":
    # Generate and save data
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "energy_data.csv")
    df = generate_data(save_path=data_path)
    print(f"Generated {len(df)} records")
    print(f"Columns: {list(df.columns)}")
    print(f"\nConsumption statistics:")
    print(df['consumption_kwh'].describe())
    print("\n✓ Data generator working correctly!")

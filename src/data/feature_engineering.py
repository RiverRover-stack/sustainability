"""
Feature Engineering Module

File Responsibility:
    Creates time-series features from energy consumption data including
    temporal features, lag features, rolling statistics, and difference features.

Inputs:
    - DataFrame with 'timestamp' and 'consumption_kwh' columns

Outputs:
    - DataFrame with additional engineered features

Assumptions:
    - Input DataFrame has a 'timestamp' column in datetime format
    - Data is sorted chronologically
    - Hourly granularity for time-series features

Failure Modes:
    - Missing 'timestamp' column will raise KeyError
    - Non-datetime timestamp will cause attribute errors
"""

import pandas as pd
import numpy as np
from typing import List


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract time-based features from timestamp.
    
    Purpose: Create temporal features for ML models including cyclical encoding.
    
    Inputs:
        df: DataFrame with 'timestamp' column
        
    Outputs:
        DataFrame with added time features (hour, day, month, season, cyclical encodings)
        
    Side effects: None (returns copy)
    """
    df = df.copy()
    
    # Basic time features (if not already present)
    if 'hour' not in df.columns:
        df['hour'] = df['timestamp'].dt.hour
    if 'day_of_week' not in df.columns:
        df['day_of_week'] = df['timestamp'].dt.dayofweek
    if 'month' not in df.columns:
        df['month'] = df['timestamp'].dt.month
    
    # Additional features
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['week_of_year'] = df['timestamp'].dt.isocalendar().week.astype(int)
    df['is_month_start'] = df['timestamp'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['timestamp'].dt.is_month_end.astype(int)
    
    # Season (India: Summer Mar-May, Monsoon Jun-Sep, Post-monsoon Oct-Nov, Winter Dec-Feb)
    season_map = {
        1: 'winter', 2: 'winter', 3: 'summer', 4: 'summer',
        5: 'summer', 6: 'monsoon', 7: 'monsoon', 8: 'monsoon',
        9: 'monsoon', 10: 'post_monsoon', 11: 'post_monsoon', 12: 'winter'
    }
    df['season'] = df['month'].map(season_map)
    
    # Time of day categories
    df['time_period'] = df['hour'].apply(_get_time_period)
    
    # Cyclical encoding for hour, month, and day
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def _get_time_period(hour: int) -> str:
    """Map hour to time period category."""
    if 0 <= hour < 6:
        return 'night'
    elif 6 <= hour < 12:
        return 'morning'
    elif 12 <= hour < 18:
        return 'afternoon'
    else:
        return 'evening'


def create_lag_features(
    df: pd.DataFrame, 
    target_col: str = 'consumption_kwh', 
    lags: List[int] = None
) -> pd.DataFrame:
    """
    Create lagged features for time series prediction.
    
    Purpose: Capture autocorrelation patterns in consumption data.
    
    Inputs:
        df: DataFrame with target column
        target_col: Name of column to create lags for
        lags: List of lag periods (default: [1, 2, 3, 6, 12, 24])
        
    Outputs:
        DataFrame with lag features added
        
    Side effects: None (returns copy)
    """
    if lags is None:
        lags = [1, 2, 3, 6, 12, 24]
    
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(
    df: pd.DataFrame, 
    target_col: str = 'consumption_kwh',
    windows: List[int] = None
) -> pd.DataFrame:
    """
    Create rolling statistics features.
    
    Purpose: Capture short-term trends and variability patterns.
    
    Inputs:
        df: DataFrame with target column
        target_col: Name of column to compute rolling stats for
        windows: List of window sizes (default: [3, 6, 12, 24])
        
    Outputs:
        DataFrame with rolling mean, std, max, min features
        
    Side effects: None (returns copy)
    """
    if windows is None:
        windows = [3, 6, 12, 24]
    
    df = df.copy()
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
    
    return df


def create_diff_features(
    df: pd.DataFrame, 
    target_col: str = 'consumption_kwh'
) -> pd.DataFrame:
    """
    Create difference features (rate of change).
    
    Purpose: Capture momentum and trend direction in consumption.
    
    Inputs:
        df: DataFrame with target column
        target_col: Name of column to compute differences for
        
    Outputs:
        DataFrame with hourly and daily difference features
        
    Side effects: None (returns copy)
    """
    df = df.copy()
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_24'] = df[target_col].diff(24)  # Daily change
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode categorical columns.
    
    Purpose: Convert season and time_period to ML-ready format.
    
    Inputs:
        df: DataFrame with categorical columns
        
    Outputs:
        DataFrame with one-hot encoded features
        
    Side effects: None (returns copy)
    """
    df = df.copy()
    categorical_cols = ['season', 'time_period']
    existing_cats = [col for col in categorical_cols if col in df.columns]
    
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    
    return df

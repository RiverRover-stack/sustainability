"""
Smart AI Energy Consumption Predictor
Data Preprocessing Module

Handles data cleaning, feature engineering, and preparation for ML models.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath: str) -> pd.DataFrame:
    """Load energy consumption data from CSV."""
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df


def create_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract additional time-based features from timestamp."""
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
    def get_time_period(hour):
        if 0 <= hour < 6:
            return 'night'
        elif 6 <= hour < 12:
            return 'morning'
        elif 12 <= hour < 18:
            return 'afternoon'
        else:
            return 'evening'
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    # Cyclical encoding for hour and month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df


def create_lag_features(df: pd.DataFrame, target_col: str = 'consumption_kwh', 
                        lags: List[int] = [1, 2, 3, 6, 12, 24]) -> pd.DataFrame:
    """Create lagged features for time series prediction."""
    df = df.copy()
    
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    return df


def create_rolling_features(df: pd.DataFrame, target_col: str = 'consumption_kwh',
                           windows: List[int] = [3, 6, 12, 24]) -> pd.DataFrame:
    """Create rolling mean and std features."""
    df = df.copy()
    
    for window in windows:
        df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window=window).mean()
        df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window=window).std()
        df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window=window).max()
        df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window=window).min()
    
    return df


def create_diff_features(df: pd.DataFrame, target_col: str = 'consumption_kwh') -> pd.DataFrame:
    """Create difference features (rate of change)."""
    df = df.copy()
    
    df[f'{target_col}_diff_1'] = df[target_col].diff(1)
    df[f'{target_col}_diff_24'] = df[target_col].diff(24)  # Daily change
    
    return df


def encode_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot encode categorical columns."""
    df = df.copy()
    
    categorical_cols = ['season', 'time_period']
    existing_cats = [col for col in categorical_cols if col in df.columns]
    
    if existing_cats:
        df = pd.get_dummies(df, columns=existing_cats, drop_first=True)
    
    return df


def prepare_features(df: pd.DataFrame, include_lags: bool = True, 
                     include_rolling: bool = True) -> pd.DataFrame:
    """Full feature engineering pipeline."""
    df = df.copy()
    
    # Time features
    df = create_time_features(df)
    
    # Lag features
    if include_lags:
        df = create_lag_features(df)
    
    # Rolling features
    if include_rolling:
        df = create_rolling_features(df)
    
    # Difference features
    df = create_diff_features(df)
    
    # Encode categorical
    df = encode_categorical(df)
    
    # Drop rows with NaN (from lag/rolling features)
    df = df.dropna()
    
    return df


def prepare_data_for_ml(df: pd.DataFrame, target_col: str = 'consumption_kwh',
                        test_size: float = 0.2, scale: bool = True) -> Tuple:
    """
    Prepare data for machine learning models.
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names, scaler
    """
    df = df.copy()
    
    # Columns to exclude from features
    exclude_cols = ['timestamp', target_col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Time-based split (not random, to preserve temporal order)
    split_idx = int(len(df) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test, feature_cols, scaler


def prepare_data_for_lstm(df: pd.DataFrame, target_col: str = 'consumption_kwh',
                          sequence_length: int = 24, test_size: float = 0.2) -> Tuple:
    """
    Prepare sequences for LSTM model.
    
    Returns:
        X_train, X_test, y_train, y_test, scalers
    """
    df = df.copy()
    
    # Select features for LSTM
    feature_cols = ['consumption_kwh', 'temperature', 'humidity', 'occupancy',
                    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
                    'is_weekend', 'ac_running', 'lighting']
    
    # Get only existing columns
    feature_cols = [col for col in feature_cols if col in df.columns]
    
    data = df[feature_cols].values
    
    # Scale data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)
    
    # Create sequences
    X, y = [], []
    for i in range(sequence_length, len(data_scaled)):
        X.append(data_scaled[i-sequence_length:i])
        y.append(data_scaled[i, 0])  # consumption_kwh is first column
    
    X, y = np.array(X), np.array(y)
    
    # Split
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test, scaler, feature_cols


if __name__ == "__main__":
    import os
    
    # Test preprocessing
    data_path = os.path.join(os.path.dirname(__file__), "..", "data", "energy_data.csv")
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        print(f"Loaded {len(df)} records")
        
        df_processed = prepare_features(df)
        print(f"After feature engineering: {len(df_processed)} records, {len(df_processed.columns)} features")
        
        X_train, X_test, y_train, y_test, features, scaler = prepare_data_for_ml(df_processed)
        print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
    else:
        print(f"Data file not found at {data_path}")
        print("Run data_generator.py first to create the dataset.")

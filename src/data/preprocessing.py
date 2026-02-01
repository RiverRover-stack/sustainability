"""
Data Preprocessing Module

File Responsibility:
    Orchestrates data loading, feature preparation, and ML-ready data splits.
    Acts as the main interface for data preprocessing pipeline.

Inputs:
    - Raw CSV file path with energy consumption data

Outputs:
    - Processed DataFrames ready for ML model training
    - Train/test splits with proper temporal ordering

Assumptions:
    - Input CSV has 'timestamp' and 'consumption_kwh' columns
    - Data is in hourly granularity
    - Temporal train-test split is required (no random shuffling)

Failure Modes:
    - Missing data file raises FileNotFoundError
    - Missing required columns raises KeyError
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Import feature engineering functions
try:
    from feature_engineering import (
        create_time_features, create_lag_features, create_rolling_features,
        create_diff_features, encode_categorical
    )
except ImportError:
    from data.feature_engineering import (
        create_time_features, create_lag_features, create_rolling_features,
        create_diff_features, encode_categorical
    )


def load_data(filepath: str) -> pd.DataFrame:
    """
    Load energy consumption data from CSV.
    
    Purpose: Read and parse energy data with proper datetime handling.
    
    Inputs:
        filepath: Path to CSV file
        
    Outputs:
        DataFrame with parsed timestamp column
        
    Side effects: None
    """
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    return df


def prepare_features(
    df: pd.DataFrame, 
    include_lags: bool = True, 
    include_rolling: bool = True
) -> pd.DataFrame:
    """
    Full feature engineering pipeline.
    
    Purpose: Apply all feature transformations to raw data.
    
    Inputs:
        df: Raw DataFrame with timestamp and consumption data
        include_lags: Whether to include lag features
        include_rolling: Whether to include rolling statistics
        
    Outputs:
        DataFrame with all engineered features, NaN rows dropped
        
    Side effects: None (returns copy)
    """
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


def prepare_data_for_ml(
    df: pd.DataFrame, 
    target_col: str = 'consumption_kwh',
    test_size: float = 0.2, 
    scale: bool = True
) -> Tuple:
    """
    Prepare data for machine learning models.
    
    Purpose: Create train/test splits with temporal ordering and optional scaling.
    
    Inputs:
        df: Processed DataFrame with features
        target_col: Name of target column
        test_size: Fraction of data for testing (0.0-1.0)
        scale: Whether to apply StandardScaler
        
    Outputs:
        Tuple of (X_train, X_test, y_train, y_test, feature_names, scaler)
        
    Side effects: None
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


def prepare_data_for_lstm(
    df: pd.DataFrame, 
    target_col: str = 'consumption_kwh',
    sequence_length: int = 24, 
    test_size: float = 0.2
) -> Tuple:
    """
    Prepare sequences for LSTM model.
    
    Purpose: Create sliding window sequences for recurrent neural network.
    
    Inputs:
        df: DataFrame with features (should have time features added)
        target_col: Name of target column
        sequence_length: Number of timesteps in each sequence
        test_size: Fraction of data for testing
        
    Outputs:
        Tuple of (X_train, X_test, y_train, y_test, scaler, feature_names)
        
    Side effects: None
    """
    df = df.copy()
    
    # Select features for LSTM
    feature_cols = [
        'consumption_kwh', 'temperature', 'humidity', 'occupancy',
        'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
        'is_weekend', 'ac_running', 'lighting'
    ]
    
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
        print("✓ Preprocessing module working correctly!")
    else:
        print(f"Data file not found at {data_path}")
        print("Run data_generator.py first to create the dataset.")

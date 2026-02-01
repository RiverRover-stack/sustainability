"""Data package - Data generation and preprocessing."""

from .data_generator import generate_data
from .tariff_calculator import calculate_bill, get_tariff_slabs
from .preprocessing import load_data, prepare_features, prepare_data_for_ml, prepare_data_for_lstm
from .feature_engineering import (
    create_time_features, create_lag_features, create_rolling_features,
    create_diff_features, encode_categorical
)

__all__ = [
    'generate_data', 'calculate_bill', 'get_tariff_slabs', 'load_data',
    'prepare_features', 'prepare_data_for_ml', 'prepare_data_for_lstm',
    'create_time_features', 'create_lag_features', 'create_rolling_features',
    'create_diff_features', 'encode_categorical'
]

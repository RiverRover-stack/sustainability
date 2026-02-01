"""Data package - Data generation and preprocessing."""

import sys, os
_pkg_dir = os.path.dirname(os.path.abspath(__file__))
_src_dir = os.path.dirname(_pkg_dir)
if _pkg_dir not in sys.path: sys.path.insert(0, _pkg_dir)
if _src_dir not in sys.path: sys.path.insert(0, _src_dir)

from data_generator import generate_data
from tariff_calculator import calculate_bill, get_tariff_slabs
from preprocessing import load_data, prepare_features, prepare_data_for_ml, prepare_data_for_lstm
from feature_engineering import (
    create_time_features, create_lag_features, create_rolling_features,
    create_diff_features, encode_categorical
)

__all__ = [
    'generate_data', 'calculate_bill', 'get_tariff_slabs', 'load_data',
    'prepare_features', 'prepare_data_for_ml', 'prepare_data_for_lstm',
    'create_time_features', 'create_lag_features', 'create_rolling_features',
    'create_diff_features', 'encode_categorical'
]

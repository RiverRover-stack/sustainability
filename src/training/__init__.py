"""Training package - ML model training utilities."""

import sys
import os

src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

from training.evaluation import evaluate_model
from training.linear_regression import train_linear_regression
from training.random_forest import train_random_forest
from training.xgboost_trainer import train_xgboost
from training.lstm_trainer import train_lstm
from training.explainability import get_shap_explanations

__all__ = [
    'evaluate_model',
    'train_linear_regression',
    'train_random_forest',
    'train_xgboost',
    'train_lstm',
    'get_shap_explanations'
]

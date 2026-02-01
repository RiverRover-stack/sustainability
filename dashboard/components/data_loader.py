"""
Dashboard Data Loader Module

File Responsibility:
    Handles data and model loading for the dashboard.

Inputs:
    - File paths for data and models

Outputs:
    - DataFrames and model dictionaries

Assumptions:
    - Data files are CSV format
    - Models are pickled sklearn/xgboost objects

Failure Modes:
    - Missing files trigger data generation or warnings
"""

import os
import pickle
import pandas as pd
import streamlit as st


@st.cache_data
def load_energy_data() -> pd.DataFrame:
    """
    Load or generate energy data.
    
    Purpose: Provide data for dashboard analysis.
    
    Inputs: None (reads from standard location)
    
    Outputs:
        DataFrame with consumption data
        
    Side effects: May generate data if missing
    """
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))
    try:
        from data.data_generator import generate_data
    except ImportError:
        from data_generator import generate_data
    
    data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'energy_data.csv')
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, parse_dates=['timestamp'])
    else:
        st.info("Generating synthetic energy data...")
        df = generate_data(save_path=data_path)
    
    return df


@st.cache_resource
def load_models() -> dict:
    """
    Load trained ML models.
    
    Purpose: Provide models for forecasting page.
    
    Inputs: None (reads from standard location)
    
    Outputs:
        Dictionary of model name -> trained model
        
    Side effects: None
    """
    models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
    models = {}
    
    model_files = {
        'Linear Regression': 'linear_regression.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost_model.pkl'
    }
    
    for name, filename in model_files.items():
        path = os.path.join(models_dir, filename)
        if os.path.exists(path):
            with open(path, 'rb') as f:
                models[name] = pickle.load(f)
    
    return models


def load_model_comparison() -> pd.DataFrame:
    """Load model comparison results."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'model_comparison.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def load_feature_importance() -> pd.DataFrame:
    """Load feature importance data."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'feature_importance.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

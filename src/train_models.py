"""
Smart AI Energy Consumption Predictor
Model Training Module

Trains and evaluates multiple ML models for energy consumption forecasting.
Includes SHAP explainability for transparent predictions.
"""

import os
import pickle
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Import local modules
from preprocessing import load_data, prepare_features, prepare_data_for_ml, prepare_data_for_lstm


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calculate evaluation metrics for a model."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100
    
    metrics = {
        'model': model_name,
        'MAE': round(mae, 4),
        'RMSE': round(rmse, 4),
        'MAPE': round(mape, 2)
    }
    
    print(f"\n{model_name} Results:")
    print(f"  MAE:  {metrics['MAE']:.4f} kWh")
    print(f"  RMSE: {metrics['RMSE']:.4f} kWh")
    print(f"  MAPE: {metrics['MAPE']:.2f}%")
    
    return metrics


def train_linear_regression(X_train, y_train, X_test, y_test, save_path=None):
    """Train and evaluate Linear Regression model."""
    print("\n" + "="*50)
    print("Training Linear Regression (Baseline)")
    print("="*50)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Linear Regression")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model, metrics, y_pred


def train_random_forest(X_train, y_train, X_test, y_test, save_path=None):
    """Train and evaluate Random Forest model."""
    print("\n" + "="*50)
    print("Training Random Forest")
    print("="*50)
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Random Forest")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model, metrics, y_pred


def train_xgboost(X_train, y_train, X_test, y_test, save_path=None):
    """Train and evaluate XGBoost model."""
    print("\n" + "="*50)
    print("Training XGBoost")
    print("="*50)
    
    model = xgb.XGBRegressor(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "XGBoost")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model, metrics, y_pred


def train_lstm(X_train, y_train, X_test, y_test, scaler, save_path=None):
    """Train and evaluate LSTM model."""
    print("\n" + "="*50)
    print("Training LSTM")
    print("="*50)
    
    try:
        import tensorflow as tf
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense, Dropout
        from tensorflow.keras.callbacks import EarlyStopping
        
        # Suppress TensorFlow warnings
        tf.get_logger().setLevel('ERROR')
        
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
        
        print("Training LSTM model... (this may take a while)")
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=50,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0
        )
        
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Inverse transform predictions
        if scaler is not None:
            # Create dummy arrays for inverse transform
            y_test_inv = y_test * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
            y_pred_inv = y_pred * (scaler.data_max_[0] - scaler.data_min_[0]) + scaler.data_min_[0]
        else:
            y_test_inv, y_pred_inv = y_test, y_pred
        
        metrics = evaluate_model(y_test_inv, y_pred_inv, "LSTM")
        
        if save_path:
            model.save(save_path)
            print(f"Model saved to {save_path}")
        
        return model, metrics, y_pred_inv
        
    except ImportError:
        print("TensorFlow not available. Skipping LSTM training.")
        return None, None, None


def get_shap_explanations(model, X_train, X_test, feature_names, model_name="Model"):
    """Generate SHAP explanations for model predictions."""
    try:
        import shap
        
        print(f"\nGenerating SHAP explanations for {model_name}...")
        
        if model_name == "XGBoost":
            explainer = shap.TreeExplainer(model)
        elif model_name == "Random Forest":
            explainer = shap.TreeExplainer(model)
        else:
            # For linear models, use a sample for speed
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        
        shap_values = explainer.shap_values(X_test[:100])
        
        # Get feature importance from SHAP
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': np.abs(shap_values).mean(axis=0)
        }).sort_values('importance', ascending=False)
        
        print(f"\nTop 10 Important Features ({model_name}):")
        print(feature_importance.head(10).to_string(index=False))
        
        return shap_values, feature_importance
        
    except ImportError:
        print("SHAP not available. Install with: pip install shap")
        return None, None
    except Exception as e:
        print(f"SHAP explanation failed: {e}")
        return None, None


def train_all_models(data_path: str, models_dir: str):
    """Train all models and generate comparison report."""
    
    # Load and prepare data
    print("Loading and preprocessing data...")
    df = load_data(data_path)
    df_processed = prepare_features(df)
    
    # Prepare data for traditional ML models
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data_for_ml(df_processed)
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Features: {len(feature_names)}")
    
    # Create models directory
    os.makedirs(models_dir, exist_ok=True)
    
    all_metrics = []
    
    # Train Linear Regression
    lr_model, lr_metrics, lr_pred = train_linear_regression(
        X_train, y_train, X_test, y_test,
        save_path=os.path.join(models_dir, "linear_regression.pkl")
    )
    all_metrics.append(lr_metrics)
    
    # Train Random Forest
    rf_model, rf_metrics, rf_pred = train_random_forest(
        X_train, y_train, X_test, y_test,
        save_path=os.path.join(models_dir, "random_forest.pkl")
    )
    all_metrics.append(rf_metrics)
    
    # Train XGBoost
    xgb_model, xgb_metrics, xgb_pred = train_xgboost(
        X_train, y_train, X_test, y_test,
        save_path=os.path.join(models_dir, "xgboost_model.pkl")
    )
    all_metrics.append(xgb_metrics)
    
    # SHAP explanations for best model (XGBoost typically)
    shap_values, feature_importance = get_shap_explanations(
        xgb_model, X_train, X_test, feature_names, "XGBoost"
    )
    
    # Train LSTM
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, lstm_scaler, lstm_features = prepare_data_for_lstm(df)
    lstm_model, lstm_metrics, lstm_pred = train_lstm(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, lstm_scaler,
        save_path=os.path.join(models_dir, "lstm_model.h5")
    )
    if lstm_metrics:
        all_metrics.append(lstm_metrics)
    
    # Summary
    print("\n" + "="*60)
    print("MODEL COMPARISON SUMMARY")
    print("="*60)
    
    results_df = pd.DataFrame(all_metrics)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_idx = results_df['MAPE'].idxmin()
    best_model = results_df.loc[best_idx, 'model']
    best_mape = results_df.loc[best_idx, 'MAPE']
    
    print(f"\n✓ Best Model: {best_model} (MAPE: {best_mape:.2f}%)")
    
    if best_mape < 10:
        print("✓ SUCCESS: Prediction error is below 10% target!")
    else:
        print("⚠ Note: Consider adding more features or tuning hyperparameters")
    
    # Save results
    results_df.to_csv(os.path.join(models_dir, "model_comparison.csv"), index=False)
    
    if feature_importance is not None:
        feature_importance.to_csv(os.path.join(models_dir, "feature_importance.csv"), index=False)
    
    return results_df, feature_importance


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "energy_data.csv")
    models_dir = os.path.join(base_dir, "models")
    
    if os.path.exists(data_path):
        results, features = train_all_models(data_path, models_dir)
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_generator.py first to create the dataset.")

"""
Model Training Pipeline Module

File Responsibility:
    Orchestrates training of all ML models and generates comparison report.
    Now includes MLflow experiment tracking for versioning and comparison.

Inputs:
    - Data file path
    - Models output directory
    - Optional: MLflow experiment name

Outputs:
    - Trained model files (.pkl, .h5)
    - Model comparison CSV
    - Feature importance CSV
    - MLflow experiment logs

Assumptions:
    - Data is preprocessed by preprocessing module
    - Models directory exists or will be created
    - MLflow is installed (optional, graceful fallback)

Failure Modes:
    - Missing data file raises FileNotFoundError
    - Insufficient memory for large models
    - MLflow errors are logged but don't stop training
"""

import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Import preprocessing functions
try:
    from data.preprocessing import load_data, prepare_features, prepare_data_for_ml, prepare_data_for_lstm
except ImportError:
    from preprocessing import load_data, prepare_features, prepare_data_for_ml, prepare_data_for_lstm

# Import training functions from training package
from training import (
    train_linear_regression,
    train_random_forest,
    train_xgboost,
    train_lstm,
    get_shap_explanations
)

# Import MLflow tracking (optional)
try:
    from mlops import MLflowTracker, log_model_metrics
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Note: MLflow not available. Install with: pip install mlflow")


def train_all_models(
    data_path: str,
    models_dir: str,
    experiment_name: str = "energy-predictor",
    use_mlflow: bool = True
):
    """
    Train all models and generate comparison report.
    
    Purpose: Complete training pipeline with evaluation and experiment tracking.
    
    Inputs:
        data_path: Path to CSV data file
        models_dir: Directory to save trained models
        experiment_name: MLflow experiment name
        use_mlflow: Whether to log to MLflow
        
    Outputs:
        Tuple of (results DataFrame, feature importance DataFrame)
        
    Side effects: Saves models, CSVs, and MLflow logs
    """
    # Initialize MLflow if available
    tracker = None
    if use_mlflow and MLFLOW_AVAILABLE:
        try:
            tracker = MLflowTracker(experiment_name)
            print(f"✓ MLflow tracking enabled (experiment: {experiment_name})")
        except Exception as e:
            print(f"Warning: MLflow setup failed: {e}")
            tracker = None
    
    # Load and prepare data
    print("\nLoading and preprocessing data...")
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
    
    # ========== Train Linear Regression ==========
    if tracker:
        tracker.start_run(run_name="linear_regression")
        tracker.log_params({"model_type": "LinearRegression", "features": len(feature_names)})
    
    lr_model, lr_metrics, lr_pred = train_linear_regression(
        X_train, y_train, X_test, y_test,
        save_path=os.path.join(models_dir, "linear_regression.pkl")
    )
    all_metrics.append(lr_metrics)
    
    if tracker:
        tracker.log_metrics({"MAE": lr_metrics['MAE'], "RMSE": lr_metrics['RMSE'], "MAPE": lr_metrics['MAPE']})
        tracker.log_model(lr_model, "linear_regression", "sklearn")
        tracker.end_run()
    
    # ========== Train Random Forest ==========
    rf_params = {
        "n_estimators": 100,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2
    }
    
    if tracker:
        tracker.start_run(run_name="random_forest")
        tracker.log_params({**rf_params, "model_type": "RandomForest", "features": len(feature_names)})
    
    rf_model, rf_metrics, rf_pred = train_random_forest(
        X_train, y_train, X_test, y_test,
        save_path=os.path.join(models_dir, "random_forest.pkl")
    )
    all_metrics.append(rf_metrics)
    
    if tracker:
        tracker.log_metrics({"MAE": rf_metrics['MAE'], "RMSE": rf_metrics['RMSE'], "MAPE": rf_metrics['MAPE']})
        tracker.log_model(rf_model, "random_forest", "sklearn")
        tracker.end_run()
    
    # ========== Train XGBoost ==========
    xgb_params = {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8
    }
    
    if tracker:
        tracker.start_run(run_name="xgboost")
        tracker.log_params({**xgb_params, "model_type": "XGBoost", "features": len(feature_names)})
    
    xgb_model, xgb_metrics, xgb_pred = train_xgboost(
        X_train, y_train, X_test, y_test,
        save_path=os.path.join(models_dir, "xgboost_model.pkl")
    )
    all_metrics.append(xgb_metrics)
    
    if tracker:
        tracker.log_metrics({"MAE": xgb_metrics['MAE'], "RMSE": xgb_metrics['RMSE'], "MAPE": xgb_metrics['MAPE']})
        tracker.log_model(xgb_model, "xgboost", "xgboost")
        tracker.end_run()
    
    # SHAP explanations for XGBoost
    shap_values, feature_importance = get_shap_explanations(
        xgb_model, X_train, X_test, feature_names, "XGBoost"
    )
    
    # ========== Train LSTM ==========
    X_train_lstm, X_test_lstm, y_train_lstm, y_test_lstm, lstm_scaler, lstm_features = prepare_data_for_lstm(df)
    
    lstm_params = {
        "sequence_length": 24,
        "lstm_units": [64, 32],
        "dropout": 0.2,
        "epochs": 50
    }
    
    if tracker:
        tracker.start_run(run_name="lstm")
        tracker.log_params({**lstm_params, "model_type": "LSTM", "features": len(lstm_features)})
    
    lstm_model, lstm_metrics, lstm_pred = train_lstm(
        X_train_lstm, y_train_lstm, X_test_lstm, y_test_lstm, lstm_scaler,
        save_path=os.path.join(models_dir, "lstm_model.h5")
    )
    
    if lstm_metrics:
        all_metrics.append(lstm_metrics)
        if tracker:
            tracker.log_metrics({"MAE": lstm_metrics['MAE'], "RMSE": lstm_metrics['RMSE'], "MAPE": lstm_metrics['MAPE']})
    
    if tracker:
        tracker.end_run()
    
    # ========== Summary ==========
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    
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
    results_path = os.path.join(models_dir, "model_comparison.csv")
    results_df.to_csv(results_path, index=False)
    
    if feature_importance is not None:
        importance_path = os.path.join(models_dir, "feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        
        # Log artifacts to MLflow
        if tracker:
            tracker.start_run(run_name="artifacts")
            tracker.log_artifact(results_path, "results")
            tracker.log_artifact(importance_path, "results")
            tracker.end_run()
    
    if MLFLOW_AVAILABLE and use_mlflow:
        print(f"\n📊 View experiments: mlflow ui --port 5000")
    
    return results_df, feature_importance


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(__file__))
    data_path = os.path.join(base_dir, "data", "energy_data.csv")
    models_dir = os.path.join(base_dir, "models")
    
    if os.path.exists(data_path):
        results, features = train_all_models(data_path, models_dir)
        print("\n✓ Training pipeline completed successfully!")
    else:
        print(f"Data file not found: {data_path}")
        print("Run data_generator.py first to create the dataset.")

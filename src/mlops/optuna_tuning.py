"""
Optuna Hyperparameter Tuning Module

File Responsibility:
    Automated hyperparameter optimization for ML models using Optuna.
    Integrates with MLflow for experiment tracking.

Inputs:
    - Training data (X_train, y_train, X_test, y_test)
    - Model type to optimize
    - Number of trials

Outputs:
    - Best hyperparameters
    - Optimized model
    - MLflow logged experiments

Assumptions:
    - Data is preprocessed and ready
    - Optuna and MLflow are installed

Failure Modes:
    - Memory errors for large search spaces
    - Timeout for long trials
"""

import os
import sys
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import cross_val_score

try:
    import optuna
    from optuna.integration import MLflowCallback
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    print("Warning: Optuna not installed. Run: pip install optuna")


class HyperparameterTuner:
    """
    Hyperparameter optimization using Optuna.
    
    Purpose: Automatically find optimal hyperparameters for ML models.
    
    Inputs:
        model_type: Type of model ('random_forest', 'xgboost', 'lstm')
        n_trials: Number of optimization trials
        use_mlflow: Whether to log to MLflow
    
    Outputs:
        Best parameters and trained model
    
    Side effects: 
        - Creates Optuna study database
        - Logs to MLflow if enabled
    """
    
    def __init__(
        self,
        model_type: str = "xgboost",
        n_trials: int = 50,
        use_mlflow: bool = True
    ):
        if not OPTUNA_AVAILABLE:
            raise ImportError("Optuna required. Install: pip install optuna")
        
        self.model_type = model_type
        self.n_trials = n_trials
        self.use_mlflow = use_mlflow
        self.best_params = None
        self.best_score = None
    
    def _get_rf_params(self, trial: optuna.Trial) -> Dict:
        """Define Random Forest search space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 50, 300, step=50),
            'max_depth': trial.suggest_int('max_depth', 5, 30),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
            'random_state': 42,
            'n_jobs': -1
        }
    
    def _get_xgb_params(self, trial: optuna.Trial) -> Dict:
        """Define XGBoost search space."""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 500, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': 42,
            'verbosity': 0
        }
    
    def _get_lstm_params(self, trial: optuna.Trial) -> Dict:
        """Define LSTM search space."""
        return {
            'lstm_units_1': trial.suggest_int('lstm_units_1', 32, 128, step=32),
            'lstm_units_2': trial.suggest_int('lstm_units_2', 16, 64, step=16),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'batch_size': trial.suggest_categorical('batch_size', [16, 32, 64]),
            'epochs': 50  # Fixed, use early stopping
        }
    
    def _create_objective(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ):
        """
        Create Optuna objective function.
        
        Purpose: Define what to optimize (minimize MAPE).
        """
        def objective(trial: optuna.Trial) -> float:
            # Get params based on model type
            if self.model_type == 'random_forest':
                from sklearn.ensemble import RandomForestRegressor
                params = self._get_rf_params(trial)
                model = RandomForestRegressor(**params)
            
            elif self.model_type == 'xgboost':
                import xgboost as xgb
                params = self._get_xgb_params(trial)
                model = xgb.XGBRegressor(**params)
            
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            # Cross-validation
            scores = cross_val_score(
                model, X_train, y_train,
                cv=5,
                scoring='neg_mean_absolute_percentage_error',
                n_jobs=-1
            )
            
            # Return mean MAPE (negated because sklearn uses negative)
            mape = -scores.mean() * 100
            
            return mape
        
        return objective
    
    def optimize(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray
    ) -> Tuple[Dict, float]:
        """
        Run hyperparameter optimization.
        
        Purpose: Find best parameters for the model.
        
        Inputs:
            X_train, y_train: Training data
            X_test, y_test: Test data for final evaluation
        
        Outputs:
            Tuple of (best_params, best_score)
        
        Side effects: Runs n_trials optimization trials
        """
        print(f"\n🔍 Starting Optuna optimization for {self.model_type}")
        print(f"   Trials: {self.n_trials}")
        
        # Create study
        study = optuna.create_study(
            direction='minimize',
            study_name=f'{self.model_type}_optimization',
            pruner=optuna.pruners.MedianPruner()
        )
        
        # Create objective
        objective = self._create_objective(X_train, y_train, X_test, y_test)
        
        # Callbacks
        callbacks = []
        if self.use_mlflow:
            try:
                callbacks.append(MLflowCallback(
                    tracking_uri="file:./mlruns",
                    metric_name="MAPE"
                ))
            except Exception:
                pass  # MLflow not configured
        
        # Optimize
        study.optimize(
            objective,
            n_trials=self.n_trials,
            callbacks=callbacks,
            show_progress_bar=True
        )
        
        # Results
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        print(f"\n✓ Best MAPE: {self.best_score:.2f}%")
        print(f"✓ Best params: {self.best_params}")
        
        return self.best_params, self.best_score
    
    def train_best_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray
    ):
        """
        Train model with best parameters.
        
        Purpose: Create final model using optimized hyperparameters.
        
        Inputs:
            X_train, y_train: Training data
        
        Outputs:
            Trained model with optimal parameters
        """
        if self.best_params is None:
            raise ValueError("Run optimize() first")
        
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model = RandomForestRegressor(**self.best_params)
        
        elif self.model_type == 'xgboost':
            import xgboost as xgb
            model = xgb.XGBRegressor(**self.best_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        model.fit(X_train, y_train)
        return model


def run_optimization(
    data_path: str,
    model_type: str = "xgboost",
    n_trials: int = 50
) -> Dict:
    """
    Run complete optimization pipeline.
    
    Purpose: End-to-end hyperparameter optimization.
    
    Inputs:
        data_path: Path to data CSV
        model_type: Model to optimize
        n_trials: Number of trials
    
    Outputs:
        Dictionary with best params, score, and model
    """
    # Add src to path
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from data.preprocessing import load_data, prepare_features, prepare_data_for_ml
    
    # Load data
    df = load_data(data_path)
    df_processed = prepare_features(df)
    X_train, X_test, y_train, y_test, feature_names, scaler = prepare_data_for_ml(df_processed)
    
    # Optimize
    tuner = HyperparameterTuner(model_type, n_trials)
    best_params, best_score = tuner.optimize(X_train, y_train, X_test, y_test)
    
    # Train best model
    best_model = tuner.train_best_model(X_train, y_train)
    
    # Evaluate on test set
    from sklearn.metrics import mean_absolute_percentage_error
    y_pred = best_model.predict(X_test)
    test_mape = mean_absolute_percentage_error(y_test, y_pred) * 100
    
    print(f"\n✓ Test MAPE with best params: {test_mape:.2f}%")
    
    return {
        'best_params': best_params,
        'cv_score': best_score,
        'test_score': test_mape,
        'model': best_model
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Hyperparameter optimization")
    parser.add_argument('--model', type=str, default='xgboost', choices=['random_forest', 'xgboost'])
    parser.add_argument('--n-trials', type=int, default=50)
    args = parser.parse_args()
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, 'data', 'energy_data.csv')
    
    if os.path.exists(data_path):
        results = run_optimization(data_path, args.model, args.n_trials)
        print("\n✓ Optimization complete!")
    else:
        print(f"Data not found: {data_path}")

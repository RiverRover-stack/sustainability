"""XGBoost model trainer."""

import pickle
import xgboost as xgb

try:
    from training.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


def train_xgboost(X_train, y_train, X_test, y_test, save_path=None):
    """Train XGBoost and return (model, metrics, predictions)."""
    print("\n" + "=" * 50)
    print("Training XGBoost")
    print("=" * 50)
    
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

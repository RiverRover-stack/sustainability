"""Random Forest model trainer."""

import pickle
from sklearn.ensemble import RandomForestRegressor
try:
    from training.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


def train_random_forest(X_train, y_train, X_test, y_test, save_path=None):
    """Train Random Forest and return (model, metrics, predictions)."""
    print("\n" + "=" * 50)
    print("Training Random Forest")
    print("=" * 50)
    
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

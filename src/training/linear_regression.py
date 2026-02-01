"""Linear Regression model trainer."""

import pickle
from sklearn.linear_model import LinearRegression
try:
    from training.evaluation import evaluate_model
except ImportError:
    from evaluation import evaluate_model


def train_linear_regression(X_train, y_train, X_test, y_test, save_path=None):
    """Train Linear Regression baseline and return (model, metrics, predictions)."""
    print("\n" + "=" * 50)
    print("Training Linear Regression (Baseline)")
    print("=" * 50)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, "Linear Regression")
    
    if save_path:
        with open(save_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {save_path}")
    
    return model, metrics, y_pred

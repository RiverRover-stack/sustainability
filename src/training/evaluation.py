"""Model evaluation metrics."""

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str) -> dict:
    """Calculate MAE, RMSE, MAPE metrics and print results."""
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

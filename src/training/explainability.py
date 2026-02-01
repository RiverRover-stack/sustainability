"""SHAP model explainability."""

import numpy as np
import pandas as pd


def get_shap_explanations(model, X_train, X_test, feature_names, model_name="Model"):
    """Generate SHAP explanations. Returns (shap_values, importance_df) or (None, None)."""
    try:
        import shap
        
        print(f"\nGenerating SHAP explanations for {model_name}...")
        
        if model_name in ["XGBoost", "Random Forest"]:
            explainer = shap.TreeExplainer(model)
        else:
            explainer = shap.KernelExplainer(model.predict, shap.sample(X_train, 100))
        
        shap_values = explainer.shap_values(X_test[:100])
        
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

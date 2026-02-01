"""
MLflow Experiment Tracker

File Responsibility:
    Provides utilities for tracking ML experiments, logging metrics,
    and managing model artifacts with MLflow.

Inputs:
    - Model training parameters
    - Evaluation metrics
    - Trained model objects

Outputs:
    - Logged experiments in MLflow
    - Model artifacts in registry

Assumptions:
    - MLflow tracking URI is local file system or server
    - Experiment names follow project convention

Failure Modes:
    - Connection error if MLflow server unavailable
    - Disk space error for large model artifacts
"""

import os
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import mlflow
    from mlflow.tracking import MlflowClient
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("Warning: MLflow not installed. Run: pip install mlflow")


# Default tracking configuration
DEFAULT_TRACKING_URI = "file:./mlruns"
DEFAULT_EXPERIMENT_NAME = "energy-predictor"


class MLflowTracker:
    """
    MLflow experiment tracking wrapper.
    
    Purpose: Simplify MLflow logging with project-specific defaults.
    
    Inputs:
        experiment_name: Name for grouping runs
        tracking_uri: MLflow server or file path
    
    Outputs:
        Logged experiments accessible via MLflow UI
    
    Side effects: Creates mlruns directory if using file storage
    """
    
    def __init__(
        self,
        experiment_name: str = DEFAULT_EXPERIMENT_NAME,
        tracking_uri: str = DEFAULT_TRACKING_URI
    ):
        if not MLFLOW_AVAILABLE:
            raise ImportError("MLflow is required. Install with: pip install mlflow")
        
        self.experiment_name = experiment_name
        self.tracking_uri = tracking_uri
        self._setup_mlflow()
    
    def _setup_mlflow(self):
        """Initialize MLflow tracking."""
        mlflow.set_tracking_uri(self.tracking_uri)
        
        # Create or get experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(
                self.experiment_name,
                tags={"project": "SDG7-Energy-Predictor"}
            )
        else:
            self.experiment_id = experiment.experiment_id
        
        mlflow.set_experiment(self.experiment_name)
    
    def start_run(self, run_name: Optional[str] = None) -> str:
        """
        Start a new MLflow run.
        
        Purpose: Begin tracking a training session.
        
        Inputs:
            run_name: Optional descriptive name for the run
        
        Outputs:
            run_id: Unique identifier for this run
        
        Side effects: Creates new run in MLflow
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.active_run = mlflow.start_run(run_name=run_name)
        return self.active_run.info.run_id
    
    def log_params(self, params: Dict[str, Any]):
        """
        Log hyperparameters.
        
        Purpose: Record model configuration.
        
        Inputs:
            params: Dictionary of parameter names and values
        
        Side effects: Logs to active MLflow run
        """
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """
        Log evaluation metrics.
        
        Purpose: Record model performance.
        
        Inputs:
            metrics: Dictionary of metric names and values
            step: Optional step number for time-series metrics
        
        Side effects: Logs to active MLflow run
        """
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, model_name: str, model_type: str = "sklearn"):
        """
        Log trained model artifact.
        
        Purpose: Save model for later retrieval.
        
        Inputs:
            model: Trained model object
            model_name: Name for the artifact
            model_type: Type of model (sklearn, xgboost, tensorflow)
        
        Side effects: Saves model artifact to MLflow
        """
        if model_type == "sklearn":
            mlflow.sklearn.log_model(model, model_name)
        elif model_type == "xgboost":
            mlflow.xgboost.log_model(model, model_name)
        elif model_type == "tensorflow":
            mlflow.tensorflow.log_model(model, model_name)
        else:
            # Generic pickle logging
            mlflow.pyfunc.log_model(model_name, python_model=model)
    
    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """
        Log any file as artifact.
        
        Purpose: Save plots, CSVs, or other files.
        
        Inputs:
            local_path: Path to file on disk
            artifact_path: Optional subdirectory in artifacts
        
        Side effects: Copies file to MLflow artifact storage
        """
        mlflow.log_artifact(local_path, artifact_path)
    
    def end_run(self):
        """End the active run."""
        mlflow.end_run()
    
    def get_best_run(self, metric: str = "MAPE", ascending: bool = True) -> Dict:
        """
        Find the best run by a metric.
        
        Purpose: Identify top-performing model.
        
        Inputs:
            metric: Metric name to compare
            ascending: True if lower is better
        
        Outputs:
            Dictionary with run info and metrics
        """
        client = MlflowClient()
        runs = client.search_runs(
            experiment_ids=[self.experiment_id],
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1
        )
        
        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "metrics": run.data.metrics,
                "params": run.data.params
            }
        return {}


# Convenience functions for simpler usage
def start_experiment(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    run_name: Optional[str] = None
) -> MLflowTracker:
    """
    Quick start for experiment tracking.
    
    Purpose: One-liner to begin tracking.
    
    Inputs:
        experiment_name: Name for the experiment
        run_name: Optional name for this run
    
    Outputs:
        Initialized MLflowTracker with active run
    """
    tracker = MLflowTracker(experiment_name)
    tracker.start_run(run_name)
    return tracker


def log_model_metrics(
    model_name: str,
    params: Dict[str, Any],
    metrics: Dict[str, float],
    model: Any = None,
    model_type: str = "sklearn"
):
    """
    Log complete model training results.
    
    Purpose: Single function to log everything about a model.
    
    Inputs:
        model_name: Identifier for this model
        params: Hyperparameters used
        metrics: Evaluation metrics
        model: Optional model object to save
        model_type: Type of model for serialization
    
    Side effects: Creates complete MLflow run
    """
    if not MLFLOW_AVAILABLE:
        print(f"MLflow not available. Metrics: {metrics}")
        return
    
    tracker = start_experiment(run_name=model_name)
    tracker.log_params(params)
    tracker.log_metrics(metrics)
    
    if model is not None:
        tracker.log_model(model, model_name, model_type)
    
    tracker.end_run()


def log_model_artifact(local_path: str, artifact_name: str = "artifacts"):
    """Log a file artifact to the current experiment."""
    if MLFLOW_AVAILABLE:
        mlflow.log_artifact(local_path, artifact_name)


def get_best_run(
    experiment_name: str = DEFAULT_EXPERIMENT_NAME,
    metric: str = "MAPE"
) -> Dict:
    """Get the best performing run from an experiment."""
    if not MLFLOW_AVAILABLE:
        return {}
    
    tracker = MLflowTracker(experiment_name)
    return tracker.get_best_run(metric)

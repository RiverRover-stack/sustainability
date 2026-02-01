"""
MLOps Package

File Responsibility:
    Export MLflow tracking and experiment management utilities.

Inputs:
    None (package initialization)

Outputs:
    Public API for MLOps functionality

Assumptions:
    - MLflow is installed
    - Local tracking server or file storage

Failure Modes:
    - ImportError if mlflow not installed
"""

from .mlflow_tracker import (
    MLflowTracker,
    start_experiment,
    log_model_metrics,
    log_model_artifact,
    get_best_run
)

from .optuna_tuning import (
    HyperparameterTuner,
    run_optimization
)

__all__ = [
    'MLflowTracker',
    'start_experiment',
    'log_model_metrics',
    'log_model_artifact',
    'get_best_run'
]

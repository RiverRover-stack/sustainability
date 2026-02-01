"""
Anomaly Detection Package

File Responsibility:
    Export anomaly detection utilities for identifying unusual consumption patterns.

Inputs:
    None (package initialization)

Outputs:
    Public API for anomaly detection

Assumptions:
    - scikit-learn is installed (Isolation Forest)

Failure Modes:
    - ImportError if sklearn not installed
"""

from .detector import (
    AnomalyDetector,
    detect_anomalies,
    get_anomaly_summary
)

__all__ = [
    'AnomalyDetector',
    'detect_anomalies',
    'get_anomaly_summary'
]

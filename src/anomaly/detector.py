"""
Energy Consumption Anomaly Detector

File Responsibility:
    Detect unusual consumption patterns that may indicate waste, faults,
    or behavioral changes requiring attention.

Inputs:
    - Consumption DataFrame with timestamp and consumption_kwh columns

Outputs:
    - Anomaly flags for each record
    - Summary of detected anomalies
    - Actionable alerts

Assumptions:
    - Data has regular time intervals
    - Historical data available for baseline

Failure Modes:
    - Insufficient data for training
    - All values flagged if contamination too high
"""

import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyDetector:
    """
    Anomaly detection using Isolation Forest.
    
    Purpose: Identify unusual consumption patterns for alerting.
    
    Inputs:
        contamination: Expected proportion of anomalies (default 5%)
        n_estimators: Number of trees in the forest
    
    Outputs:
        DataFrame with anomaly flags and scores
    
    Side effects: None
    """
    
    def __init__(
        self,
        contamination: float = 0.05,
        n_estimators: int = 100
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = None
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def _prepare_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Prepare features for anomaly detection.
        
        Purpose: Extract relevant features from consumption data.
        
        Inputs:
            df: DataFrame with consumption data
        
        Outputs:
            Feature matrix for Isolation Forest
        """
        features = []
        
        # Base consumption
        if 'consumption_kwh' in df.columns:
            features.append(df['consumption_kwh'].values)
        
        # Hour of day
        if 'hour' in df.columns:
            features.append(df['hour'].values)
        elif 'timestamp' in df.columns:
            features.append(pd.to_datetime(df['timestamp']).dt.hour.values)
        
        # Temperature if available
        if 'temperature' in df.columns:
            features.append(df['temperature'].values)
        
        # Deviation from daily mean
        if 'consumption_kwh' in df.columns and 'timestamp' in df.columns:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['timestamp']).dt.date
            daily_means = df_temp.groupby('date')['consumption_kwh'].transform('mean')
            deviation = df_temp['consumption_kwh'] - daily_means
            features.append(deviation.values)
        
        return np.column_stack(features)
    
    def fit(self, df: pd.DataFrame):
        """
        Fit the anomaly detector on historical data.
        
        Purpose: Learn normal consumption patterns.
        
        Inputs:
            df: DataFrame with consumption history
        
        Side effects: Trains the Isolation Forest model
        """
        X = self._prepare_features(df)
        X_scaled = self.scaler.fit_transform(X)
        
        self.model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=42,
            n_jobs=-1
        )
        
        self.model.fit(X_scaled)
        self.is_fitted = True
    
    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in consumption data.
        
        Purpose: Flag unusual consumption patterns.
        
        Inputs:
            df: DataFrame with consumption data
        
        Outputs:
            DataFrame with added columns:
            - is_anomaly: Boolean flag
            - anomaly_score: Lower = more anomalous
        """
        if not self.is_fitted:
            self.fit(df)
        
        X = self._prepare_features(df)
        X_scaled = self.scaler.transform(X)
        
        result = df.copy()
        
        # Predict: 1 = normal, -1 = anomaly
        predictions = self.model.predict(X_scaled)
        result['is_anomaly'] = predictions == -1
        
        # Get anomaly scores (lower = more anomalous)
        result['anomaly_score'] = self.model.decision_function(X_scaled)
        
        return result
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of detected anomalies.
        
        Purpose: Provide actionable summary for dashboard.
        
        Inputs:
            df: DataFrame with anomaly flags (from detect())
        
        Outputs:
            Dictionary with anomaly statistics and top anomalies
        """
        if 'is_anomaly' not in df.columns:
            df = self.detect(df)
        
        anomalies = df[df['is_anomaly']]
        
        summary = {
            'total_records': len(df),
            'anomaly_count': len(anomalies),
            'anomaly_percent': round(len(anomalies) / len(df) * 100, 2),
            'top_anomalies': []
        }
        
        # Get top 5 most anomalous records
        if len(anomalies) > 0:
            top = anomalies.nsmallest(5, 'anomaly_score')
            
            for _, row in top.iterrows():
                summary['top_anomalies'].append({
                    'timestamp': str(row.get('timestamp', 'N/A')),
                    'consumption_kwh': round(row.get('consumption_kwh', 0), 2),
                    'score': round(row['anomaly_score'], 4),
                    'reason': self._infer_reason(row, df)
                })
        
        return summary
    
    def _infer_reason(self, row: pd.Series, df: pd.DataFrame) -> str:
        """
        Infer likely reason for anomaly.
        
        Purpose: Provide actionable context for alerts.
        
        Note: This is heuristic-based, not definitive.
        """
        consumption = row.get('consumption_kwh', 0)
        avg_consumption = df['consumption_kwh'].mean()
        
        if consumption > avg_consumption * 2:
            return "Unusually high consumption (>2x average)"
        elif consumption > avg_consumption * 1.5:
            return "High consumption (>1.5x average)"
        elif consumption < avg_consumption * 0.2:
            return "Unusually low consumption (<20% average)"
        elif consumption < avg_consumption * 0.5:
            return "Low consumption (<50% average)"
        else:
            return "Unusual pattern (time/temperature mismatch)"
    
    def get_alerts(self, df: pd.DataFrame, threshold: float = -0.3) -> List[Dict]:
        """
        Generate alerts for severe anomalies.
        
        Purpose: Create actionable notifications.
        
        Inputs:
            df: DataFrame with consumption data
            threshold: Anomaly score threshold for alerts
        
        Outputs:
            List of alert dictionaries
        """
        if 'is_anomaly' not in df.columns:
            df = self.detect(df)
        
        alerts = []
        severe = df[df['anomaly_score'] < threshold]
        
        for _, row in severe.iterrows():
            severity = 'high' if row['anomaly_score'] < -0.5 else 'medium'
            
            alerts.append({
                'timestamp': str(row.get('timestamp', 'N/A')),
                'severity': severity,
                'consumption_kwh': round(row.get('consumption_kwh', 0), 2),
                'message': self._infer_reason(row, df),
                'recommended_action': self._recommend_action(row, df)
            })
        
        return alerts
    
    def _recommend_action(self, row: pd.Series, df: pd.DataFrame) -> str:
        """Generate recommended action for an anomaly."""
        consumption = row.get('consumption_kwh', 0)
        avg = df['consumption_kwh'].mean()
        
        if consumption > avg * 2:
            return "Check for faulty appliances or unintended usage"
        elif consumption > avg * 1.5:
            return "Review recent appliance usage patterns"
        elif consumption < avg * 0.2:
            return "Verify meter is functioning correctly"
        else:
            return "Investigate unusual consumption timing"


# Convenience functions

def detect_anomalies(df: pd.DataFrame, contamination: float = 0.05) -> pd.DataFrame:
    """
    Quick anomaly detection.
    
    Purpose: One-liner for anomaly detection.
    
    Inputs:
        df: DataFrame with consumption data
        contamination: Expected anomaly proportion
    
    Outputs:
        DataFrame with anomaly flags
    """
    detector = AnomalyDetector(contamination=contamination)
    return detector.detect(df)


def get_anomaly_summary(df: pd.DataFrame) -> Dict:
    """
    Get anomaly detection summary.
    
    Purpose: Quick summary for dashboard display.
    
    Inputs:
        df: DataFrame with consumption data
    
    Outputs:
        Summary dictionary with statistics and top anomalies
    """
    detector = AnomalyDetector()
    df_with_anomalies = detector.detect(df)
    return detector.get_summary(df_with_anomalies)


if __name__ == "__main__":
    # Test with sample data
    import sys
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    
    from data.preprocessing import load_data
    
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data_path = os.path.join(base_dir, 'data', 'energy_data.csv')
    
    if os.path.exists(data_path):
        df = load_data(data_path)
        
        detector = AnomalyDetector(contamination=0.05)
        df_flagged = detector.detect(df)
        summary = detector.get_summary(df_flagged)
        
        print("\n🔍 Anomaly Detection Results")
        print(f"   Total records: {summary['total_records']}")
        print(f"   Anomalies found: {summary['anomaly_count']} ({summary['anomaly_percent']}%)")
        
        if summary['top_anomalies']:
            print("\n   Top anomalies:")
            for a in summary['top_anomalies']:
                print(f"   - {a['timestamp']}: {a['consumption_kwh']} kWh ({a['reason']})")
    else:
        print(f"Data not found: {data_path}")

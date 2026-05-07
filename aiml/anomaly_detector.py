# ============================================================================
# ANOMALY_DETECTOR.py - Isolation Forest-based anomaly detection
# ============================================================================

from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


class AnomalyDetector:
    """IsolationForest-based market anomaly detector."""
    
    def __init__(self, contamination: float = 0.05):
        """
        Initialize anomaly detector.
        
        Args:
            contamination: Expected % of anomalies
        """
        self.model = IsolationForest(contamination=contamination, random_state=42)
        self.is_trained = False
        self.anomalies = []
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for anomaly detection."""
        features = pd.DataFrame()
        
        close = df['Close']
        volume = df['Volume'] if 'Volume' in df else pd.Series(0, index=df.index)
        
        # Features
        features['returns'] = close.pct_change()
        features['returns_squared'] = features['returns'] ** 2
        features['volume_change'] = volume.pct_change()
        features['high_low_range'] = (df['High'] - df['Low']) / close
        features['open_close_range'] = (df['Close'] - df['Open']) / close
        
        # Rolling statistics
        features['volatility'] = features['returns'].rolling(20).std()
        features['volume_sma_ratio'] = volume / volume.rolling(20).mean()
        
        return features.fillna(0)
    
    def train(self, df: pd.DataFrame) -> None:
        """Train anomaly detector."""
        features = self.extract_features(df)
        self.model.fit(features)
        self.is_trained = True
    
    def detect_anomalies(self, df: pd.DataFrame) -> Dict[str, any]:
        """Detect anomalies."""
        if not self.is_trained:
            return {'anomalies': [], 'count': 0}
        
        features = self.extract_features(df)
        predictions = self.model.predict(features)
        scores = self.model.score_samples(features)
        
        anomaly_indices = np.where(predictions == -1)[0]
        
        anomalies = []
        for idx in anomaly_indices:
            anomalies.append({
                'date': str(df.index[idx]) if hasattr(df, 'index') else str(idx),
                'anomaly_score': float(scores[idx]),
                'close': float(df['Close'].iloc[idx]) if 'Close' in df else None,
            })
        
        return {
            'anomalies': anomalies,
            'count': len(anomalies),
            'anomaly_scores': scores,
        }


def main():
    """Test anomaly detector."""
    detector = AnomalyDetector()
    print("Anomaly detector initialized")


if __name__ == "__main__":
    main()

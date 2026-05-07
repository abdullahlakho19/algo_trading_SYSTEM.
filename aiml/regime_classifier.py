# ============================================================================
# REGIME_CLASSIFIER.PY - RandomForest regime classification
# ============================================================================

from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


class RegimeClassifier:
    """RandomForest-based market regime classifier."""
    
    def __init__(self):
        """Initialize classifier."""
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.is_trained = False
        self.feature_names = []
    
    def extract_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features for classification."""
        features = pd.DataFrame()
        
        # Technical indicators as features
        close = df['Close']
        volume = df['Volume']
        
        features['sma_ratio'] = close / close.rolling(20).mean()
        features['rsi'] = self._calculate_rsi(close)
        features['bbh_pos'] = (close - close.rolling(20).mean()) / (2 * close.rolling(20).std())
        features['volume_sma'] = volume / volume.rolling(20).mean()
        features['volatility'] = close.pct_change().rolling(20).std()
        
        self.feature_names = features.columns.tolist()
        
        return features.fillna(0)
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def train(self, df: pd.DataFrame, labels: np.ndarray) -> None:
        """Train classifier."""
        features = self.extract_features(df)
        self.model.fit(features, labels)
        self.is_trained = True
    
    def predict(self, df: pd.DataFrame) -> str:
        """Predict regime."""
        if not self.is_trained:
            return "untrained"
        
        features = self.extract_features(df)
        predictions = self.model.predict(features)
        
        regime_map = {0: 'accumulation', 1: 'distribution', 2: 'uptrend', 3: 'downtrend'}
        return regime_map.get(predictions[-1], 'unknown')
    
    def predict_proba(self, df: pd.DataFrame) -> Dict[str, float]:
        """Get prediction probabilities."""
        if not self.is_trained:
            return {}
        
        features = self.extract_features(df)
        proba = self.model.predict_proba(features)[-1]
        
        regime_map = {0: 'accumulation', 1: 'distribution', 2: 'uptrend', 3: 'downtrend'}
        
        return {regime_map.get(i, 'unknown'): float(p) for i, p in enumerate(proba)}


def main():
    """Test regime classifier."""
    classifier = RegimeClassifier()
    print("Regime classifier initialized")


if __name__ == "__main__":
    main()

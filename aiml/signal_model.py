# ============================================================================
# SIGNAL_MODEL.PY - XGBoost/LSTM main signal generation model
# ============================================================================

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


class SignalModel:
    """XGBoost-based primary signal generation model."""
    
    def __init__(self):
        """Initialize signal model."""
        self.model = XGBClassifier(n_estimators=100, random_state=42, early_stopping_rounds=10)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def create_training_data(self, df: pd.DataFrame, target: np.ndarray) -> tuple:
        """Create training data."""
        features = self._extract_features(df)
        X_scaled = self.scaler.fit_transform(features)
        return X_scaled, target
    
    def _extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """Extract features for model."""
        close = df['Close'].values
        volume = df['Volume'].values if 'Volume' in df else np.ones(len(df))
        high = df['High'].values
        low = df['Low'].values
        
        features = []
        
        # Price features
        features.append((close[-1] - close[0]) / close[0])  # Return
        features.append(np.std(np.diff(close)))              # Volatility
        
        # Volume features
        features.append(np.mean(volume[-20:]) if len(volume) > 20 else 0)
        
        # Trend features
        sma_20 = np.mean(close[-20:]) if len(close) > 20 else close[-1]
        features.append(close[-1] / sma_20 if sma_20 > 0 else 1)
        
        # Range
        features.append((high[-1] - low[-1]) / low[-1])
        
        return np.array(features).reshape(1, -1)
    
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        """Train model."""
        self.model.fit(X, y)
        self.is_trained = True
    
    def predict_signal(self, df: pd.DataFrame) -> dict:
        """Generate signal."""
        if not self.is_trained:
            return {'signal': 'no_model', 'confidence': 0}
        
        X = self._extract_features(df)
        X_scaled = self.scaler.transform(X)
        
        prediction = self.model.predict(X_scaled)[0]
        probability = self.model.predict_proba(X_scaled)[0]
        
        signal_map = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
        
        return {
            'signal': signal_map.get(prediction, 'HOLD'),
            'confidence': float(max(probability)),
            'probabilities': {signal_map.get(i, 'UNKNOWN'): float(p) for i, p in enumerate(probability)},
        }


def main():
    """Test signal model."""
    model = SignalModel()
    print("Signal model initialized")


if __name__ == "__main__":
    main()

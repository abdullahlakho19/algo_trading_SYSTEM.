# ============================================================================
# MOMENTUM.PY - Momentum and gradient analysis
# ============================================================================

from typing import Optional

import pandas as pd


class MomentumAnalyzer:
    """Analyzes momentum and rate of change."""
    
    @staticmethod
    def calculate_momentum(df: pd.DataFrame, period: int = 12) -> float:
        """Calculate momentum (ROC)."""
        if len(df) < period + 1:
            return 0.0
        
        current = df['Close'].iloc[-1]
        previous = df['Close'].iloc[-period-1]
        
        if previous == 0:
            return 0.0
        
        return ((current - previous) / previous) * 100
    
    @staticmethod
    def calculate_gradient(df: pd.DataFrame, window: int = 5) -> float:
        """Calculate price gradient (slope)."""
        if len(df) < window:
            return 0.0
        
        recent = df['Close'].tail(window).values
        x = list(range(len(recent)))
        y = recent
        
        # Linear regression slope
        n = len(x)
        x_mean = sum(x) / n
        y_mean = sum(y) / n
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return 0.0
        
        slope = numerator / denominator
        return slope
    
    @staticmethod
    def detect_convergence(df: pd.DataFrame) -> str:
        """
        Detect price convergence.
        
        Returns:
            'bullish', 'bearish', or 'neutral'
        """
        if len(df) < 20:
            return 'neutral'
        
        # Calculate multiple momentum indicators
        momentum_12 = MomentumAnalyzer.calculate_momentum(df, 12)
        momentum_26 = MomentumAnalyzer.calculate_momentum(df, 26)
        gradient = MomentumAnalyzer.calculate_gradient(df, 5)
        
        # Divergence = converging (momentum lessening)
        momentum_diff = abs(momentum_12) - abs(momentum_26)
        
        if momentum_diff < 0 and gradient > 0:
            return 'bullish_convergence'
        elif momentum_diff < 0 and gradient < 0:
            return 'bearish_convergence'
        elif momentum_diff > 0:
            return 'divergence'
        
        return 'neutral'
    
    @staticmethod
    def calculate_macd(df: pd.DataFrame) -> dict:
        """Calculate MACD indicator."""
        close = df['Close']
        
        exp12 = close.ewm(span=12, adjust=False).mean()
        exp26 = close.ewm(span=26, adjust=False).mean()
        
        macd = exp12 - exp26
        signal = macd.ewm(span=9, adjust=False).mean()
        histogram = macd - signal
        
        return {
            'macd': float(macd.iloc[-1]),
            'signal': float(signal.iloc[-1]),
            'histogram': float(histogram.iloc[-1]),
        }


def main():
    """Test momentum analysis."""
    print("Momentum analyzer initialized")


if __name__ == "__main__":
    main()

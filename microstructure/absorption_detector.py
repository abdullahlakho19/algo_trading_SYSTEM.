# ============================================================================
# ABSORPTION_DETECTOR.PY - Detects volume absorption (smart money buying/selling)
# ============================================================================

from typing import List

import pandas as pd


class AbsorptionDetector:
    """Detects volume absorption patterns."""
    
    @staticmethod
    def detect_absorption(df: pd.DataFrame, lookback: int = 20) -> dict:
        """
        Detect if volume is being absorbed.
        
        Absorption = price barely moves despite high volume (smart money building position)
        
        Args:
            df: OHLCV DataFrame
            lookback: Bars to analyze
        
        Returns:
            Absorption detection results
        """
        if len(df) < lookback:
            return {}
        
        recent = df.tail(lookback)
        
        # Metrics
        avg_range = ((recent['High'] - recent['Low']) / recent['Close']).mean()
        avg_volume = recent['Volume'].mean()
        current_volume = recent['Volume'].iloc[-1]
        current_range = (recent['High'].iloc[-1] - recent['Low'].iloc[-1]) / recent['Close'].iloc[-1]
        
        # Absorption = high volume + small range
        absorption_score = 0.0
        
        if current_volume > avg_volume * 1.5:
            if current_range < avg_range * 0.5:
                absorption_score = 100.0  # Strong absorption
            elif current_range < avg_range * 0.75:
                absorption_score = 75.0   # Moderate absorption
            else:
                absorption_score = 50.0   # Weak absorption
        
        # Direction
        direction = 'bullish' if recent['Close'].iloc[-1] > recent['Open'].iloc[-1] else 'bearish'
        
        return {
            'absorption_score': absorption_score,
            'current_volume': current_volume,
            'avg_volume': avg_volume,
            'volume_ratio': current_volume / avg_volume,
            'current_range': current_range,
            'avg_range': avg_range,
            'direction': direction,
            'is_absorbing': absorption_score > 50,
        }


def main():
    """Test absorption detection."""
    print("Absorption detector initialized")


if __name__ == "__main__":
    main()

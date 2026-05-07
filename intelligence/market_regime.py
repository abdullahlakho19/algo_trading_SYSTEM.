# ============================================================================
# MARKET_REGIME.PY - Market regime classification (Accum/Distrib/Balance/Trend)
# ============================================================================

from enum import Enum
from typing import Optional

import pandas as pd


class MarketRegime(Enum):
    """Market regime types."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BALANCE = "balance"
    TREND_UP = "trend_up"
    TREND_DOWN = "trend_down"
    BREAKOUT = "breakout"


class RegimeDetector:
    """Detects market regime from price action."""
    
    @staticmethod
    def detect_regime(df: pd.DataFrame, window: int = 50) -> MarketRegime:
        """
        Detect market regime.
        
        Args:
            df: OHLCV DataFrame
            window: Lookback period
        
        Returns:
            MarketRegime classification
        """
        if len(df) < window:
            return MarketRegime.BALANCE
        
        recent = df.tail(window)
        
        # Calculate metrics
        high = recent['High'].max()
        low = recent['Low'].min()
        close_avg = recent['Close'].mean()
        
        # Trend detection
        sma_short = recent['Close'].rolling(5).mean().iloc[-1]
        sma_long = recent['Close'].rolling(20).mean().iloc[-1]
        
        # Volatility
        returns = recent['Close'].pct_change()
        volatility = returns.std()
        
        # Range compression
        range_high = recent['High'].rolling(20).max().iloc[-1]
        range_low = recent['Low'].rolling(20).min().iloc[-1]
        range_pct = (range_high - range_low) / range_low
        
        current = recent['Close'].iloc[-1]
        
        # Classify
        if volatility < returns.mean() * 0.02 and range_pct < 0.05:
            # Low volatility, tight range = accumulation/consolidation
            if current < close_avg:
                return MarketRegime.ACCUMULATION
            else:
                return MarketRegime.DISTRIBUTION
        
        elif sma_short > sma_long and current > sma_short:
            return MarketRegime.TREND_UP
        
        elif sma_short < sma_long and current < sma_short:
            return MarketRegime.TREND_DOWN
        
        elif volatility > returns.mean() * 0.04 and range_pct > 0.10:
            # High volatility potential breakout
            return MarketRegime.BREAKOUT
        
        return MarketRegime.BALANCE
    
    @staticmethod
    def get_regime_characteristics(regime: MarketRegime) -> dict:
        """Get trading characteristics for regime."""
        characteristics = {
            MarketRegime.ACCUMULATION: {
                'description': 'Quiet range, smart money accumulating',
                'strategy': 'mean_reversion, support/resistance',
                'volatility': 'low',
                'activity': 'institutional buying',
            },
            MarketRegime.DISTRIBUTION: {
                'description': 'Quiet range, distribution at resistance',
                'strategy': 'mean_reversion, short setup',
                'volatility': 'low',
                'activity': 'institutional selling',
            },
            MarketRegime.TREND_UP: {
                'description': 'Strong uptrend',
                'strategy': 'momentum, breakout, follow-through',
                'volatility': 'medium',
                'activity': 'buying pressure',
            },
            MarketRegime.TREND_DOWN: {
                'description': 'Strong downtrend',
                'strategy': 'momentum short, breakdown, follow-through',
                'volatility': 'medium',
                'activity': 'selling pressure',
            },
            MarketRegime.BREAKOUT: {
                'description': 'High volatility, breakout potential',
                'strategy': 'breakout trading, momentum',
                'volatility': 'high',
                'activity': 'uncertain',
            },
            MarketRegime.BALANCE: {
                'description': 'Balanced market',
                'strategy': 'range-bound, options',
                'volatility': 'normal',
                'activity': 'mixed',
            },
        }
        return characteristics.get(regime, {})


def main():
    """Test regime detection."""
    detector = RegimeDetector()
    print("Regime detector initialized")


if __name__ == "__main__":
    main()

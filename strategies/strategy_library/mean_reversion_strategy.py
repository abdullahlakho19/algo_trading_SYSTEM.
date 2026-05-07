# ============================================================================
# MEAN_REVERSION_STRATEGY.PY (strategies/strategy_library/)
# ============================================================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class MeanReversionSignal:
    """Mean reversion strategy signal."""
    direction: str
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float


class MeanReversionStrategy:
    """Range-bound mean reversion strategy."""
    
    def __init__(self, bb_period: int = 20, bb_std: float = 2.0):
        """Initialize strategy."""
        self.bb_period = bb_period
        self.bb_std = bb_std
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[MeanReversionSignal]:
        """Generate mean reversion signal."""
        if len(df) < self.bb_period + 1:
            return None
        
        close = df['Close'].values
        recent = close[-self.bb_period:]
        
        sma = pd.Series(recent).mean()
        std = pd.Series(recent).std()
        
        upper_band = sma + (self.bb_std * std)
        lower_band = sma - (self.bb_std * std)
        
        current = close[-1]
        
        if current < lower_band:
            direction = 'BUY'
            confidence = 0.7
        elif current > upper_band:
            direction = 'SELL'
            confidence = 0.7
        else:
            return None
        
        if direction == 'BUY':
            stop_loss = lower_band
            take_profit = sma
        else:
            stop_loss = upper_band
            take_profit = sma
        
        return MeanReversionSignal(
            direction=direction,
            confidence=confidence,
            entry_price=current,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )


def main():
    """Test mean reversion strategy."""
    strategy = MeanReversionStrategy()
    print("Mean reversion strategy initialized")


if __name__ == "__main__":
    main()

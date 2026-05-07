# ============================================================================
# MOMENTUM_STRATEGY.PY (strategies/strategy_library/)
# ============================================================================

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class MomentumSignal:
    """Momentum strategy signal."""
    direction: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float


class MomentumStrategy:
    """Trend-following momentum strategy."""
    
    def __init__(self, fast_ma: int = 12, slow_ma: int = 26):
        """Initialize strategy."""
        self.fast_ma = fast_ma
        self.slow_ma = slow_ma
    
    def generate_signal(self, df: pd.DataFrame) -> Optional[MomentumSignal]:
        """Generate momentum signal."""
        if len(df) < self.slow_ma + 1:
            return None
        
        close = df['Close'].values
        fast = pd.Series(close).ewm(span=self.fast_ma, adjust=False).mean().values[-1]
        slow = pd.Series(close).ewm(span=self.slow_ma, adjust=False).mean().values[-1]
        
        current = close[-1]
        
        if fast > slow and current > fast:
            direction = 'BUY'
            confidence = min((fast - slow) / slow * 100, 100)
        elif fast < slow and current < fast:
            direction = 'SELL'
            confidence = min((slow - fast) / slow * 100, 100)
        else:
            return None
        
        # Calculate stops and targets
        atr = (df['High'] - df['Low']).tail(14).mean()
        
        if direction == 'BUY':
            stop_loss = current - (atr * 2)
            take_profit = current + (atr * 3)
        else:
            stop_loss = current + (atr * 2)
            take_profit = current - (atr * 3)
        
        return MomentumSignal(
            direction=direction,
            confidence=confidence / 100,
            entry_price=current,
            stop_loss=stop_loss,
            take_profit=take_profit,
        )


def main():
    """Test momentum strategy."""
    strategy = MomentumStrategy()
    print("Momentum strategy initialized")


if __name__ == "__main__":
    main()

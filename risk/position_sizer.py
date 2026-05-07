# ============================================================================
# POSITION_SIZER.PY - Volatility-adjusted position sizing
# ============================================================================

from typing import Optional

import pandas as pd


class PositionSizer:
    """Volatility-adjusted position sizing engine."""
    
    def __init__(self, account_size: float, risk_per_trade: float = 0.02):
        """
        Initialize position sizer.
        
        Args:
            account_size: Total trading capital
            risk_per_trade: Risk per trade as % of account (2% default)
        """
        self.account_size = account_size
        self.risk_per_trade = risk_per_trade
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        current_volatility: float = None,
    ) -> dict:
        """
        Calculate position size based on risk.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss level
            current_volatility: Current ATR or volatility
        
        Returns:
            Position sizing info
        """
        # Risk amount
        risk_amount = self.account_size * self.risk_per_trade
        
        # Distance to stop
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return {'error': 'Invalid stop loss'}
        
        # Position size
        position_size = risk_amount / stop_distance
        
        # Adjust for volatility
        if current_volatility:
            vol_adjustment = 1.0 / (1 + (current_volatility / 0.02))  # Normalize to 2% vol
            position_size *= vol_adjustment
        
        return {
            'position_size': round(position_size),
            'risk_amount': risk_amount,
            'stop_distance': stop_distance,
            'position_value': position_size * entry_price,
        }


def main():
    """Test position sizer."""
    sizer = PositionSizer(account_size=100000, risk_per_trade=0.02)
    print("Position sizer initialized")


if __name__ == "__main__":
    main()

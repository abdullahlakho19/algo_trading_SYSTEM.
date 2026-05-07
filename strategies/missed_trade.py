# ============================================================================
# MISSED_TRADE.PY - Missed trade protocol (no-chase rule)
# ============================================================================

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd


class MissedTradeProtocol:
    """
    Enforces no-chase rule: if entry opportunity is missed, don't chase.
    
    Key Rules:
    - Order submitted at calculated entry price
    - If price moves away from entry during order lifetime, cancel order
    - Don't re-submit if first attempt fails
    """
    
    def __init__(self, max_order_lifetime_seconds: int = 300):
        """
        Initialize protocol.
        
        Args:
            max_order_lifetime_seconds: How long to keep order active (5 min default)
        """
        self.max_order_lifetime = max_order_lifetime_seconds
        self.active_orders = {}
    
    def should_cancel_order(
        self,
        order_id: str,
        entry_price: float,
        current_price: float,
        submitted_at: datetime,
        max_adverse_move: float = 0.02,  # 2% tolerance
    ) -> bool:
        """
        Determine if order should be cancelled.
        
        Args:
            order_id: Order identifier
            entry_price: Original entry price
            current_price: Current market price
            submitted_at: When order was submitted
            max_adverse_move: Max adverse price move before cancel
        
        Returns:
            True if order should be cancelled
        """
        # Check order lifetime
        lifetime = (datetime.now() - submitted_at).total_seconds()
        if lifetime > self.max_order_lifetime:
            return True
        
        # Check adverse price move
        adverse_move = abs(current_price - entry_price) / entry_price
        if adverse_move > max_adverse_move:
            return True
        
        return False
    
    def log_missed_trade(self, symbol: str, entry_price: float, reason: str) -> None:
        """Log missed trade attempt."""
        self.active_orders[symbol] = {
            'entry_price': entry_price,
            'missed_at': datetime.now(),
            'reason': reason,
        }


def main():
    """Test missed trade protocol."""
    protocol = MissedTradeProtocol()
    print("Missed trade protocol initialized")


if __name__ == "__main__":
    main()

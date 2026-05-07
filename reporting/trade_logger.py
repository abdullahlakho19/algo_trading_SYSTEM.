# ============================================================================
# TRADE_LOGGER.PY - Real-time trade logging & audit trail
# ============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd


@dataclass
class TradeLog:
    """Trade log entry."""
    timestamp: datetime
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    strategy: str
    status: str  # pending, filled, cancelled
    pnl: Optional[float] = None


class TradeLogger:
    """Real-time trade logging system."""
    
    def __init__(self, log_dir: str = "logs"):
        """Initialize logger."""
        self.log_dir = log_dir
        self.trades: List[TradeLog] = []
    
    def log_trade(
        self,
        symbol: str,
        side: str,
        quantity: int,
        price: float,
        strategy: str,
        status: str = 'pending',
    ) -> None:
        """Log a trade."""
        entry = TradeLog(
            timestamp=datetime.now(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            strategy=strategy,
            status=status,
        )
        self.trades.append(entry)
    
    def export_to_csv(self, filename: str = None) -> str:
        """Export trades to CSV."""
        if not filename:
            filename = f"{self.log_dir}/trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        df = pd.DataFrame([{
            'timestamp': t.timestamp,
            'symbol': t.symbol,
            'side': t.side,
            'quantity': t.quantity,
            'price': t.price,
            'strategy': t.strategy,
            'status': t.status,
            'pnl': t.pnl,
        } for t in self.trades])
        
        df.to_csv(filename, index=False)
        return filename


def main():
    """Test trade logger."""
    logger = TradeLogger()
    print("Trade logger initialized")


if __name__ == "__main__":
    main()

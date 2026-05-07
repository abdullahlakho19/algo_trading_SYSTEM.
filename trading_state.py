# ============================================================================
# TRADING_STATE.PY - Real-time Trading State Management
# Shared data store between bot and dashboard
# ============================================================================

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Single executed trade."""
    trade_id: str
    symbol: str
    side: str  # BUY or SELL
    size: float
    entry_price: float
    entry_time: str
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl: float = 0.0
    pnl_percent: float = 0.0
    duration_minutes: float = 0.0
    strategy: str = "Unknown"
    status: str = "OPEN"  # OPEN, CLOSED
    
    def to_dict(self):
        return asdict(self)


class TradingStateManager:
    """Manages real-time trading state for bot and dashboard."""
    
    STATE_FILE = Path("data/trading_state.json")
    
    @staticmethod
    def ensure_state_file():
        """Create state file if it doesn't exist."""
        TradingStateManager.STATE_FILE.parent.mkdir(exist_ok=True)
        
        if not TradingStateManager.STATE_FILE.exists():
            initial_state = {
                "portfolio": {
                    "initial_capital": 100000.0,
                    "current_equity": 100000.0,
                    "available_cash": 50000.0,
                    "open_pnl": 0.0,
                    "closed_pnl": 0.0,
                    "last_updated": datetime.now().isoformat(),
                },
                "trades": [],
                "positions": [],
            }
            with open(TradingStateManager.STATE_FILE, "w") as f:
                json.dump(initial_state, f, indent=2)
    
    @staticmethod
    def load_state() -> Dict:
        """Load current trading state."""
        TradingStateManager.ensure_state_file()
        try:
            with open(TradingStateManager.STATE_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            logger.error(f"Failed to load state: {e}")
            return None
    
    @staticmethod
    def save_state(state: Dict):
        """Save trading state."""
        TradingStateManager.ensure_state_file()
        try:
            with open(TradingStateManager.STATE_FILE, "w") as f:
                json.dump(state, f, indent=2)
                logger.info("State saved")
        except IOError as e:
            logger.error(f"Failed to save state: {e}")
    
    @staticmethod
    def record_trade(trade: Trade):
        """Record a new trade."""
        state = TradingStateManager.load_state()
        state["trades"].append(trade.to_dict())
        TradingStateManager.save_state(state)
        logger.info(f"Trade recorded: {trade.trade_id}")
    
    @staticmethod
    def close_trade(trade_id: str, exit_price: float, exit_time: str):
        """Close an open trade."""
        state = TradingStateManager.load_state()
        
        for trade in state["trades"]:
            if trade["trade_id"] == trade_id and trade["status"] == "OPEN":
                trade["exit_price"] = exit_price
                trade["exit_time"] = exit_time
                trade["status"] = "CLOSED"
                
                # Calculate P&L
                if trade["side"] == "BUY":
                    pnl = (exit_price - trade["entry_price"]) * trade["size"]
                else:  # SELL
                    pnl = (trade["entry_price"] - exit_price) * trade["size"]
                
                trade["pnl"] = pnl
                trade["pnl_percent"] = (pnl / (trade["entry_price"] * trade["size"])) * 100 if trade["entry_price"] > 0 else 0
                
                # Update portfolio
                state["portfolio"]["closed_pnl"] += pnl
                state["portfolio"]["current_equity"] += pnl
                
                break
        
        state["portfolio"]["last_updated"] = datetime.now().isoformat()
        TradingStateManager.save_state(state)
        logger.info(f"Trade closed: {trade_id}")
    
    @staticmethod
    def update_portfolio_metrics(open_pnl: float, cash: float):
        """Update portfolio metrics."""
        state = TradingStateManager.load_state()
        state["portfolio"]["open_pnl"] = open_pnl
        state["portfolio"]["available_cash"] = cash
        state["portfolio"]["current_equity"] = state["portfolio"]["initial_capital"] - cash + state["portfolio"]["closed_pnl"] + open_pnl
        state["portfolio"]["last_updated"] = datetime.now().isoformat()
        TradingStateManager.save_state(state)
    
    @staticmethod
    def get_recent_trades(limit: int = 10) -> List[Trade]:
        """Get recent trades."""
        state = TradingStateManager.load_state()
        if not state or not state.get("trades"):
            return []
        
        trades = [Trade(**t) for t in state["trades"]]
        return sorted(trades, key=lambda t: t.entry_time, reverse=True)[:limit]
    
    @staticmethod
    def get_portfolio():
        """Get portfolio summary."""
        state = TradingStateManager.load_state()
        return state.get("portfolio", {}) if state else {}


if __name__ == "__main__":
    # Initialize state
    TradingStateManager.ensure_state_file()
    
    # Example: Record a trade
    trade = Trade(
        trade_id="TRADE_001",
        symbol="AAPL",
        side="BUY",
        size=100,
        entry_price=150.25,
        entry_time=datetime.now().isoformat(),
        strategy="Momentum",
    )
    TradingStateManager.record_trade(trade)
    
    # Close the trade
    TradingStateManager.close_trade(
        "TRADE_001",
        exit_price=152.50,
        exit_time=datetime.now().isoformat(),
    )
    
    # Get portfolio
    portfolio = TradingStateManager.get_portfolio()
    print(f"Portfolio: {portfolio}")
    
    # Get recent trades
    trades = TradingStateManager.get_recent_trades()
    for trade in trades:
        print(f"{trade.trade_id}: {trade.symbol} {trade.side} {trade.size} @ {trade.entry_price}")

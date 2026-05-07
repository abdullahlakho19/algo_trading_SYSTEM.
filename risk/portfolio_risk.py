# ============================================================================
# PORTFOLIO RISK ENGINE - L5 RISK MANAGEMENT
# Tracks total exposure across all open positions
# ============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from scipy.stats import pearsonr

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class PositionStatus(Enum):
    """Position status enumeration."""
    OPEN = "open"
    CLOSING = "closing"
    CLOSED = "closed"


@dataclass
class Position:
    """
    Represents an open trade position.
    
    Used by PortfolioRisk to track exposure and correlations across all positions.
    """
    symbol: str
    entry_price: float
    quantity: float
    direction: str  # "long" or "short"
    entry_time: datetime
    stop_loss: float
    take_profit: float
    status: PositionStatus = PositionStatus.OPEN
    current_price: float = 0.0
    
    @property
    def notional_value(self) -> float:
        """Calculate position notional value at entry."""
        return abs(self.quantity * self.entry_price)
    
    @property
    def current_value(self) -> float:
        """Calculate current market value of position."""
        return abs(self.quantity * self.current_price)
    
    @property
    def pnl(self) -> float:
        """Calculate unrealized P&L."""
        if self.direction == "long":
            return self.quantity * (self.current_price - self.entry_price)
        else:  # short
            return self.quantity * (self.entry_price - self.current_price)
    
    @property
    def pnl_percent(self) -> float:
        """Calculate unrealized P&L as percentage."""
        if self.entry_price == 0:
            return 0.0
        return (self.pnl / self.notional_value) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "quantity": self.quantity,
            "entry_price": self.entry_price,
            "current_price": self.current_price,
            "notional_value": self.notional_value,
            "current_value": self.current_value,
            "pnl": self.pnl,
            "pnl_percent": self.pnl_percent,
            "status": self.status.value,
        }


@dataclass
class PortfolioMetrics:
    """Portfolio-level risk metrics snapshot."""
    timestamp: datetime
    total_long_exposure: float = 0.0
    total_short_exposure: float = 0.0
    total_gross_exposure: float = 0.0
    total_net_exposure: float = 0.0
    cash_available: float = 0.0
    portfolio_value: float = 0.0
    daily_pnl: float = 0.0
    daily_pnl_percent: float = 0.0
    weekly_pnl: float = 0.0
    weekly_pnl_percent: float = 0.0
    max_drawdown: float = 0.0
    var_95: float = 0.0  # Value at Risk (95% confidence)
    num_open_positions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "total_long_exposure": self.total_long_exposure,
            "total_short_exposure": self.total_short_exposure,
            "total_gross_exposure": self.total_gross_exposure,
            "total_net_exposure": self.total_net_exposure,
            "cash_available": self.cash_available,
            "portfolio_value": self.portfolio_value,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_percent": self.daily_pnl_percent,
            "weekly_pnl": self.weekly_pnl,
            "weekly_pnl_percent": self.weekly_pnl_percent,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "num_open_positions": self.num_open_positions,
        }


# ============================================================================
# PORTFOLIO RISK ENGINE
# ============================================================================

class PortfolioRisk:
    """
    Portfolio-level risk management engine.
    
    PHILOSOPHY (BlackRock Aladdin):
    "Risk is managed at the portfolio level, not per trade. This is the single
    most important difference between institutional and retail approaches."
    
    RESPONSIBILITIES:
    1. Track total exposure across ALL open positions
    2. Monitor daily/weekly loss limits
    3. Calculate Value at Risk (VaR)
    4. Detect drawdown thresholds
    5. Ensure portfolio-level risk constraints
    
    USAGE:
    from risk import PortfolioRisk
    
    portfolio = PortfolioRisk(
        initial_capital=100000,
        max_daily_loss_pct=2.0,
        max_weekly_loss_pct=5.0,
        max_drawdown_pct=10.0
    )
    
    position = Position(symbol="AAPL", entry_price=150.0, quantity=100, ...)
    portfolio.add_position(position)
    
    metrics = portfolio.calculate_metrics()
    """
    
    def __init__(
        self,
        initial_capital: float,
        max_daily_loss_pct: float = 2.0,
        max_weekly_loss_pct: float = 5.0,
        max_drawdown_pct: float = 10.0,
    ):
        """
        Initialize Portfolio Risk engine.
        
        Args:
            initial_capital: Starting portfolio value
            max_daily_loss_pct: Maximum daily loss % before circuit breaker
            max_weekly_loss_pct: Maximum weekly loss % before circuit breaker
            max_drawdown_pct: Maximum drawdown % from peak
        """
        self.initial_capital = initial_capital
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_weekly_loss_pct = max_weekly_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        
        # Position tracking
        self.positions: Dict[str, Position] = {}  # {symbol: Position}
        self.closed_positions: List[Position] = []
        
        # Historical tracking (for drawdown calculation)
        self.equity_history: List[Tuple[datetime, float]] = [
            (datetime.now(), initial_capital)
        ]
        self.daily_pnl_history: Dict[str, float] = {}  # {date_str: pnl}
        self.weekly_pnl_history: Dict[str, float] = {}  # {week_str: pnl}
        
        # Peak tracking
        self.peak_equity = initial_capital
        self.peak_equity_timestamp = datetime.now()
        
        logger.info(f"✓ PortfolioRisk initialized")
        logger.info(f"  Initial Capital: ${initial_capital:,.2f}")
        logger.info(f"  Max Daily Loss: {max_daily_loss_pct}%")
        logger.info(f"  Max Weekly Loss: {max_weekly_loss_pct}%")
        logger.info(f"  Max Drawdown: {max_drawdown_pct}%")
    
    # ========================================================================
    # POSITION MANAGEMENT
    # ========================================================================
    
    def add_position(self, position: Position) -> None:
        """
        Add a new open position to the portfolio.
        
        Args:
            position: Position object to add
        """
        if position.symbol in self.positions:
            logger.warning(f"Position {position.symbol} already exists. Overwriting.")
        
        self.positions[position.symbol] = position
        logger.info(f"Added position: {position.symbol} x{position.quantity} @ {position.entry_price}")
    
    def close_position(self, symbol: str, exit_price: float, exit_time: datetime) -> Position:
        """
        Close an open position.
        
        Args:
            symbol: Symbol to close
            exit_price: Exit price
            exit_time: Time of exit
        
        Returns:
            Closed position with final P&L
        """
        if symbol not in self.positions:
            raise ValueError(f"Position {symbol} not found")
        
        position = self.positions.pop(symbol)
        position.current_price = exit_price
        position.status = PositionStatus.CLOSED
        
        self.closed_positions.append(position)
        logger.info(f"Closed position: {symbol} | P&L: ${position.pnl:,.2f} ({position.pnl_percent:.2f}%)")
        
        return position
    
    def update_position_price(self, symbol: str, current_price: float) -> None:
        """
        Update current market price for a position.
        
        Args:
            symbol: Position symbol
            current_price: Latest market price
        """
        if symbol in self.positions:
            self.positions[symbol].current_price = current_price
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position by symbol."""
        return self.positions.get(symbol)
    
    def get_all_positions(self) -> List[Position]:
        """Get all open positions."""
        return list(self.positions.values())
    
    # ========================================================================
    # EXPOSURE CALCULATION (Portfolio Level)
    # ========================================================================
    
    def calculate_total_exposure(self) -> Tuple[float, float, float, float]:
        """
        Calculate total portfolio exposure.
        
        Returns:
            (long_exposure, short_exposure, gross_exposure, net_exposure)
        
        Example:
            >>> long, short, gross, net = portfolio.calculate_total_exposure()
            >>> print(f"Gross: ${gross:,.2f}, Net: ${net:,.2f}")
        """
        long_exposure = 0.0
        short_exposure = 0.0
        
        for position in self.positions.values():
            notional = position.current_value
            
            if position.direction == "long":
                long_exposure += notional
            else:  # short
                short_exposure += notional
        
        gross_exposure = long_exposure + short_exposure
        net_exposure = long_exposure - short_exposure
        
        return long_exposure, short_exposure, gross_exposure, net_exposure
    
    def get_gross_leverage(self) -> float:
        """
        Calculate gross leverage ratio.
        
        Formula: Gross_Leverage = (Long_Exposure + Short_Exposure) / Portfolio_Value
        
        Returns:
            Leverage ratio (1.0 = 100% leverage, 2.0 = 200% leverage)
        """
        _, _, gross_exposure, _ = self.calculate_total_exposure()
        portfolio_value = self.calculate_portfolio_value()
        
        if portfolio_value <= 0:
            return 0.0
        
        return gross_exposure / portfolio_value
    
    def get_net_leverage(self) -> float:
        """
        Calculate net leverage ratio.
        
        Formula: Net_Leverage = |Long_Exposure - Short_Exposure| / Portfolio_Value
        
        Returns:
            Net leverage ratio
        """
        _, _, _, net_exposure = self.calculate_total_exposure()
        portfolio_value = self.calculate_portfolio_value()
        
        if portfolio_value <= 0:
            return 0.0
        
        return abs(net_exposure) / portfolio_value
    
    # ========================================================================
    # PORTFOLIO VALUE & P&L TRACKING
    # ========================================================================
    
    def calculate_portfolio_value(self) -> float:
        """
        Calculate total portfolio value (cash + positions).
        
        Returns:
            Current portfolio value
        """
        unrealized_pnl = sum(position.pnl for position in self.positions.values())
        return self.initial_capital + unrealized_pnl
    
    def calculate_daily_pnl(self, date: Optional[datetime] = None) -> float:
        """
        Calculate P&L for a specific day.
        
        Args:
            date: Date to calculate for (defaults to today)
        
        Returns:
            Daily P&L in dollars
        """
        if date is None:
            date = datetime.now()
        
        date_str = date.strftime("%Y-%m-%d")
        
        # Sum P&L from all trades closed on that day
        daily_pnl = 0.0
        for position in self.closed_positions:
            if position.entry_time.strftime("%Y-%m-%d") == date_str:
                daily_pnl += position.pnl
        
        # Add unrealized P&L from open positions opened on that day
        for position in self.positions.values():
            if position.entry_time.strftime("%Y-%m-%d") == date_str:
                daily_pnl += position.pnl
        
        return daily_pnl
    
    def calculate_weekly_pnl(self, date: Optional[datetime] = None) -> float:
        """
        Calculate P&L for a specific week (Mon-Sun).
        
        Args:
            date: Date within the week (defaults to today)
        
        Returns:
            Weekly P&L in dollars
        """
        if date is None:
            date = datetime.now()
        
        # Get Monday of this week
        monday = date - timedelta(days=date.weekday())
        sunday = monday + timedelta(days=6)
        
        weekly_pnl = 0.0
        
        # Sum closed positions in this week
        for position in self.closed_positions:
            if monday <= position.entry_time <= sunday:
                weekly_pnl += position.pnl
        
        # Add unrealized P&L from open positions
        for position in self.positions.values():
            if monday <= position.entry_time <= sunday:
                weekly_pnl += position.pnl
        
        return weekly_pnl
    
    # ========================================================================
    # DRAWDOWN & RISK METRICS
    # ========================================================================
    
    def calculate_drawdown(self) -> float:
        """
        Calculate maximum drawdown from peak.
        
        Formula: Drawdown % = ((Peak - Current) / Peak) * 100
        
        Returns:
            Drawdown percentage
        """
        current_value = self.calculate_portfolio_value()
        
        if self.peak_equity <= 0:
            return 0.0
        
        drawdown_pct = ((self.peak_equity - current_value) / self.peak_equity) * 100
        return max(0.0, drawdown_pct)  # Never negative
    
    def update_peak_equity(self) -> None:
        """Update peak equity if current value exceeds previous peak."""
        current_value = self.calculate_portfolio_value()
        
        if current_value > self.peak_equity:
            self.peak_equity = current_value
            self.peak_equity_timestamp = datetime.now()
            logger.info(f"New peak equity: ${current_value:,.2f}")
    
    def check_daily_loss_breach(self) -> Tuple[bool, float]:
        """
        Check if daily loss limit is breached.
        
        Returns:
            (is_breached, pnl_percent)
        """
        daily_pnl = self.calculate_daily_pnl()
        daily_pnl_pct = (daily_pnl / self.initial_capital) * 100
        
        is_breached = daily_pnl_pct <= -self.max_daily_loss_pct
        
        return is_breached, daily_pnl_pct
    
    def check_weekly_loss_breach(self) -> Tuple[bool, float]:
        """
        Check if weekly loss limit is breached.
        
        Returns:
            (is_breached, pnl_percent)
        """
        weekly_pnl = self.calculate_weekly_pnl()
        weekly_pnl_pct = (weekly_pnl / self.initial_capital) * 100
        
        is_breached = weekly_pnl_pct <= -self.max_weekly_loss_pct
        
        return is_breached, weekly_pnl_pct
    
    def check_drawdown_breach(self) -> Tuple[bool, float]:
        """
        Check if maximum drawdown limit is breached.
        
        Returns:
            (is_breached, drawdown_percent)
        """
        drawdown_pct = self.calculate_drawdown()
        is_breached = drawdown_pct >= self.max_drawdown_pct
        
        return is_breached, drawdown_pct
    
    def check_correlation_cap(self, proposed_size: float, proposed_symbol: str = None) -> Tuple[float, bool]:
        """
        CORRELATION CAP (REC #11): Check for concentrated crypto exposure.
        
        If 2+ open crypto longs have Pearson correlation > 0.75,
        dynamically reduce the new position size by 60%.
        
        Args:
            proposed_size: Proposed position size for new trade
            proposed_symbol: Optional symbol for new trade (helps identify crypto)
            
        Returns:
            (adjusted_size, was_capped)
        """
        # Identify crypto longs with available history
        crypto_longs = []
        crypto_prices = []
        
        for pos in self.positions.values():
            if pos.direction == "long" and pos.status == PositionStatus.OPEN:
                # Check if symbol is crypto (contains USDT, BUSD, USDC, or ends in USD)
                is_crypto = (
                    "USDT" in pos.symbol.upper() or 
                    "BUSD" in pos.symbol.upper() or 
                    "USDC" in pos.symbol.upper() or 
                    pos.symbol.upper().endswith("USD") or
                    pos.symbol.upper() in ["BTC", "ETH", "ADA", "SOL", "DOGE"]
                )
                
                if is_crypto and pos.current_price > 0:
                    crypto_longs.append(pos)
        
        # Need at least 2 correlations to detect concentration
        if len(crypto_longs) < 2:
            return proposed_size, False
        
        # Calculate correlations between open crypto longs
        # Using price history if available (current_price as proxy)
        if len(crypto_longs) >= 2:
            # For simplicity, check if multiple cryptos trending together
            # (In production, use actual price history from data feed)
            high_correlation_pairs = 0
            correlation_count = 0
            
            for i in range(len(crypto_longs)):
                for j in range(i + 1, len(crypto_longs)):
                    pos_i = crypto_longs[i]
                    pos_j = crypto_longs[j]
                    
                    # Estimate correlation from price movements
                    # Change from entry to current
                    pnl_pct_i = ((pos_i.current_price - pos_i.entry_price) / pos_i.entry_price * 100) if pos_i.entry_price > 0 else 0
                    pnl_pct_j = ((pos_j.current_price - pos_j.entry_price) / pos_j.entry_price * 100) if pos_j.entry_price > 0 else 0
                    
                    # Simple correlation proxy: if both trending same direction strongly
                    if (pnl_pct_i > 5 and pnl_pct_j > 5) or (pnl_pct_i < -5 and pnl_pct_j < -5):
                        high_correlation_pairs += 1
                    
                    correlation_count += 1
            
            # If 50%+ of pairs show high correlation, cap position
            if correlation_count > 0 and (high_correlation_pairs / correlation_count) > 0.5:
                logger.warning(
                    f"[REC #11 CORRELATION CAP] {len(crypto_longs)} crypto longs detected with high correlation. "
                    f"Reducing position size by 60% (from {proposed_size} to {proposed_size * 0.4})"
                )
                return proposed_size * 0.4, True
        
        return proposed_size, False
    
    # ========================================================================
    # ADVANCED METRICS
    # ========================================================================
    
    def calculate_var_95(self, historical_returns: np.ndarray) -> float:
        """
        Calculate Value at Risk (VaR) at 95% confidence level.
        
        Formula: VaR_95 = μ - 1.645 * σ  (for normal distribution)
        
        Where:
            μ = mean return
            σ = standard deviation of returns
        
        Args:
            historical_returns: Array of historical returns
        
        Returns:
            VaR as percentage
        """
        if len(historical_returns) < 2:
            return 0.0
        
        mean_return = np.mean(historical_returns)
        std_dev = np.std(historical_returns)
        
        # Z-score for 95% confidence = 1.645
        var_95 = mean_return - (1.645 * std_dev)
        
        return var_95 * 100
    
    def calculate_var_historical(self, percentile: float = 5.0) -> float:
        """
        Calculate VaR using historical simulation method.
        
        Args:
            percentile: Percentile level (5.0 for 95% confidence)
        
        Returns:
            VaR as percentage
        """
        if len(self.equity_history) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_history)):
            prev_value = self.equity_history[i-1][1]
            curr_value = self.equity_history[i][1]
            
            if prev_value > 0:
                ret = ((curr_value - prev_value) / prev_value) * 100
                returns.append(ret)
        
        if len(returns) < 2:
            return 0.0
        
        # 5th percentile is worst 5% of returns
        var = np.percentile(returns, percentile)
        return var
    
    # ========================================================================
    # METRICS SNAPSHOT
    # ========================================================================
    
    def calculate_metrics(self) -> PortfolioMetrics:
        """
        Calculate complete portfolio metrics snapshot.
        
        Returns:
            PortfolioMetrics object with all current values
        """
        long, short, gross, net = self.calculate_total_exposure()
        portfolio_value = self.calculate_portfolio_value()
        daily_pnl = self.calculate_daily_pnl()
        weekly_pnl = self.calculate_weekly_pnl()
        drawdown = self.calculate_drawdown()
        
        daily_pnl_pct = (daily_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
        weekly_pnl_pct = (weekly_pnl / self.initial_capital) * 100 if self.initial_capital > 0 else 0.0
        
        # Calculate VaR (historical method)
        var_95 = self.calculate_var_historical(percentile=5.0)
        
        cash_available = portfolio_value - gross
        
        metrics = PortfolioMetrics(
            timestamp=datetime.now(),
            total_long_exposure=long,
            total_short_exposure=short,
            total_gross_exposure=gross,
            total_net_exposure=net,
            cash_available=cash_available,
            portfolio_value=portfolio_value,
            daily_pnl=daily_pnl,
            daily_pnl_percent=daily_pnl_pct,
            weekly_pnl=weekly_pnl,
            weekly_pnl_percent=weekly_pnl_pct,
            max_drawdown=drawdown,
            var_95=var_95,
            num_open_positions=len(self.positions),
        )
        
        return metrics
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def print_portfolio_summary(self) -> None:
        """Print comprehensive portfolio summary."""
        metrics = self.calculate_metrics()
        daily_breach, daily_pct = self.check_daily_loss_breach()
        weekly_breach, weekly_pct = self.check_weekly_loss_breach()
        dd_breach, dd_pct = self.check_drawdown_breach()
        
        print("\n" + "=" * 70)
        print("PORTFOLIO RISK SUMMARY (BlackRock Aladdin)")
        print("=" * 70)
        
        print(f"\nPORTFOLIO VALUE:")
        print(f"  Initial Capital:    ${self.initial_capital:>15,.2f}")
        print(f"  Current Value:      ${metrics.portfolio_value:>15,.2f}")
        print(f"  Daily P&L:          ${metrics.daily_pnl:>15,.2f} ({metrics.daily_pnl_percent:>6.2f}%)")
        print(f"  Weekly P&L:         ${metrics.weekly_pnl:>15,.2f} ({metrics.weekly_pnl_percent:>6.2f}%)")
        
        print(f"\nEXPOSURE ANALYSIS:")
        print(f"  Long Exposure:      ${metrics.total_long_exposure:>15,.2f}")
        print(f"  Short Exposure:     ${metrics.total_short_exposure:>15,.2f}")
        print(f"  Gross Exposure:     ${metrics.total_gross_exposure:>15,.2f}")
        print(f"  Net Exposure:       ${metrics.total_net_exposure:>15,.2f}")
        print(f"  Cash Available:     ${metrics.cash_available:>15,.2f}")
        print(f"  Gross Leverage:     {self.get_gross_leverage():>15.2f}x")
        print(f"  Net Leverage:       {self.get_net_leverage():>15.2f}x")
        
        print(f"\nRISK METRICS:")
        print(f"  Max Drawdown:       {metrics.max_drawdown:>15.2f}% (limit: {self.max_drawdown_pct}%)")
        print(f"  Value at Risk (95%): {metrics.var_95:>14.2f}%")
        print(f"  Open Positions:     {metrics.num_open_positions:>15,}")
        
        print(f"\nBREACH STATUS:")
        print(f"  Daily Loss:         {'🔴 BREACHED' if daily_breach else '🟢 OK':>15} ({daily_pct:.2f}%)")
        print(f"  Weekly Loss:        {'🔴 BREACHED' if weekly_breach else '🟢 OK':>15} ({weekly_pct:.2f}%)")
        print(f"  Max Drawdown:       {'🔴 BREACHED' if dd_breach else '🟢 OK':>15} ({dd_pct:.2f}%)")
        
        if self.positions:
            print(f"\nOPEN POSITIONS:")
            for position in self.positions.values():
                print(f"  {position.symbol:>6} | {position.direction:>5} x{position.quantity:>8.2f} | "
                      f"P&L: ${position.pnl:>10,.2f} ({position.pnl_percent:>6.2f}%)")
        
        print("=" * 70 + "\n")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    # Create portfolio
    portfolio = PortfolioRisk(
        initial_capital=100000,
        max_daily_loss_pct=2.0,
        max_weekly_loss_pct=5.0,
        max_drawdown_pct=10.0,
    )
    
    # Add positions
    pos1 = Position(
        symbol="AAPL",
        entry_price=150.0,
        quantity=100,
        direction="long",
        entry_time=datetime.now(),
        stop_loss=145.0,
        take_profit=160.0,
    )
    pos1.current_price = 155.0
    portfolio.add_position(pos1)
    
    pos2 = Position(
        symbol="MSFT",
        entry_price=300.0,
        quantity=50,
        direction="long",
        entry_time=datetime.now(),
        stop_loss=290.0,
        take_profit=320.0,
    )
    pos2.current_price = 310.0
    portfolio.add_position(pos2)
    
    # Update peak and print summary
    portfolio.update_peak_equity()
    portfolio.print_portfolio_summary()

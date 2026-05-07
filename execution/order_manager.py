# ============================================================================
# ORDER_MANAGER.PY - Master Order Manager with Missed Trade Protocol
# Enforces "never chase price" via intelligent order cancellation
# ============================================================================

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class TradeState(str, Enum):
    """Trade state enumeration."""
    PENDING = "PENDING"
    ORDER_SUBMITTED = "ORDER_SUBMITTED"
    ORDER_FILLED = "ORDER_FILLED"
    ORDER_CANCELLED = "ORDER_CANCELLED"
    MISSED_TRADE = "MISSED_TRADE"
    ERROR = "ERROR"


class CancellationReason(str, Enum):
    """Order cancellation reason."""
    PRICE_MOVED_AWAY = "PRICE_MOVED_AWAY"
    MISSED_TRADE_PROTOCOL = "MISSED_TRADE_PROTOCOL"
    USER_CANCEL = "USER_CANCEL"
    TIMEOUT = "TIMEOUT"
    STRATEGY_EXIT = "STRATEGY_EXIT"
    RISK_MANAGEMENT = "RISK_MANAGEMENT"


@dataclass
class TradeOrder:
    """Single trade order record."""
    order_id: str
    signal_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    entry_price: float  # Limit price we want
    tp_price: Optional[float] = None
    sl_price: Optional[float] = None
    state: TradeState = TradeState.PENDING
    
    # Submission tracking
    submitted_at: Optional[datetime] = None
    submitted_bid: float = 0.0  # Market bid when submitted
    submitted_ask: float = 0.0  # Market ask when submitted
    
    # Fill tracking
    filled_at: Optional[datetime] = None
    filled_price: float = 0.0
    filled_quantity: float = 0.0
    
    # Cancellation tracking
    cancelled_at: Optional[datetime] = None
    cancellation_reason: Optional[CancellationReason] = None
    
    # Pricing history
    price_checks: List[Tuple[datetime, float, float]] = field(default_factory=list)
    max_price_deviation: float = 0.0
    
    # Execution metadata
    slippage_realized: float = 0.0
    order_lifetime_seconds: float = 0.0


@dataclass
class MissedTradeLog:
    """Log entry for missed trades."""
    timestamp: datetime
    signal_id: str
    symbol: str
    side: str
    quantity: float
    entry_price: float
    reason: CancellationReason
    deviation_pct: float
    order_lifetime_seconds: float


class MissedTradeProtocol:
    """
    Core protocol: enforce strict order management.
    
    Rules:
    1. If entry price MUST be at or better than limit price
    2. If market moves AWAY from entry, cancel order (don't chase)
    3. Track deviations and missed trade statistics
    4. Never retry same entry price (re-signal required)
    
    Prevents:
    - Chasing price during drawdowns
    - Revenge trading / escalating losses
    - Overexposure from multiple retries
    """
    
    def __init__(
        self,
        max_order_lifetime_seconds: int = 300,  # 5 minutes
        price_deviation_tolerance_bps: float = 10.0,  # 10 basis points before cancel
        check_interval_seconds: float = 1.0,
    ):
        """
        Initialize Missed Trade Protocol.
        
        Args:
            max_order_lifetime_seconds: Max time order can stay alive
            price_deviation_tolerance_bps: How far price can move before cancel
            check_interval_seconds: How often to check price
        """
        self.max_order_lifetime_seconds = max_order_lifetime_seconds
        self.price_deviation_tolerance_bps = price_deviation_tolerance_bps / 10000
        self.check_interval_seconds = check_interval_seconds
    
    async def should_cancel_order(
        self,
        order: TradeOrder,
        current_bid: float,
        current_ask: float,
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if order should be cancelled based on price movement.
        
        Cancellation triggers:
        1. Price moved away from entry (no longer achievable)
        2. Order exceeded max lifetime
        3. Bid-ask both moved against us
        
        Args:
            order: Trade order to evaluate
            current_bid: Current market bid
            current_ask: Current market ask
            
        Returns:
            (should_cancel, reason)
        """
        if order.state != TradeState.ORDER_SUBMITTED:
            return False, None
        
        if order.submitted_at is None:
            return False, None
        
        # Check 1: Order timeout
        lifetime = (datetime.utcnow() - order.submitted_at).total_seconds()
        if lifetime > self.max_order_lifetime_seconds:
            return True, "Order max lifetime exceeded"
        
        # Check 2: Price moved away from entry
        if order.side == "BUY":
            # We want to buy at entry_price. If current ask > entry_price,
            # we can no longer fill at our target. CANCEL.
            if current_ask > order.entry_price * (1 + self.price_deviation_tolerance_bps):
                deviation = (current_ask - order.entry_price) / order.entry_price
                order.max_price_deviation = max(order.max_price_deviation, abs(deviation))
                return True, f"Ask moved away: {current_ask:.4f} > target {order.entry_price:.4f}"
        
        else:  # SELL
            # We want to sell at entry_price. If current bid < entry_price,
            # we can no longer fill at our target. CANCEL.
            if current_bid < order.entry_price * (1 - self.price_deviation_tolerance_bps):
                deviation = (order.entry_price - current_bid) / order.entry_price
                order.max_price_deviation = max(order.max_price_deviation, abs(deviation))
                return True, f"Bid moved away: {current_bid:.4f} < target {order.entry_price:.4f}"
        
        return False, None


class OrderManager:
    """
    Master order manager.
    
    Responsibilities:
    1. Submit orders via configured executor (Alpaca, Paper) - LIMIT ONLY
    2. Monitor order state and prices
    3. Enforce Missed Trade Protocol
    4. Track fills and executions
    5. Cancel orders that miss entry price
    6. Maintain trade ledger and statistics
    7. Prevent order chasing and revenge trading
    
    Features:
    - Real-time price monitoring
    - Automatic order cancellation on price deviation
    - Bracket order management (entry + TP + SL)
    - Trade lifecycle tracking
    - Comprehensive audit trail
    - Statistics on missed trades
    
    CRITICAL (REC #13): ALL ORDERS ARE LIMIT ORDERS ONLY
    - No market order fallback under any circumstances
    - Entry price is locked exactly at calculated entry_price
    - This eliminates slippage and enforces price discipline
    - If limit order is missed, trade is rejected (Missed Trade Protocol)
    
    Designed for:
    - Production execution with strict risk control
    - Institutional trading compliance
    - Preventing emotional trading decisions
    - Audit and regulatory reporting
    - Zero-slippage entry execution
    - Stocks and Forex only (Crypto removed)
    """
    
    def __init__(
        self,
        executor_type: str = "paper",  # 'paper', 'alpaca'
        executor = None,
        protocol: Optional[MissedTradeProtocol] = None,
    ):
        """
        Initialize Order Manager.
        
        Args:
            executor_type: Type of executor ('paper' or 'alpaca')
            executor: Executor instance
            protocol: MissedTradeProtocol instance
        """
        self.executor_type = executor_type
        self.executor = executor
        self.protocol = protocol or MissedTradeProtocol()
        
        self.orders: Dict[str, TradeOrder] = {}
        self.trades: Dict[str, TradeOrder] = {}
        self.missed_trades: List[MissedTradeLog] = []
        
        self.price_monitor_tasks: Dict[str, asyncio.Task] = {}
        self.bracket_manager_tasks: Dict[str, asyncio.Task] = {}
        
        logger.info(
            f"✓ OrderManager initialized: executor={executor_type}, "
            f"protocol_lifetime={protocol.max_order_lifetime_seconds}s"
        )
    
    async def submit_trade_order(
        self,
        signal_id: str,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        tp_price: Optional[float] = None,
        sl_price: Optional[float] = None,
        current_bid: float = 0.0,
        current_ask: float = 0.0,
    ) -> Optional[TradeOrder]:
        """
        Submit trade order with Missed Trade Protocol checks.
        
        REC #13 (LIMIT ORDERS ONLY):
        - All orders MUST be submitted as limit orders locked at entry_price
        - NO market order fallback is permitted
        - If limit order cannot be filled, order is REJECTED via Missed Trade Protocol
        - This ensures exact entry price execution with zero slippage
        
        Args:
            signal_id: Signal ID from strategy
            symbol: Trading symbol
            side: 'BUY' or 'SELL'
            quantity: Order size
            entry_price: Desired entry price (LIMIT PRICE - NON-NEGOTIABLE)
            tp_price: Take profit price (optional)
            sl_price: Stop loss price (optional)
            current_bid: Current market bid
            current_ask: Current market ask
            
        Returns:
            TradeOrder object or None if rejected
        """
        # Create order record
        order_id = f"{signal_id}_{symbol}_{datetime.utcnow().timestamp()}"
        order = TradeOrder(
            order_id=order_id,
            signal_id=signal_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            tp_price=tp_price,
            sl_price=sl_price,
        )
        
        # Check if entry is achievable Right NOW
        if side == "BUY":
            if current_ask > entry_price * 1.001:  # Ask is > our entry (even with tiny slippage)
                # Entry price is not achievable right now
                logger.warning(
                    f"⚠️  Order REJECTED: {order_id} | entry price not achievable | "
                    f"ask {current_ask:.4f} > entry {entry_price:.4f}"
                )
                order.state = TradeState.MISSED_TRADE
                order.cancellation_reason = CancellationReason.PRICE_MOVED_AWAY
                self.missed_trades.append(MissedTradeLog(
                    timestamp=datetime.utcnow(),
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=entry_price,
                    reason=CancellationReason.PRICE_MOVED_AWAY,
                    deviation_pct=((current_ask - entry_price) / entry_price) * 100,
                    order_lifetime_seconds=0,
                ))
                return None
        
        else:  # SELL
            if current_bid < entry_price * 0.999:  # Bid is < our entry
                logger.warning(
                    f"⚠️  Order REJECTED: {order_id} | entry price not achievable | "
                    f"bid {current_bid:.4f} < entry {entry_price:.4f}"
                )
                order.state = TradeState.MISSED_TRADE
                order.cancellation_reason = CancellationReason.PRICE_MOVED_AWAY
                self.missed_trades.append(MissedTradeLog(
                    timestamp=datetime.utcnow(),
                    signal_id=signal_id,
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    entry_price=entry_price,
                    reason=CancellationReason.PRICE_MOVED_AWAY,
                    deviation_pct=((entry_price - current_bid) / entry_price) * 100,
                    order_lifetime_seconds=0,
                ))
                return None
        
        # Entry is achievable, submit to executor
        try:
            exec_result = await self.executor.submit_limit_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                limit_price=entry_price,
            )
            
            order.state = TradeState.ORDER_SUBMITTED
            order.submitted_at = datetime.utcnow()
            order.submitted_bid = current_bid
            order.submitted_ask = current_ask
            
            self.orders[order_id] = order
            
            logger.info(
                f"✓ Order submitted: {order_id} | {side} {quantity} {symbol} "
                f"@ {entry_price:.4f}"
            )
            
            # Start price monitoring task
            self.price_monitor_tasks[order_id] = asyncio.create_task(
                self._monitor_order_price(order_id)
            )
            
            return order
        
        except Exception as e:
            logger.error(f"✗ Order submission failed: {e}")
            order.state = TradeState.ERROR
            return None
    
    async def _monitor_order_price(self, order_id: str) -> None:
        """
        Monitor order price and enforce Missed Trade Protocol.
        
        Continuously checks market price and cancels if it moves away.
        """
        if order_id not in self.orders:
            return
        
        order = self.orders[order_id]
        
        try:
            while order.state == TradeState.ORDER_SUBMITTED:
                # Simulate getting current price (would come from data feed)
                await asyncio.sleep(self.protocol.check_interval_seconds)
                
                # Get market prices (mock: would come from real data feed)
                current_bid, current_ask = await self._get_current_prices(order.symbol)
                
                order.price_checks.append((datetime.utcnow(), current_bid, current_ask))
                
                # Check Missed Trade Protocol
                should_cancel, reason = await self.protocol.should_cancel_order(
                    order, current_bid, current_ask
                )
                
                if should_cancel:
                    await self._cancel_order(order_id, CancellationReason.MISSED_TRADE_PROTOCOL, reason)
                    break
        
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"✗ Price monitoring error: {e}")
    
    async def _cancel_order(
        self,
        order_id: str,
        reason: CancellationReason,
        detail: Optional[str] = None,
    ) -> bool:
        """
        Cancel order and log reason.
        
        Args:
            order_id: Order to cancel
            reason: Cancellation reason
            detail: Additional detail
            
        Returns:
            True if cancelled successfully
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        
        try:
            # Cancel via executor
            await self.executor.cancel_order(order_id)
            
            # Update order state
            order.state = TradeState.ORDER_CANCELLED
            order.cancelled_at = datetime.utcnow()
            order.cancellation_reason = reason
            order.order_lifetime_seconds = (
                order.cancelled_at - order.submitted_at
            ).total_seconds() if order.submitted_at else 0
            
            # Log missed trade
            if reason == CancellationReason.MISSED_TRADE_PROTOCOL:
                self.missed_trades.append(MissedTradeLog(
                    timestamp=datetime.utcnow(),
                    signal_id=order.signal_id,
                    symbol=order.symbol,
                    side=order.side,
                    quantity=order.quantity,
                    entry_price=order.entry_price,
                    reason=reason,
                    deviation_pct=order.max_price_deviation * 100,
                    order_lifetime_seconds=order.order_lifetime_seconds,
                ))
            
            # Cancel monitoring task
            if order_id in self.price_monitor_tasks:
                self.price_monitor_tasks[order_id].cancel()
                del self.price_monitor_tasks[order_id]
            
            logger.info(
                f"✗ Order cancelled: {order_id} | reason={reason.value} | "
                f"detail={detail} | lifetime={order.order_lifetime_seconds:.1f}s"
            )
            
            return True
        
        except Exception as e:
            logger.error(f"✗ Cancel order failed: {e}")
            return False
    
    async def record_order_fill(
        self,
        order_id: str,
        filled_price: float,
        filled_quantity: float,
    ) -> bool:
        """
        Record order fill.
        
        Args:
            order_id: Filled order
            filled_price: Fill price
            filled_quantity: Filled quantity
            
        Returns:
            True if recorded successfully
        """
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        order.state = TradeState.ORDER_FILLED
        order.filled_at = datetime.utcnow()
        order.filled_price = filled_price
        order.filled_quantity = filled_quantity
        order.slippage_realized = (
            (filled_price - order.entry_price) / order.entry_price * 10000
        )
        
        # Move to trades
        self.trades[order_id] = order
        
        # Cancel monitoring task
        if order_id in self.price_monitor_tasks:
            self.price_monitor_tasks[order_id].cancel()
            del self.price_monitor_tasks[order_id]
        
        logger.info(
            f"✓ Order filled: {order_id} | {filled_quantity} @ {filled_price:.4f} | "
            f"slippage {order.slippage_realized:.1f}bps"
        )
        
        return True
    
    async def _get_current_prices(self, symbol: str) -> Tuple[float, float]:
        """
        Get current bid/ask prices (mock).
        
        In production: would get from real data feed.
        """
        # Mock implementation
        return (100.0, 100.05)
    
    def get_order(self, order_id: str) -> Optional[TradeOrder]:
        """Get order details."""
        return self.orders.get(order_id)
    
    def get_trade(self, order_id: str) -> Optional[TradeOrder]:
        """Get filled trade details."""
        return self.trades.get(order_id)
    
    def get_missed_trades_summary(self) -> Dict:
        """
        Get statistics on missed trades.
        
        Returns:
            Summary dictionary
        """
        if not self.missed_trades:
            return {
                'total_missed': 0,
                'by_reason': {},
                'average_deviation_pct': 0.0,
                'average_lifetime_seconds': 0.0,
            }
        
        by_reason = {}
        total_deviation = 0.0
        total_lifetime = 0.0
        
        for mt in self.missed_trades:
            reason_str = mt.reason.value
            by_reason[reason_str] = by_reason.get(reason_str, 0) + 1
            total_deviation += mt.deviation_pct
            total_lifetime += mt.order_lifetime_seconds
        
        return {
            'total_missed': len(self.missed_trades),
            'by_reason': by_reason,
            'average_deviation_pct': total_deviation / len(self.missed_trades),
            'average_lifetime_seconds': total_lifetime / len(self.missed_trades),
            'missed_trades': [
                {
                    'timestamp': str(mt.timestamp),
                    'signal_id': mt.signal_id,
                    'symbol': mt.symbol,
                    'side': mt.side,
                    'quantity': mt.quantity,
                    'entry_price': mt.entry_price,
                    'reason': mt.reason.value,
                    'deviation_pct': mt.deviation_pct,
                }
                for mt in self.missed_trades[-20:]  # Last 20
            ]
        }
    
    def get_execution_statistics(self) -> Dict:
        """
        Get execution statistics.
        
        Returns:
            Stats dictionary
        """
        filled_trades = [t for t in self.trades.values() if t.state == TradeState.ORDER_FILLED]
        cancelled_orders = [o for o in self.orders.values() if o.state == TradeState.ORDER_CANCELLED]
        
        if not filled_trades:
            avg_slippage = 0.0
        else:
            avg_slippage = sum(t.slippage_realized for t in filled_trades) / len(filled_trades)
        
        return {
            'total_orders_submitted': len(self.orders),
            'filled_orders': len(filled_trades),
            'cancelled_orders': len(cancelled_orders),
            'missed_trades': len(self.missed_trades),
            'fill_rate_percent': (len(filled_trades) / len(self.orders) * 100)
            if self.orders else 0,
            'average_slippage_bps': avg_slippage,
            'execution_type': self.executor_type,
        }

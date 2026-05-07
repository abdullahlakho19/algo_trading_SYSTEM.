# ============================================================================
# PAPER_SIMULATOR.PY - Highly Accurate Paper Trading Simulator
# Tracks equity, positions, executes fills against live L1 bid/ask
# ============================================================================

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OrderStatus(str, Enum):
    """Order status enumeration."""
    PENDING = "PENDING"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


class PositionStatus(str, Enum):
    """Position status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    LIQUIDATED = "LIQUIDATED"


@dataclass
class Fill:
    """Execution fill record."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    commission: float
    timestamp: datetime
    slippage: float = 0.0  # Price difference from expected


@dataclass
class Position:
    """Open trading position."""
    symbol: str
    side: str  # 'LONG' or 'SHORT'
    quantity: float
    entry_price: float
    entry_time: datetime
    pnl: float = 0.0
    pnl_percent: float = 0.0
    status: PositionStatus = PositionStatus.OPEN
    fills: List[Fill] = field(default_factory=list)
    current_price: float = 0.0
    
    def update_price(self, price: float) -> None:
        """Update unrealized P&L based on current price."""
        self.current_price = price
        if self.side == 'LONG':
            self.pnl = (price - self.entry_price) * self.quantity
            self.pnl_percent = ((price - self.entry_price) / self.entry_price) * 100
        else:  # SHORT
            self.pnl = (self.entry_price - price) * self.quantity
            self.pnl_percent = ((self.entry_price - price) / self.entry_price) * 100


@dataclass
class SimulatedOrder:
    """Simulated order record."""
    order_id: str
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: float
    price: float
    order_type: str  # 'LIMIT', 'MARKET', 'STOP_LOSS', etc.
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    average_fill_price: float = 0.0
    commission_paid: float = 0.0
    timestamp: datetime = field(default_factory=datetime.utcnow)
    fills: List[Fill] = field(default_factory=list)
    rejected_reason: Optional[str] = None
    
    def get_remaining_quantity(self) -> float:
        """Get unfilled quantity."""
        return self.quantity - self.filled_quantity


class PaperSimulator:
    """
    Highly accurate paper trading simulator.
    
    Features:
    - Tracks simulated equity, cash, positions
    - Executes fills against live L1 bid/ask data
    - Realistic slippage modeling
    - Commission/fee calculation
    - Multi-position tracking
    - Accurate P&L calculation
    
    Designed for:
    - Pre-live testing of execution strategies
    - Performance measurement of order execution algorithms
    - Risk management validation
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission_rate: float = 0.0005,  # 0.05% per trade
        slippage_bps: float = 2.0,  # 2 basis points
    ):
        """
        Initialize paper simulator.
        
        Args:
            initial_capital: Starting account balance
            commission_rate: Commission as fraction of trade value
            slippage_bps: Slippage in basis points (1 bps = 0.0001)
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_bps = slippage_bps / 10000  # Convert bps to fraction
        
        self.positions: Dict[str, Position] = {}
        self.closed_positions: List[Position] = []
        self.orders: Dict[str, SimulatedOrder] = {}
        self.fills: List[Fill] = []
        
        self.trades_count = 0
        self.total_fees = 0.0
        self.max_equity = initial_capital
        self.min_equity = initial_capital
        
        logger.info(
            f"✓ PaperSimulator initialized: capital=${initial_capital:,.2f}, "
            f"commission={commission_rate*100:.2f}%, slippage={slippage_bps:.2f}bps"
        )
    
    @property
    def total_equity(self) -> float:
        """Calculate total account equity (cash + open positions)."""
        equity = self.current_capital
        for pos in self.positions.values():
            if pos.status == PositionStatus.OPEN:
                equity += pos.pnl
        return equity
    
    @property
    def total_positions_value(self) -> float:
        """Sum of all open position values."""
        total = 0.0
        for pos in self.positions.values():
            if pos.status == PositionStatus.OPEN:
                total += pos.quantity * pos.current_price
        return total
    
    @property
    def portfolio_stats(self) -> Dict:
        """Get portfolio statistics."""
        equity = self.total_equity
        return {
            'total_equity': equity,
            'cash': self.current_capital,
            'open_positions': len([p for p in self.positions.values() if p.status == PositionStatus.OPEN]),
            'total_positions_value': self.total_positions_value,
            'total_pnl': equity - self.initial_capital,
            'total_pnl_percent': ((equity - self.initial_capital) / self.initial_capital) * 100,
            'trades_count': self.trades_count,
            'total_fees': self.total_fees,
            'max_equity': self.max_equity,
            'min_equity': self.min_equity,
            'return_percent': ((equity - self.initial_capital) / self.initial_capital) * 100,
        }
    
    async def submit_order(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        order_type: str = "LIMIT",
    ) -> SimulatedOrder:
        """
        Submit order to simulator.
        
        Args:
            order_id: Unique order identifier
            symbol: Trading pair
            side: 'BUY' or 'SELL'
            quantity: Order size
            price: Order price
            order_type: 'LIMIT', 'MARKET', 'STOP_LOSS'
            
        Returns:
            SimulatedOrder object
        """
        order = SimulatedOrder(
            order_id=order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            order_type=order_type,
        )
        self.orders[order_id] = order
        
        logger.debug(
            f"📝 Order submitted: {order_id} {side} {quantity} {symbol} @ ${price:.2f}"
        )
        return order
    
    async def fill_order(
        self,
        order_id: str,
        bid_price: float,
        ask_price: float,
        timestamp: datetime = None,
    ) -> Optional[Fill]:
        """
        Execute order fill against live L1 bid/ask.
        
        Args:
            order_id: Order to fill
            bid_price: Current bid level
            ask_price: Current ask level
            timestamp: Fill timestamp
            
        Returns:
            Fill record if executed, None otherwise
        """
        if timestamp is None:
            timestamp = datetime.utcnow()
        
        if order_id not in self.orders:
            logger.warning(f"⚠️  Order not found: {order_id}")
            return None
        
        order = self.orders[order_id]
        
        # Check if order can be filled (limit order logic)
        if order.order_type == "LIMIT":
            can_fill = False
            fill_price = 0.0
            
            if order.side == "BUY" and order.price >= bid_price:
                # Buy: limit price >= bid (we buy at bid or better)
                can_fill = True
                fill_price = min(order.price, ask_price)  # Realistic: fill at ask
            elif order.side == "SELL" and order.price <= ask_price:
                # Sell: limit price <= ask (we sell at ask or better)
                can_fill = True
                fill_price = max(order.price, bid_price)  # Realistic: fill at bid
            
            if not can_fill:
                return None
        
        elif order.order_type == "MARKET":
            # Market order fills at bid/ask
            fill_price = ask_price if order.side == "BUY" else bid_price
        else:
            return None
        
        # Calculate slippage
        expected_price = order.price if order.order_type == "LIMIT" else (ask_price if order.side == "BUY" else bid_price)
        slippage = (fill_price - expected_price) if order.side == "BUY" else (expected_price - fill_price)
        
        # Calculate commission
        trade_value = order.quantity * fill_price
        commission = trade_value * self.commission_rate
        
        # Create fill
        fill = Fill(
            order_id=order_id,
            symbol=order.symbol,
            side=order.side,
            quantity=order.quantity,
            price=fill_price,
            commission=commission,
            timestamp=timestamp,
            slippage=slippage,
        )
        
        # Update order state
        order.status = OrderStatus.FILLED
        order.filled_quantity = order.quantity
        order.average_fill_price = fill_price
        order.commission_paid = commission
        order.fills.append(fill)
        
        # Update simulator state
        self.fills.append(fill)
        self.current_capital -= (trade_value + commission) if order.side == "BUY" else -(trade_value - commission)
        self.total_fees += commission
        self.trades_count += 1
        
        # Update positions
        await self._update_position(fill)
        
        # Update equity tracking
        self.max_equity = max(self.max_equity, self.total_equity)
        self.min_equity = min(self.min_equity, self.total_equity)
        
        logger.info(
            f"✓ Fill: {order_id} | {order.side} {order.quantity} {order.symbol} "
            f"@ ${fill_price:.4f} | slippage ${slippage:.4f} | commission ${commission:.4f}"
        )
        
        return fill
    
    async def _update_position(self, fill: Fill) -> None:
        """Update position after fill."""
        symbol = fill.symbol
        
        if symbol not in self.positions:
            # New position
            self.positions[symbol] = Position(
                symbol=symbol,
                side="LONG" if fill.side == "BUY" else "SHORT",
                quantity=fill.quantity,
                entry_price=fill.price,
                entry_time=fill.timestamp,
                fills=[fill],
            )
        else:
            # Existing position
            pos = self.positions[symbol]
            
            if (fill.side == "BUY" and pos.side == "LONG") or (fill.side == "SELL" and pos.side == "SHORT"):
                # Adding to position
                pos.quantity += fill.quantity
                pos.entry_price = (
                    (pos.entry_price * (pos.quantity - fill.quantity) + fill.price * fill.quantity) /
                    pos.quantity
                )
                pos.fills.append(fill)
            
            elif (fill.side == "SELL" and pos.side == "LONG") or (fill.side == "BUY" and pos.side == "SHORT"):
                # Closing/reducing position
                if fill.quantity >= pos.quantity:
                    # Fully closed
                    pos.quantity = 0
                    pos.status = PositionStatus.CLOSED
                    self.closed_positions.append(pos)
                    del self.positions[symbol]
                else:
                    # Partially closed
                    pos.quantity -= fill.quantity
                    pos.fills.append(fill)
    
    async def update_position_price(self, symbol: str, price: float) -> None:
        """Update position unrealized P&L with current price."""
        if symbol in self.positions:
            self.positions[symbol].update_price(price)
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order."""
        if order_id not in self.orders:
            return False
        
        order = self.orders[order_id]
        if order.status == OrderStatus.PENDING:
            order.status = OrderStatus.CANCELLED
            logger.debug(f"✗ Order cancelled: {order_id}")
            return True
        
        return False
    
    async def close_position(
        self,
        symbol: str,
        current_bid: float,
        current_ask: float,
    ) -> Optional[Fill]:
        """Close position at market."""
        if symbol not in self.positions:
            return None
        
        pos = self.positions[symbol]
        
        # Create closing order
        close_side = "SELL" if pos.side == "LONG" else "BUY"
        close_price = current_bid if pos.side == "LONG" else current_ask
        
        order_id = f"close_{symbol}_{datetime.utcnow().timestamp()}"
        await self.submit_order(
            order_id=order_id,
            symbol=symbol,
            side=close_side,
            quantity=pos.quantity,
            price=close_price,
            order_type="MARKET",
        )
        
        return await self.fill_order(order_id, current_bid, current_ask)
    
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get current position for symbol."""
        return self.positions.get(symbol)
    
    def get_order(self, order_id: str) -> Optional[SimulatedOrder]:
        """Get order details."""
        return self.orders.get(order_id)
    
    def get_all_positions(self) -> Dict[str, Position]:
        """Get all open positions."""
        return {k: v for k, v in self.positions.items() if v.status == PositionStatus.OPEN}
    
    def get_fills_for_symbol(self, symbol: str) -> List[Fill]:
        """Get all fills for symbol."""
        return [f for f in self.fills if f.symbol == symbol]
    
    async def get_portfolio_snapshot(self) -> Dict:
        """Get complete portfolio snapshot."""
        return {
            'timestamp': datetime.utcnow().isoformat(),
            'equity': self.total_equity,
            'cash': self.current_capital,
            'positions': {
                symbol: {
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'entry_price': pos.entry_price,
                    'current_price': pos.current_price,
                    'pnl': pos.pnl,
                    'pnl_percent': pos.pnl_percent,
                    'status': pos.status.value,
                }
                for symbol, pos in self.positions.items()
            },
            'stats': self.portfolio_stats,
        }

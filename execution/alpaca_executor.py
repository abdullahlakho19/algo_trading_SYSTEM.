# ============================================================================
# ALPACA_EXECUTOR.PY - Live Alpaca API Execution Wrapper
# Formats and sends limit/stop-loss bracket orders to Alpaca
# ============================================================================

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple

import aiohttp

logger = logging.getLogger(__name__)


class OrderSide(str, Enum):
    """Order side enumeration."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(str, Enum):
    """Alpaca order status."""
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    DONE_FOR_DAY = "done_for_day"
    CANCELLED = "cancelled"
    EXPIRED = "expired"
    REJECTED = "rejected"
    PENDING_NEW = "pending_new"
    PENDING_CANCEL = "pending_cancel"


class TimeInForce(str, Enum):
    """Time in force for orders."""
    DAY = "day"
    GTC = "gtc"  # Good till cancelled
    OPX = "opx"
    CLS = "cls"


@dataclass
class AlpacaBracketOrder:
    """Bracket order structure (parent + legs)."""
    parent_order_id: str
    entry_order_id: str
    take_profit_order_id: Optional[str] = None
    stop_loss_order_id: Optional[str] = None
    symbol: str = ""
    side: str = ""
    quantity: float = 0.0
    entry_price: float = 0.0
    tp_price: float = 0.0
    sl_price: float = 0.0
    timestamp: datetime = None
    status: str = "PENDING"


class AlpacaExecutor:
    """
    Live Alpaca API execution wrapper.
    
    Features:
    - Async HTTP requests to Alpaca API
    - Bracket order execution (entry + TP + SL)
    - Real-time order status tracking
    - Position management
    - Account information retrieval
    - Order cancellation and replacement
    
    Designed for:
    - Production live trading execution
    - Multiple order management
    - Risk-controlled bracket orders
    - Real-time fill tracking
    """
    
    BASE_URL = "https://api.alpaca.markets"
    PAPER_URL = "https://paper-api.alpaca.markets"
    
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        use_paper: bool = True,
        request_timeout: float = 10.0,
    ):
        """
        Initialize Alpaca executor.
        
        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            use_paper: Use paper trading endpoint
            request_timeout: HTTP request timeout in seconds
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = self.PAPER_URL if use_paper else self.BASE_URL
        self.request_timeout = request_timeout
        
        self.session: Optional[aiohttp.ClientSession] = None
        self.bracket_orders: Dict[str, AlpacaBracketOrder] = {}
        self.active_orders: Dict[str, Dict] = {}
        
        logger.info(
            f"✓ AlpacaExecutor initialized: base_url={self.base_url}, "
            f"timeout={request_timeout}s"
        )
    
    def _get_headers(self) -> Dict[str, str]:
        """Get Alpaca API headers."""
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.api_secret,
            "Content-Type": "application/json",
        }
    
    async def connect(self) -> None:
        """Initialize async session."""
        if self.session is None:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.request_timeout)
            )
            logger.debug("✓ Alpaca session connected")
    
    async def disconnect(self) -> None:
        """Close async session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.debug("✓ Alpaca session closed")
    
    async def _request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict] = None,
    ) -> Dict:
        """
        Make authenticated request to Alpaca API.
        
        Args:
            method: HTTP method (GET, POST, DELETE, etc.)
            endpoint: API endpoint path
            data: Request payload
            
        Returns:
            Response dictionary
            
        Raises:
            Exception on API error
        """
        if self.session is None:
            await self.connect()
        
        url = f"{self.base_url}{endpoint}"
        headers = self._get_headers()
        
        try:
            async with self.session.request(
                method, url, json=data, headers=headers
            ) as resp:
                response_data = await resp.json()
                
                if resp.status >= 400:
                    logger.error(
                        f"✗ Alpaca API error: {resp.status} {response_data}"
                    )
                    raise Exception(f"API error {resp.status}: {response_data}")
                
                return response_data
        
        except asyncio.TimeoutError:
            logger.error(f"✗ Alpaca request timeout: {endpoint}")
            raise
        except Exception as e:
            logger.error(f"✗ Alpaca request failed: {e}")
            raise
    
    async def submit_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        limit_price: float,
        time_in_force: str = "day",
    ) -> Dict:
        """
        Submit limit order to Alpaca.
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            limit_price: Limit price
            time_in_force: 'day', 'gtc', etc.
            
        Returns:
            Order response from Alpaca
        """
        order_data = {
            "symbol": symbol,
            "qty": quantity,
            "side": side,
            "type": "limit",
            "limit_price": limit_price,
            "time_in_force": time_in_force,
        }
        
        response = await self._request("POST", "/v2/orders", order_data)
        
        order_id = response.get("id")
        self.active_orders[order_id] = response
        
        logger.info(
            f"✓ Limit order: {order_id} {side.upper()} {quantity} {symbol} "
            f"@ ${limit_price:.2f}"
        )
        
        return response
    
    async def submit_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        entry_price: float,
        take_profit_price: float,
        stop_loss_price: float,
        time_in_force: str = "day",
    ) -> AlpacaBracketOrder:
        """
        Submit bracket order (entry + TP + SL).
        
        Args:
            symbol: Stock symbol
            side: 'buy' or 'sell'
            quantity: Order quantity
            entry_price: Entry limit price
            take_profit_price: TP price for profit taking
            stop_loss_price: SL price for risk management
            time_in_force: 'day' or 'gtc'
            
        Returns:
            AlpacaBracketOrder object with order IDs
        """
        # Submit entry order
        entry_order = await self.submit_limit_order(
            symbol, side, quantity, entry_price, time_in_force
        )
        entry_order_id = entry_order["id"]
        
        # Determine TP/SL sides (opposite to entry)
        tp_side = "sell" if side == "buy" else "buy"
        sl_side = "sell" if side == "buy" else "buy"
        
        # Submit TP and SL orders (typically as OCO or separate)
        tp_order = await self.submit_limit_order(
            symbol, tp_side, quantity, take_profit_price, "gtc"
        )
        tp_order_id = tp_order["id"]
        
        sl_order = await self.submit_limit_order(
            symbol, sl_side, quantity, stop_loss_price, "gtc"
        )
        sl_order_id = sl_order["id"]
        
        # Track bracket order
        bracket = AlpacaBracketOrder(
            parent_order_id=entry_order_id,
            entry_order_id=entry_order_id,
            take_profit_order_id=tp_order_id,
            stop_loss_order_id=sl_order_id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            tp_price=take_profit_price,
            sl_price=stop_loss_price,
            timestamp=datetime.utcnow(),
        )
        
        self.bracket_orders[entry_order_id] = bracket
        
        logger.info(
            f"✓ Bracket order: {entry_order_id} | entry ${entry_price:.2f}, "
            f"TP ${take_profit_price:.2f}, SL ${stop_loss_price:.2f}"
        )
        
        return bracket
    
    async def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status from Alpaca.
        
        Args:
            order_id: Order ID to query
            
        Returns:
            Order details
        """
        response = await self._request("GET", f"/v2/orders/{order_id}")
        return response
    
    async def cancel_order(self, order_id: str) -> bool:
        """
        Cancel order.
        
        Args:
            order_id: Order to cancel
            
        Returns:
            True if cancelled successfully
        """
        try:
            await self._request("DELETE", f"/v2/orders/{order_id}")
            self.active_orders.pop(order_id, None)
            logger.info(f"✓ Order cancelled: {order_id}")
            return True
        except Exception as e:
            logger.error(f"✗ Cancel order failed: {e}")
            return False
    
    async def cancel_bracket_order(
        self,
        bracket: AlpacaBracketOrder,
    ) -> bool:
        """
        Cancel entire bracket order (all legs).
        
        Args:
            bracket: AlpacaBracketOrder to cancel
            
        Returns:
            True if all legs cancelled
        """
        results = []
        
        if bracket.entry_order_id:
            results.append(await self.cancel_order(bracket.entry_order_id))
        
        if bracket.take_profit_order_id:
            results.append(await self.cancel_order(bracket.take_profit_order_id))
        
        if bracket.stop_loss_order_id:
            results.append(await self.cancel_order(bracket.stop_loss_order_id))
        
        bracket.status = "CANCELLED"
        logger.info(f"✓ Bracket order cancelled: {bracket.parent_order_id}")
        
        return all(results)
    
    async def get_positions(self) -> List[Dict]:
        """
        Get all open positions.
        
        Returns:
            List of position objects
        """
        response = await self._request("GET", "/v2/positions")
        return response if isinstance(response, list) else []
    
    async def get_position(self, symbol: str) -> Optional[Dict]:
        """
        Get position for specific symbol.
        
        Args:
            symbol: Stock symbol
            
        Returns:
            Position details or None
        """
        try:
            response = await self._request("GET", f"/v2/positions/{symbol}")
            return response
        except Exception:
            return None
    
    async def close_position(self, symbol: str) -> bool:
        """
        Close position for symbol (market order).
        
        Args:
            symbol: Stock symbol to close
            
        Returns:
            True if closed successfully
        """
        try:
            await self._request("DELETE", f"/v2/positions/{symbol}")
            logger.info(f"✓ Position closed: {symbol}")
            return True
        except Exception as e:
            logger.error(f"✗ Close position failed: {e}")
            return False
    
    async def get_account(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Account details
        """
        response = await self._request("GET", "/v2/account")
        return response
    
    async def replace_order(
        self,
        order_id: str,
        quantity: float = None,
        limit_price: float = None,
    ) -> Dict:
        """
        Replace (modify) existing order.
        
        Args:
            order_id: Order to replace
            quantity: New quantity (optional)
            limit_price: New limit price (optional)
            
        Returns:
            New order details
        """
        data = {}
        if quantity is not None:
            data["qty"] = quantity
        if limit_price is not None:
            data["limit_price"] = limit_price
        
        response = await self._request(
            "PATCH", f"/v2/orders/{order_id}", data
        )
        return response
    
    async def get_order_list(self, status: str = "all") -> List[Dict]:
        """
        Get list of orders.
        
        Args:
            status: Order status filter ('open', 'closed', 'all')
            
        Returns:
            List of orders
        """
        params = f"?status={status}" if status else ""
        response = await self._request("GET", f"/v2/orders{params}")
        return response if isinstance(response, list) else []
    
    async def track_bracket_order_fills(
        self,
        bracket: AlpacaBracketOrder,
    ) -> Tuple[bool, float, float]:
        """
        Track bracket order fills (entry + one leg of TP/SL).
        
        Args:
            bracket: Bracket order to track
            
        Returns:
            (filled, entry_fill_price, exit_fill_price)
        """
        entry_order = await self.get_order_status(bracket.entry_order_id)
        entry_filled = entry_order.get("status") == OrderStatus.FILLED.value
        entry_price = float(entry_order.get("filled_avg_price", 0))
        
        if entry_filled:
            # Check which exit was filled (TP or SL)
            tp_order = await self.get_order_status(bracket.take_profit_order_id)
            sl_order = await self.get_order_status(bracket.stop_loss_order_id)
            
            tp_filled = tp_order.get("status") == OrderStatus.FILLED.value
            sl_filled = sl_order.get("status") == OrderStatus.FILLED.value
            
            exit_price = 0.0
            if tp_filled:
                exit_price = float(tp_order.get("filled_avg_price", 0))
                await self.cancel_order(bracket.stop_loss_order_id)  # Cancel SL
            elif sl_filled:
                exit_price = float(sl_order.get("filled_avg_price", 0))
                await self.cancel_order(bracket.take_profit_order_id)  # Cancel TP
            
            return (entry_filled and (tp_filled or sl_filled), entry_price, exit_price)
        
        return (False, 0.0, 0.0)

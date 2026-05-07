# ============================================================================
# ALPACA FEED - L1 DATA LAYER
# Handles real-time Stocks & Forex data via Alpaca WebSocket
# ============================================================================

import logging
import asyncio
import json
from typing import Optional, Dict, List, Any, Callable, Coroutine
from dataclasses import dataclass, field
from datetime import datetime
import websockets
from alpaca.data.live import StockDataStream, CryptoDataStream
from alpaca.data.requests import StockLatestBarRequest, CryptoLatestBarRequest

from config import get_config

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BarData:
    """Unified bar/candle data structure."""
    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    trade_count: int = 0
    vwap: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "trade_count": self.trade_count,
            "vwap": self.vwap,
        }


@dataclass
class QuoteData:
    """Unified quote/tick data structure."""
    symbol: str
    timestamp: datetime
    bid_price: float
    bid_size: int
    ask_price: float
    ask_size: int
    last_trade: Optional[float] = None
    last_trade_size: Optional[int] = None
    
    @property
    def bid_ask_spread(self) -> float:
        """Calculate bid-ask spread in basis points."""
        if self.ask_price <= 0:
            return 0.0
        return ((self.ask_price - self.bid_price) / self.ask_price) * 10000
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            "bid": self.bid_price,
            "bid_size": self.bid_size,
            "ask": self.ask_price,
            "ask_size": self.ask_size,
            "spread_bps": self.bid_ask_spread,
        }


# ============================================================================
# ALPACA FEED CLASS
# ============================================================================

class AlpacaFeed:
    """
    Real-time Alpaca data feed for Stocks & Forex.
    
    Connects via WebSocket for 24/7 streaming of:
    - Bars (OHLCV data)
    - Quotes (bid/ask/last)
    - Trades (individual transactions)
    
    Implements exponential backoff reconnection and robust error handling.
    """
    
    def __init__(self):
        """Initialize Alpaca feed with configuration."""
        self.config = get_config()
        
        # Connection state
        self.is_connected = False
        self.reconnect_count = 0
        self.max_reconnect_attempts = 5
        self.base_reconnect_delay = 1  # seconds
        
        # Subscriptions
        self.bar_callbacks: Dict[str, List[Callable]] = {}
        self.quote_callbacks: Dict[str, List[Callable]] = {}
        self.trade_callbacks: Dict[str, List[Callable]] = {}
        
        # Data buffers
        self.latest_bars: Dict[str, BarData] = {}
        self.latest_quotes: Dict[str, QuoteData] = {}
        
        logger.info("✓ AlpacaFeed initialized")
    
    async def connect(self) -> None:
        """
        Establish WebSocket connection to Alpaca.
        
        Raises:
            ConnectionError: If connection fails after max retries
        """
        logger.info("Connecting to Alpaca WebSocket...")
        
        try:
            async with StockDataStream(
                self.config.ALPACA_API_KEY,
                self.config.ALPACA_SECRET_KEY,
                base_url=self.config.ALPACA_BASE_URL,
            ) as stream:
                self.is_connected = True
                self.reconnect_count = 0
                logger.info("✓ Connected to Alpaca WebSocket")
                
                # Keep connection alive
                async for msg in stream:
                    await self._process_message(msg)
                    
        except Exception as e:
            await self._handle_connection_error(e)
    
    async def _handle_connection_error(self, error: Exception) -> None:
        """
        Handle connection errors with exponential backoff.
        
        Args:
            error: The exception that occurred
        """
        self.is_connected = False
        self.reconnect_count += 1
        
        if self.reconnect_count > self.max_reconnect_attempts:
            logger.error(f"❌ Max reconnection attempts ({self.max_reconnect_attempts}) exceeded")
            raise ConnectionError(f"Failed to connect to Alpaca after {self.max_reconnect_attempts} attempts: {error}")
        
        # Exponential backoff: 1s, 2s, 4s, 8s, 16s
        delay = self.base_reconnect_delay * (2 ** (self.reconnect_count - 1))
        logger.warning(f"⚠ Connection lost: {error}. Reconnecting in {delay}s (attempt {self.reconnect_count}/{self.max_reconnect_attempts})")
        
        await asyncio.sleep(delay)
        await self.connect()
    
    async def _process_message(self, message: Dict[str, Any]) -> None:
        """
        Process incoming messages from Alpaca WebSocket.
        
        Args:
            message: Message data from Alpaca
        """
        try:
            if "bar" in message:
                await self._handle_bar(message["bar"])
            elif "quote" in message:
                await self._handle_quote(message["quote"])
            elif "trade" in message:
                await self._handle_trade(message["trade"])
        except Exception as e:
            logger.error(f"Error processing Alpaca message: {e}", exc_info=True)
    
    async def _handle_bar(self, bar_data: Dict[str, Any]) -> None:
        """
        Handle incoming bar/candle data.
        
        Args:
            bar_data: OHLCV bar data from Alpaca
        """
        try:
            symbol = bar_data.get("S")  # Symbol
            
            bar = BarData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(bar_data.get("t", 0)),
                open=float(bar_data.get("o", 0)),
                high=float(bar_data.get("h", 0)),
                low=float(bar_data.get("l", 0)),
                close=float(bar_data.get("c", 0)),
                volume=int(bar_data.get("v", 0)),
                trade_count=int(bar_data.get("n", 0)),
                vwap=float(bar_data.get("vw", 0)) if "vw" in bar_data else None,
            )
            
            # Store in buffer
            self.latest_bars[symbol] = bar
            
            # Execute callbacks
            if symbol in self.bar_callbacks:
                for callback in self.bar_callbacks[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(bar)
                        else:
                            callback(bar)
                    except Exception as e:
                        logger.error(f"Error in bar callback for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing bar data: {e}", exc_info=True)
    
    async def _handle_quote(self, quote_data: Dict[str, Any]) -> None:
        """
        Handle incoming quote/tick data.
        
        Args:
            quote_data: Bid/ask quote from Alpaca
        """
        try:
            symbol = quote_data.get("S")  # Symbol
            
            quote = QuoteData(
                symbol=symbol,
                timestamp=datetime.fromtimestamp(quote_data.get("t", 0)),
                bid_price=float(quote_data.get("bp", 0)),
                bid_size=int(quote_data.get("bs", 0)),
                ask_price=float(quote_data.get("ap", 0)),
                ask_size=int(quote_data.get("as", 0)),
            )
            
            # Store in buffer
            self.latest_quotes[symbol] = quote
            
            # Execute callbacks
            if symbol in self.quote_callbacks:
                for callback in self.quote_callbacks[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(quote)
                        else:
                            callback(quote)
                    except Exception as e:
                        logger.error(f"Error in quote callback for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing quote data: {e}", exc_info=True)
    
    async def _handle_trade(self, trade_data: Dict[str, Any]) -> None:
        """
        Handle individual trade data.
        
        Args:
            trade_data: Trade tick from Alpaca
        """
        try:
            symbol = trade_data.get("S")  # Symbol
            
            # Execute trade callbacks
            if symbol in self.trade_callbacks:
                for callback in self.trade_callbacks[symbol]:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(trade_data)
                        else:
                            callback(trade_data)
                    except Exception as e:
                        logger.error(f"Error in trade callback for {symbol}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing trade data: {e}", exc_info=True)
    
    def subscribe_bars(
        self,
        symbols: List[str],
        callback: Callable[[BarData], Coroutine]
    ) -> None:
        """
        Subscribe to bar updates for symbols.
        
        Args:
            symbols: List of symbols (e.g., ["AAPL", "MSFT"])
            callback: Async callback function called on each bar
        """
        for symbol in symbols:
            if symbol not in self.bar_callbacks:
                self.bar_callbacks[symbol] = []
            self.bar_callbacks[symbol].append(callback)
            logger.debug(f"Subscribed to bars: {symbol}")
    
    def subscribe_quotes(
        self,
        symbols: List[str],
        callback: Callable[[QuoteData], Coroutine]
    ) -> None:
        """
        Subscribe to quote updates for symbols.
        
        Args:
            symbols: List of symbols
            callback: Async callback function called on each quote
        """
        for symbol in symbols:
            if symbol not in self.quote_callbacks:
                self.quote_callbacks[symbol] = []
            self.quote_callbacks[symbol].append(callback)
            logger.debug(f"Subscribed to quotes: {symbol}")
    
    def subscribe_trades(
        self,
        symbols: List[str],
        callback: Callable[[Dict], Coroutine]
    ) -> None:
        """
        Subscribe to trade updates for symbols.
        
        Args:
            symbols: List of symbols
            callback: Async callback function called on each trade
        """
        for symbol in symbols:
            if symbol not in self.trade_callbacks:
                self.trade_callbacks[symbol] = []
            self.trade_callbacks[symbol].append(callback)
            logger.debug(f"Subscribed to trades: {symbol}")
    
    def get_latest_bar(self, symbol: str) -> Optional[BarData]:
        """Get most recent bar for symbol."""
        return self.latest_bars.get(symbol)
    
    def get_latest_quote(self, symbol: str) -> Optional[QuoteData]:
        """Get most recent quote for symbol."""
        return self.latest_quotes.get(symbol)
    
    async def disconnect(self) -> None:
        """Gracefully disconnect from Alpaca."""
        self.is_connected = False
        logger.info("✓ Disconnected from Alpaca")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

async def main():
    """Test Alpaca feed connection."""
    feed = AlpacaFeed()
    
    async def on_bar(bar: BarData):
        logger.info(f"Bar: {bar.symbol} @ {bar.close}")
    
    feed.subscribe_bars(["AAPL"], on_bar)
    await feed.connect()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    asyncio.run(main())

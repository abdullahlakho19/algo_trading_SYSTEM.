# ============================================================================
# DATA ENGINE - L1 DATA LAYER (MASTER)
# Normalizes and streams multi-market data into unified OHLCV format
# ============================================================================

import logging
import asyncio
from typing import Optional, Dict, List, Any, Callable, Set
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import pandas as pd

from data_feeds.alpaca_feed import AlpacaFeed, BarData as AlpacaBar, QuoteData
from data_feeds.yfinance_feed import yfinance_feed

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class MarketType(Enum):
    """Market type enumeration."""
    STOCKS = "stocks"
    FOREX = "forex"


@dataclass
class OHLCV:
    """
    Unified OHLCV candle data (normalized across all markets).
    
    This is the canonical data structure that ALL downstream layers consume.
    It normalizes data from Alpaca (stocks/forex) and yFinance (historical).
    Crypto removed — Stocks and Forex only.
    """
    symbol: str
    market_type: MarketType
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float  # Share count for stocks
    
    # Optional enriched fields
    vwap: Optional[float] = None  # Volume-weighted average price
    bid: Optional[float] = None
    ask: Optional[float] = None
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    trade_count: Optional[int] = None
    adjusted_close: Optional[float] = None  # For stocks with splits/dividends
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "market_type": self.market_type.value,
            "timestamp": self.timestamp.isoformat(),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "vwap": self.vwap,
            "bid": self.bid,
            "ask": self.ask,
            "bid_size": self.bid_size,
            "ask_size": self.ask_size,
            "trade_count": self.trade_count,
            "adjusted_close": self.adjusted_close,
        }
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to single-row DataFrame."""
        return pd.DataFrame([self.to_dict()])


@dataclass
class OrderBook:
    """
    Unified order book snapshot (bid/ask + depth).
    
    Used for: slippage calculation, order execution analysis, microstructure research.
    """
    symbol: str
    timestamp: datetime
    
    bids: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    asks: List[tuple] = field(default_factory=list)  # [(price, size), ...]
    
    @property
    def mid_price(self) -> Optional[float]:
        """Calculate mid-price from order book."""
        if not self.bids or not self.asks:
            return None
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        return (best_bid + best_ask) / 2
    
    @property
    def spread(self) -> Optional[float]:
        """Calculate spread in basis points."""
        if not self.bids or not self.asks:
            return None
        best_bid = self.bids[0][0]
        best_ask = self.asks[0][0]
        if best_ask == 0:
            return 0.0
        return ((best_ask - best_bid) / best_ask) * 10000


@dataclass
class MarketContext:
    """
    Complete market context snapshot (OHLCV + order book + metadata).
    
    This is passed to L2 (Intelligence) and L3 (AI/ML) layers for analysis.
    One MarketContext = everything needed to make a trade decision.
    """
    symbol: str
    market_type: MarketType
    timestamp: datetime
    
    # Core OHLCV data
    bars: Dict[str, OHLCV] = field(default_factory=dict)  # {timeframe: OHLCV}
    
    # Order book
    order_book: Optional[OrderBook] = None
    
    # Bid/ask quotes
    bid: Optional[float] = None
    ask: Optional[float] = None
    last_price: Optional[float] = None
    
    # Market metadata
    is_market_open: bool = False
    session: Optional[str] = None  # "pre", "regular", "post", "sydney", "london", "ny"
    
    def get_bar(self, timeframe: str = "1m") -> Optional[OHLCV]:
        """Get bar for specific timeframe."""
        return self.bars.get(timeframe)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "market_type": self.market_type.value,
            "timestamp": self.timestamp.isoformat(),
            "bars": {tf: bar.to_dict() for tf, bar in self.bars.items()},
            "bid": self.bid,
            "ask": self.ask,
            "last_price": self.last_price,
            "is_market_open": self.is_market_open,
            "session": self.session,
        }


# ============================================================================
# MASTER DATA ENGINE
# ============================================================================

class DataEngine:
    """
    Master Data Engine for L1 (Data Layer).
    
    PURPOSE (from PDF):
    "Fetches, normalises and streams live multi-market data into a unified
    OHLCV + Volume + OrderBook format that all downstream layers consume."
    
    WORKFLOW:
    1. yFinance Feed → Primary historical data (batch requests)
    2. Alpaca Feed → Live stock/forex data (WebSocket), fallback for stocks
    3. All feeds → Normalized to OHLCV
    4. All OHLCV → MarketContext objects
    5. Distribute MarketContext → L2 (Intelligence), L3 (AI/ML), etc.
    
    FEATURES:
    - Multi-market seamless normalization (stocks, forex only)
    - Dual-mode: live streaming + historical backtesting
    - Async/concurrent data collection
    - Rate limit handling (exponential backoff)
    - Data validation and cleaning
    - Callback system for consumers
    - Crypto removed — Alpaca-only for stocks and forex
    """
    
    def __init__(self):
        """Initialize Data Engine with all feeds."""
        self.yfinance = yfinance_feed  # Primary feed
        self.alpaca = AlpacaFeed()
        
        # Connection state
        self.is_running = False
        self.read_only_backtesting = False
        
        # Subscriptions
        self.market_context_callbacks: Dict[str, List[Callable]] = {}

        # Data buffers (latest market context for each symbol)
        self.contexts: Dict[str, MarketContext] = {}
        self.order_books: Dict[str, OrderBook] = {}
        
        # Tracking
        self.symbols_tracked: Set[str] = set()
        self.market_types: Dict[str, MarketType] = {}  # {symbol: market_type}
        
        logger.info("✓ DataEngine initialized with yFinance (primary) + Alpaca (fallback)")
    
    # ========================================================================
    # INITIALIZATION & CONNECTION
    # ========================================================================
    
    async def initialize_live(self, symbols: List[str]) -> None:
        """
        Initialize Data Engine for LIVE streaming (paper or live trading).
        
        Args:
            symbols: List of symbols to track (e.g., ["AAPL", "EUR/USD"])
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING DATA ENGINE (LIVE MODE)")
        logger.info("=" * 70)
        
        try:
            # All symbols go through Alpaca (stocks/forex only)
            alpaca_symbols = symbols  # AAPL, MSFT, EUR/USD, etc
            
            logger.info(f"Initializing Alpaca for stocks/forex: {alpaca_symbols}")
            await self.alpaca.connect()
            
            # Subscribe to bars
            async def on_alpaca_bar(bar: AlpacaBar):
                await self._on_alpaca_bar(bar)
            
            self.alpaca.subscribe_bars(alpaca_symbols, on_alpaca_bar)
            self.symbols_tracked.update(alpaca_symbols)
            
            for symbol in alpaca_symbols:
                # Distinguish forex vs stock
                if "/" in symbol:
                    self.market_types[symbol] = MarketType.FOREX
                else:
                    self.market_types[symbol] = MarketType.STOCKS
            
            logger.info(f"✓ Alpaca initialized for {len(alpaca_symbols)} symbols")
            
            self.is_running = True
            self.read_only_backtesting = False
            logger.info("✓ Data Engine ready for LIVE streaming (Stocks & Forex only)")
            logger.info("=" * 70)
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize Data Engine: {e}", exc_info=True)
            raise
    
    async def initialize_backtest(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> None:
        """
        Initialize Data Engine for BACKTESTING (historical data only).
        
        Priority chain: yFinance → Alpaca (stocks only)
        
        Args:
            symbols: List of symbols to backtest
            start_date: Start date as "YYYY-MM-DD"
            end_date: End date as "YYYY-MM-DD"
            interval: Candle interval (1m, 5m, 1h, 1d, etc)
        """
        logger.info("=" * 70)
        logger.info("INITIALIZING DATA ENGINE (BACKTEST MODE)")
        logger.info("=" * 70)
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Interval: {interval}")
        logger.info("Data source priority: yFinance → Alpaca (stocks only)")
        
        try:
            # Fetch all historical data with fallback chain
            historical_data = {}
            
            for symbol in symbols:
                bars_data = None
                data_source = None
                
                # Determine market type
                if "/" in symbol:
                    market_type = MarketType.FOREX
                else:
                    market_type = MarketType.STOCKS
                
                # PRIORITY 1: Try yFinance (primary)
                try:
                    lookback_days = (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days
                    bars_df = self.yfinance.get_bars(
                        symbol=symbol,
                        timeframe=interval,
                        lookback_days=lookback_days,
                    )
                    if bars_df is not None and len(bars_df) > 0:
                        data_source = "yFinance"
                        bars_data = bars_df
                        logger.info(f"  ✓ {symbol}: {len(bars_df)} bars from {data_source}")
                except Exception as e:
                    logger.debug(f"  yFinance fetch failed for {symbol}: {e}")
                
                # PRIORITY 2: Fallback to Alpaca (stocks only)
                if (bars_data is None or len(bars_data) == 0) and market_type == MarketType.STOCKS:
                    try:
                        bars_df = await self.alpaca.fetch_historical(
                            symbol, interval=interval, start=start_date, end=end_date
                        )
                        if bars_df is not None and len(bars_df) > 0:
                            data_source = "Alpaca"
                            bars_data = bars_df
                            logger.info(f"  ✓ {symbol}: {len(bars_df)} bars from {data_source}")
                    except Exception as e:
                        logger.debug(f"  Alpaca fetch failed for {symbol}: {e}")
                
                if bars_data is None or len(bars_data) == 0:
                    logger.warning(f"  ⚠ No data retrieved for {symbol}")
                    continue
                
                # Convert DataFrame to OHLCV objects
                bar_list = []
                if isinstance(bars_data, pd.DataFrame):
                    for idx, row in bars_data.iterrows():
                        ts = row.get("date") or row.get("timestamp") or idx
                        if isinstance(ts, str):
                            ts = pd.to_datetime(ts)
                        
                        ohlcv = OHLCV(
                            symbol=symbol,
                            market_type=market_type,
                            timestamp=ts,
                            open=float(row.get("open", 0)),
                            high=float(row.get("high", 0)),
                            low=float(row.get("low", 0)),
                            close=float(row.get("close", 0)),
                            volume=float(row.get("volume", 0)),
                            adjusted_close=float(row.get("adjusted_close") or row.get("adj close") or row.get("close", 0)),
                        )
                        bar_list.append(ohlcv)
                
                if bar_list:
                    historical_data[symbol] = bar_list
                    self.market_types[symbol] = market_type
                    self.symbols_tracked.add(symbol)
            
            # Convert to market contexts
            for symbol, bars in historical_data.items():
                for bar in bars:
                    market_type = self.market_types[symbol]
                    
                    if symbol not in self.contexts:
                        self.contexts[symbol] = MarketContext(
                            symbol=symbol,
                            market_type=market_type,
                            timestamp=bar.timestamp,
                            bars={interval: bar},
                        )
                    else:
                        self.contexts[symbol].timestamp = bar.timestamp
                        self.contexts[symbol].bars[interval] = bar
            
            self.is_running = True
            self.read_only_backtesting = True
            logger.info(f"✓ Loaded historical data for {len(historical_data)} symbols")
            logger.info("=" * 70)
        
        except Exception as e:
            logger.error(f"❌ Failed to initialize backtest: {e}", exc_info=True)
            raise
    
    # ========================================================================
    # LIVE FEED CALLBACKS
    # ========================================================================
    
    async def _on_alpaca_bar(self, bar: AlpacaBar) -> None:
        """
        Handle incoming Alpaca bar data.
        
        Called every time a bar closes on Alpaca.
        """
        try:
            # Normalize to OHLCV
            ohlcv = OHLCV(
                symbol=bar.symbol,
                market_type=self.market_types.get(bar.symbol, MarketType.STOCKS),
                timestamp=bar.timestamp,
                open=bar.open,
                high=bar.high,
                low=bar.low,
                close=bar.close,
                volume=bar.volume,
                vwap=bar.vwap,
                trade_count=bar.trade_count,
            )
            
            # Update or create market context
            if bar.symbol not in self.contexts:
                self.contexts[bar.symbol] = MarketContext(
                    symbol=bar.symbol,
                    market_type=self.market_types[bar.symbol],
                    timestamp=ohlcv.timestamp,
                    bars={"1m": ohlcv},
                )
            else:
                self.contexts[bar.symbol].bars["1m"] = ohlcv
                self.contexts[bar.symbol].timestamp = ohlcv.timestamp
            
            # Trigger callbacks
            await self._trigger_callbacks(bar.symbol)
        
        except Exception as e:
            logger.error(f"Error processing Alpaca bar for {bar.symbol}: {e}", exc_info=True)
    
    # ========================================================================
    # SUBSCRIPTIONS & CALLBACKS
    # ========================================================================
    
    def subscribe(
        self,
        symbol: str,
        callback: Callable[[MarketContext], None]
    ) -> None:
        """
        Subscribe to market context updates for a symbol.
        
        Args:
            symbol: Symbol to track
            callback: Function called when data updates
        """
        if symbol not in self.market_context_callbacks:
            self.market_context_callbacks[symbol] = []
        
        self.market_context_callbacks[symbol].append(callback)
        logger.debug(f"Subscribed to {symbol}")
    
    async def _trigger_callbacks(self, symbol: str) -> None:
        """Execute all callbacks for a symbol."""
        if symbol in self.market_context_callbacks and symbol in self.contexts:
            context = self.contexts[symbol]
            for callback in self.market_context_callbacks[symbol]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(context)
                    else:
                        callback(context)
                except Exception as e:
                    logger.error(f"Error in callback for {symbol}: {e}")
    
    # ========================================================================
    # DATA ACCESS
    # ========================================================================
    
    def get_context(self, symbol: str) -> Optional[MarketContext]:
        """Get latest market context for a symbol."""
        return self.contexts.get(symbol)
    
    def get_contexts(self) -> Dict[str, MarketContext]:
        """Get all market contexts."""
        return self.contexts.copy()
    
    def get_bar(self, symbol: str, timeframe: str = "1m") -> Optional[OHLCV]:
        """Get latest bar for symbol and timeframe."""
        context = self.contexts.get(symbol)
        return context.get_bar(timeframe) if context else None
    
    async def get_historical(
        self,
        symbol: str,
        start: str,
        end: str,
        interval: str = "1d"
    ) -> List[OHLCV]:
        """
        Get historical bars using priority chain: yFinance → Alpaca (stocks only).
        
        Args:
            symbol: Symbol
            start: Start date "YYYY-MM-DD"
            end: End date "YYYY-MM-DD"
            interval: Candle interval
        
        Returns:
            List of OHLCV bars
        """
        market_type = self.market_types.get(symbol, MarketType.STOCKS)
        
        # PRIORITY 1: Try yFinance (primary)
        try:
            lookback_days = (pd.to_datetime(end) - pd.to_datetime(start)).days
            bars_df = self.yfinance.get_bars(
                symbol=symbol,
                timeframe=interval,
                lookback_days=lookback_days,
            )
            if bars_df is not None and len(bars_df) > 0:
                logger.debug(f"✓ {symbol}: {len(bars_df)} bars from yFinance")
                result = []
                for idx, row in bars_df.iterrows():
                    ts = row.get("date") or row.get("timestamp") or idx
                    if isinstance(ts, str):
                        ts = pd.to_datetime(ts)
                    result.append(OHLCV(
                        symbol=symbol,
                        market_type=market_type,
                        timestamp=ts,
                        open=float(row.get("open", 0)),
                        high=float(row.get("high", 0)),
                        low=float(row.get("low", 0)),
                        close=float(row.get("close", 0)),
                        volume=float(row.get("volume", 0)),
                        adjusted_close=float(row.get("adjusted_close") or row.get("adj close") or row.get("close", 0)),
                    ))
                return result
        except Exception as e:
            logger.debug(f"yFinance fetch failed for {symbol}: {e}")
        
        # PRIORITY 2: Fallback to Alpaca (stocks only)
        if market_type == MarketType.STOCKS:
            try:
                bars_df = await self.alpaca.fetch_historical(
                    symbol, interval=interval, start=start, end=end
                )
                if bars_df is not None and len(bars_df) > 0:
                    logger.debug(f"✓ {symbol}: {len(bars_df)} bars from Alpaca")
                    result = []
                    for idx, row in bars_df.iterrows():
                        result.append(OHLCV(
                            symbol=symbol,
                            market_type=market_type,
                            timestamp=row.get("timestamp") or idx,
                            open=float(row.get("open", 0)),
                            high=float(row.get("high", 0)),
                            low=float(row.get("low", 0)),
                            close=float(row.get("close", 0)),
                            volume=float(row.get("volume", 0)),
                        ))
                    return result
            except Exception as e:
                logger.debug(f"Alpaca fetch failed for {symbol}: {e}")
        
        logger.warning(f"All data sources failed for {symbol}")
        return []
    
    # ========================================================================
    # LIFECYCLE
    # ========================================================================
    
    async def start(self) -> None:
        """Start the data engine (keep alive)."""
        if not self.is_running:
            raise RuntimeError("Data Engine not initialized. Call initialize_live() or initialize_backtest()")
        
        logger.info("✓ Data Engine started")
        
        try:
            # Keep Alpaca connection alive
            if not self.read_only_backtesting:
                await self.alpaca.connect()
        
        except asyncio.CancelledError:
            logger.info("Data Engine stopped")
        except Exception as e:
            logger.error(f"Error in Data Engine: {e}", exc_info=True)
    
    async def stop(self) -> None:
        """Stop the data engine gracefully."""
        self.is_running = False
        
        if not self.read_only_backtesting:
            await self.alpaca.disconnect()
        
        logger.info("✓ Data Engine stopped")
    
    def print_status(self) -> None:
        """Print data engine status."""
        print("\n" + "=" * 70)
        print("DATA ENGINE STATUS")
        print("=" * 70)
        print(f"Running: {self.is_running}")
        print(f"Backtesting: {self.read_only_backtesting}")
        print(f"Symbols tracked: {len(self.symbols_tracked)}")
        print(f"Market contexts available: {len(self.contexts)}")
        print(f"\nTracked symbols by market:")
        
        for market_type in MarketType:
            symbols = [s for s, mt in self.market_types.items() if mt == market_type]
            if symbols:
                print(f"  {market_type.value}: {', '.join(symbols)}")
        
        print(f"\nLatest data:")
        for symbol, context in list(self.contexts.items())[:5]:
            bar = context.get_bar()
            if bar:
                print(f"  {symbol}: {bar.close} @ {context.timestamp}")
        
        print("=" * 70 + "\n")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

async def main():
    """Test Data Engine."""
    engine = DataEngine()
    
    # Test backtest initialization
    await engine.initialize_backtest(
        symbols=["AAPL", "MSFT"],
        start_date="2024-01-01",
        end_date="2024-01-31",
        interval="1d"
    )
    
    engine.print_status()
    
    # Test data access
    context = engine.get_context("AAPL")
    if context:
        bar = context.get_bar("1d")
        logger.info(f"AAPL latest bar: {bar.close if bar else 'N/A'}")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    asyncio.run(main())

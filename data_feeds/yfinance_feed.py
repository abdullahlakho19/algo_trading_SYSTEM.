# ============================================================================
# YFINANCE FEED - PRIMARY DATA SOURCE
# ============================================================================
# Robust, frictionless data aggregation for equities and forex
# Supports multiple timeframes with automatic resampling
# Clean OHLCV DataFrames: UTC-localized, drops invalid rows

import logging
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional, Dict, List
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class YFinanceFeed:
    """
    Primary data feed for stocks and forex using yFinance.
    
    Features:
    - Fetches OHLCV for equities (AAPL) and forex (EURUSD=X)
    - Multiple timeframes: 1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo
    - Automatic resampling for non-native intervals
    - Data cleaning: removes 0/negative prices, UTC-localizes index
    - Batch fetching via get_multiple()
    - Internal caching with TTL
    
    Used by: L1 Data Layer (backtesting), L4 AI/ML (training), L6 Execution (paper trading)
    """
    
    # yFinance native intervals
    NATIVE_INTERVALS = {"1m", "5m", "15m", "1h", "1d", "1wk", "1mo"}
    
    # Resample mappings: requested → (fetch_interval, resample_rule)
    RESAMPLE_MAP = {
        "30m": ("15m", "30min"),
        "4h": ("1h", "4h"),
        "2h": ("1h", "2h"),
    }
    
    def __init__(self, cache_ttl: int = 3600):
        """
        Initialize YFinanceFeed.
        
        Args:
            cache_ttl: Cache time-to-live in seconds (default: 1 hour)
        """
        self.cache_ttl = cache_ttl
        self.cache: Dict[str, tuple] = {}  # key: (symbol, interval, start, end) → (df, timestamp)
        self.available = True
        logger.info(f"✓ YFinanceFeed initialized (cache_ttl={cache_ttl}s)")
    
    def _get_cache_key(self, symbol: str, interval: str, start: str, end: str) -> str:
        """Generate cache key."""
        return f"{symbol}|{interval}|{start}|{end}"
    
    def _is_cache_fresh(self, cached_at: datetime) -> bool:
        """Check if cache entry is still valid."""
        age_seconds = (datetime.now() - cached_at).total_seconds()
        return age_seconds < self.cache_ttl
    
    def _clean_ohlcv(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Clean OHLCV DataFrame.
        
        - Normalize column names to lowercase
        - Drop rows with 0 or negative prices
        - UTC-localize index
        - Remove duplicates
        - Sort by timestamp
        
        Args:
            df: Raw DataFrame from yfinance
            symbol: Ticker symbol
        
        Returns:
            Clean OHLCV DataFrame
        """
        if df.empty:
            return df
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Ensure required columns exist
        required = {"open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            logger.warning(f"Missing columns for {symbol}: {required - set(df.columns)}")
            return pd.DataFrame()
        
        # Drop rows where OHLC <= 0
        valid_mask = (df[["open", "high", "low", "close"]] > 0).all(axis=1)
        df = df[valid_mask]
        
        # Ensure volume column is numeric
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0).astype(int)
        
        # UTC-localize index
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        
        # Remove duplicates (keep first)
        df = df[~df.index.duplicated(keep="first")]
        
        # Sort by timestamp
        df = df.sort_index()
        
        # Keep only OHLCV + Adj Close (if present)
        cols = ["open", "high", "low", "close", "volume"]
        if "adj close" in df.columns:
            cols.append("adj close")
        df = df[cols]
        
        return df
    
    def _resample_to_target(self, df: pd.DataFrame, target_interval: str) -> pd.DataFrame:
        """
        Resample DataFrame to target interval.
        
        Args:
            df: Input DataFrame with DatetimeIndex
            target_interval: Target interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, etc.)
        
        Returns:
            Resampled OHLCV DataFrame
        """
        if df.empty or target_interval in self.NATIVE_INTERVALS:
            return df
        
        if target_interval not in self.RESAMPLE_MAP:
            logger.warning(f"No resample rule for {target_interval}, returning as-is")
            return df
        
        _, resample_rule = self.RESAMPLE_MAP[target_interval]
        
        try:
            resampler = df.resample(resample_rule)
            resampled = pd.DataFrame({
                "open": resampler["open"].first(),
                "high": resampler["high"].max(),
                "low": resampler["low"].min(),
                "close": resampler["close"].last(),
                "volume": resampler["volume"].sum(),
            })
            
            if "adj close" in df.columns:
                resampled["adj close"] = resampler["adj close"].last()
            
            # Drop NaN rows (gaps in data)
            resampled = resampled.dropna(subset=["close"])
            
            return resampled
        except Exception as e:
            logger.error(f"Error resampling to {target_interval}: {e}")
            return df
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 30,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for a symbol.
        
        Args:
            symbol: Ticker (e.g., "AAPL", "EURUSD=X")
            timeframe: Interval (1m, 5m, 15m, 30m, 1h, 4h, 1d, 1wk, 1mo)
            lookback_days: Days to fetch
        
        Returns:
            Clean OHLCV DataFrame (UTC-localized index, no invalid rows)
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")
        
        # Determine fetch interval (native or resampled)
        fetch_interval = timeframe
        if timeframe in self.RESAMPLE_MAP:
            fetch_interval = self.RESAMPLE_MAP[timeframe][0]
        
        # Check cache
        cache_key = self._get_cache_key(symbol, fetch_interval, start_str, end_str)
        if cache_key in self.cache:
            df_cached, cached_at = self.cache[cache_key]
            if self._is_cache_fresh(cached_at):
                logger.debug(f"✓ Cache hit: {symbol} {timeframe}")
                return self._resample_to_target(df_cached.copy(), timeframe)
        
        try:
            logger.info(f"Fetching {timeframe} data for {symbol} ({lookback_days}d)...")
            
            # Download from yFinance
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_str, end=end_str, interval=fetch_interval)
            
            if df.empty:
                logger.warning(f"No data for {symbol} at {fetch_interval}")
                return pd.DataFrame()
            
            # Clean
            df = self._clean_ohlcv(df, symbol)
            
            # Cache (before resampling, so we cache raw fetched interval)
            if not df.empty:
                self.cache[cache_key] = (df.copy(), datetime.now())
            
            # Resample if needed
            df = self._resample_to_target(df, timeframe)
            
            logger.info(f"✓ Fetched {len(df)} {timeframe} bars for {symbol}")
            return df
        
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}", exc_info=True)
            self.available = False
            return pd.DataFrame()
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Ticker symbol
        
        Returns:
            Latest close price or None if unavailable
        """
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period="1d")
            if not df.empty:
                return float(df["Close"].iloc[-1])
        except Exception as e:
            logger.warning(f"Error fetching latest price for {symbol}: {e}")
        return None
    
    def get_multiple(
        self,
        symbols: List[str],
        timeframe: str = "1d",
        lookback_days: int = 30,
    ) -> Dict[str, pd.DataFrame]:
        """
        Batch fetch OHLCV for multiple symbols.
        
        Args:
            symbols: List of tickers
            timeframe: Interval (same for all)
            lookback_days: Days to fetch
        
        Returns:
            Dict mapping symbol → clean OHLCV DataFrame
        """
        results = {}
        for symbol in symbols:
            try:
                df = self.get_bars(symbol, timeframe, lookback_days)
                results[symbol] = df
            except Exception as e:
                logger.error(f"Batch fetch failed for {symbol}: {e}")
                results[symbol] = pd.DataFrame()
        return results
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.cache.clear()
        logger.info("✓ Cache cleared")


# Singleton instance
yfinance_feed = YFinanceFeed()

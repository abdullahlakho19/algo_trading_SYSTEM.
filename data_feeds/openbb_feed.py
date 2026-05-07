# ============================================================================
# OPENBB FEED - PRIMARY DATA SOURCE
# ============================================================================
# OpenBB SDK integration for stocks, forex, and economic data
# This is the primary data source. yFinance is fallback only.
#
# OpenBB aggregates 100+ free financial data sources:
# - Stocks: PolygonIO, YFinance, FMP
# - Forex: FMP, FRED, YFinance
# - Economics: FRED, Trading Economics, OECD
# - News: NewsAPI, Finnhub, YFinance
#
# Documentation: https://docs.openbb.co/platform

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

logger = logging.getLogger(__name__)


class OpenBBFeed:
    """
    OpenBB data feed for stocks and forex.
    
    Provides OHLCV data, economic calendar, and news from OpenBB.
    Falls back to yFinance for reliability.
    """
    
    # Timeframe mapping to OpenBB format
    TIMEFRAME_MAP = {
        "1m": "1min",
        "5m": "5min",
        "15m": "15min",
        "1h": "1hour",
        "4h": "4hour",
        "1d": "1day",
        "1wk": "1week",
        "1mo": "1month",
    }
    
    def __init__(self):
        """Initialize OpenBB feed."""
        try:
            from openbb import obb
            self.obb = obb
            self.available = True
            logger.info("✓ OpenBB initialized successfully")
        except ImportError:
            logger.warning("⚠ OpenBB not installed. Using fallback feeds only.")
            self.obb = None
            self.available = False
    
    def get_bars(
        self,
        symbol: str,
        timeframe: str = "1d",
        lookback_days: int = 365,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get OHLCV bars for a symbol.
        
        Args:
            symbol: Stock ticker (AAPL) or Forex pair (EUR/USD)
            timeframe: 1m, 5m, 15m, 1h, 4h, 1d, 1wk, 1mo
            lookback_days: Days of history to fetch
            start_date: Optional start date "YYYY-MM-DD"
            end_date: Optional end date "YYYY-MM-DD"
        
        Returns:
            DataFrame with OHLCV data (columns: open, high, low, close, volume)
            Index is DatetimeIndex in UTC
        """
        if not self.available:
            return None
        
        try:
            # Calculate date range
            if end_date is None:
                end_date = datetime.utcnow().strftime("%Y-%m-%d")
            if start_date is None:
                start_dt = datetime.utcnow() - timedelta(days=lookback_days)
                start_date = start_dt.strftime("%Y-%m-%d")
            
            tf_mapped = self.TIMEFRAME_MAP.get(timeframe, "1day")
            
            # Determine if stock or forex
            if "/" in symbol:
                # Forex pair (EUR/USD)
                return self._get_forex_bars(symbol, tf_mapped, start_date, end_date)
            else:
                # Stock ticker (AAPL)
                return self._get_stock_bars(symbol, tf_mapped, start_date, end_date)
        
        except Exception as e:
            logger.warning(f"OpenBB get_bars failed for {symbol}: {e}")
            return None
    
    def _get_stock_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch stock data from OpenBB."""
        try:
            data = self.obb.equity.price.historical(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                interval=timeframe,
                provider="yfinance",  # Use yFinance as provider (most reliable)
            )
            
            if data is None or len(data) == 0:
                return None
            
            df = data.to_df()
            
            # Ensure standard columns
            if "open" in df.columns and "close" in df.columns:
                # Already has OHLCV
                return df[["open", "high", "low", "close", "volume"]]
            
            logger.debug(f"✓ OpenBB stock data: {symbol} {len(df)} bars")
            return df
        
        except Exception as e:
            logger.debug(f"Stock fetch failed: {e}")
            return None
    
    def _get_forex_bars(
        self,
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
    ) -> Optional[pd.DataFrame]:
        """Fetch forex data from OpenBB."""
        try:
            # Convert EUR/USD to EURUSD for OpenBB
            clean_symbol = symbol.replace("/", "")
            
            # Try FMP first (has reliable forex)
            try:
                data = self.obb.currency.price.historical(
                    symbol=clean_symbol,
                    start_date=start_date,
                    end_date=end_date,
                    interval=timeframe,
                    provider="fmp",  # FMP is free and has forex
                )
            except Exception:
                # Fallback to YFinance
                data = self.obb.equity.price.historical(
                    symbol=symbol,  # Try with slash format
                    start_date=start_date,
                    end_date=end_date,
                    interval=timeframe,
                    provider="yfinance",
                )
            
            if data is None or len(data) == 0:
                return None
            
            df = data.to_df()
            
            if "open" in df.columns:
                return df[["open", "high", "low", "close", "volume"]]
            
            logger.debug(f"✓ OpenBB forex data: {symbol} {len(df)} bars")
            return df
        
        except Exception as e:
            logger.debug(f"Forex fetch failed: {e}")
            return None
    
    def get_latest_price(self, symbol: str) -> Optional[float]:
        """
        Get latest price for a symbol.
        
        Args:
            symbol: Stock (AAPL) or Forex (EUR/USD)
        
        Returns:
            Latest closing price or None if failed
        """
        if not self.available:
            return None
        
        try:
            if "/" in symbol:
                # Forex
                clean = symbol.replace("/", "")
                quote = self.obb.currency.price.quote(
                    symbol=clean,
                    provider="fmp",
                )
            else:
                # Stock
                quote = self.obb.equity.price.quote(
                    symbol=symbol,
                    provider="yfinance",
                )
            
            if quote is None:
                return None
            
            df = quote.to_df()
            if len(df) > 0 and "close" in df.columns:
                return float(df["close"].iloc[0])
        
        except Exception as e:
            logger.debug(f"Failed to get latest price for {symbol}: {e}")
        
        return None
    
    def get_economic_calendar(
        self,
        days_ahead: int = 7,
        countries: Optional[List[str]] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Get economic calendar events.
        
        Args:
            days_ahead: Number of days to look ahead
            countries: List of country codes (US, EU, GB, JP, etc.)
        
        Returns:
            DataFrame with columns:
            - date: Event date/time
            - event: Event name (FOMC, NFP, CPI, etc.)
            - country: Country code
            - impact: High, Medium, Low
            - forecast: Expected value
            - previous: Previous value
            - actual: Actual value (if released)
        """
        if not self.available:
            return None
        
        try:
            calendar = self.obb.economy.calendar(
                start_date=datetime.utcnow().strftime("%Y-%m-%d"),
                end_date=(
                    datetime.utcnow() + timedelta(days=days_ahead)
                ).strftime("%Y-%m-%d"),
            )
            
            if calendar is None:
                return None
            
            df = calendar.to_df()
            
            # Filter to high-impact events if requested
            if "impact" in df.columns:
                df = df[df["impact"].str.upper().isin(["HIGH", "MEDIUM"])]
            
            logger.debug(f"✓ Economic calendar: {len(df)} events found")
            return df
        
        except Exception as e:
            logger.debug(f"Economic calendar failed: {e}")
            return None
    
    def get_news(
        self,
        symbol: str,
        limit: int = 20,
    ) -> Optional[List[Dict[str, Any]]]:
        """
        Get latest news for a symbol.
        
        Args:
            symbol: Stock (AAPL) or Forex (EUR/USD)
            limit: Number of articles to return
        
        Returns:
            List of articles with structure:
            {
                "title": str,
                "summary": str,
                "source": str,
                "url": str,
                "date": datetime,
                "sentiment": float (-1 to 1)
            }
        """
        if not self.available:
            return None
        
        try:
            if "/" in symbol:
                # Forex - get world news on forex
                news = self.obb.news.world(
                    topics=["Forex", symbol.replace("/", "")],
                    limit=limit,
                )
            else:
                # Stock company news
                news = self.obb.news.company(
                    symbol=symbol,
                    limit=limit,
                )
            
            if news is None:
                return None
            
            df = news.to_df()
            
            # Convert to standardized format
            articles = []
            for _, row in df.iterrows():
                article = {
                    "title": row.get("title", ""),
                    "summary": row.get("summary", "") or row.get("description", ""),
                    "source": row.get("source", "OpenBB"),
                    "url": row.get("url", ""),
                    "date": row.get("date", datetime.utcnow()),
                    "sentiment": row.get("sentiment", 0.0),
                }
                articles.append(article)
            
            logger.debug(f"✓ News fetched: {symbol} {len(articles)} articles")
            return articles
        
        except Exception as e:
            logger.debug(f"News fetch failed for {symbol}: {e}")
            return None


# Singleton instance
openbb_feed = OpenBBFeed()

# ============================================================================
# TIMEFRAME_SCANNER.PY - Multi-timeframe analysis engine
# ============================================================================

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional

import pandas as pd


class Timeframe(Enum):
    """Available timeframes."""
    ONE_MIN = "1m"
    FIVE_MIN = "5m"
    FIFTEEN_MIN = "15m"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"
    ONE_WEEK = "1w"
    ONE_MONTH = "1M"


@dataclass
class TimeframeData:
    """Data for a single timeframe."""
    timeframe: Timeframe
    data: pd.DataFrame
    last_update: datetime
    bar_count: int
    is_complete: bool


class TimeframeScanner:
    """Multi-timeframe scanner engine."""
    
    TIMEFRAME_MINUTES = {
        Timeframe.ONE_MIN: 1,
        Timeframe.FIVE_MIN: 5,
        Timeframe.FIFTEEN_MIN: 15,
        Timeframe.ONE_HOUR: 60,
        Timeframe.FOUR_HOUR: 240,
        Timeframe.ONE_DAY: 1440,
        Timeframe.ONE_WEEK: 10080,
        Timeframe.ONE_MONTH: 43200,
    }
    
    def __init__(self, symbol: str):
        """
        Initialize scanner.
        
        Args:
            symbol: Trading symbol
        """
        self.symbol = symbol
        self.data: Dict[Timeframe, TimeframeData] = {}
        self.analysis: Dict[Timeframe, dict] = {}
    
    def add_data(self, timeframe: Timeframe, data: pd.DataFrame) -> None:
        """Add data for a timeframe."""
        now = datetime.utcnow()
        self.data[timeframe] = TimeframeData(
            timeframe=timeframe,
            data=data,
            last_update=now,
            bar_count=len(data),
            is_complete=True,
        )
    
    def scan_all(self) -> Dict[Timeframe, dict]:
        """Scan all timeframes."""
        results = {}
        for timeframe, tf_data in self.data.items():
            results[timeframe] = self._analyze_timeframe(tf_data)
        return results
    
    def _analyze_timeframe(self, tf_data: TimeframeData) -> dict:
        """Analyze single timeframe."""
        df = tf_data.data
        if df.empty:
            return {}
        
        recent = df.iloc[-1]
        sma_20 = df['Close'].rolling(20).mean().iloc[-1]
        rsi = self._calculate_rsi(df['Close'])
        
        return {
            'symbol': self.symbol,
            'timeframe': tf_data.timeframe.value,
            'close': float(recent['Close']),
            'open': float(recent['Open']),
            'high': float(recent['High']),
            'low': float(recent['Low']),
            'volume': float(recent['Volume']) if 'Volume' in recent else 0,
            'sma_20': float(sma_20) if pd.notna(sma_20) else None,
            'rsi': float(rsi) if pd.notna(rsi) else None,
            'trend': self._detect_trend(df),
            'momentum': self._calculate_momentum(df),
        }
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> float:
        """Calculate RSI."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]
    
    @staticmethod
    def _detect_trend(df: pd.DataFrame) -> str:
        """Detect trend from data."""
        if len(df) < 20:
            return 'unknown'
        
        sma20 = df['Close'].rolling(20).mean().iloc[-1]
        sma50 = df['Close'].rolling(50).mean().iloc[-1] if len(df) >= 50 else None
        
        current = df['Close'].iloc[-1]
        
        if sma50:
            if current > sma50 and sma20 > sma50:
                return 'uptrend'
            elif current < sma50 and sma20 < sma50:
                return 'downtrend'
        
        if current > sma20:
            return 'uptrend'
        elif current < sma20:
            return 'downtrend'
        
        return 'range'
    
    @staticmethod
    def _calculate_momentum(df: pd.DataFrame) -> float:
        """Calculate momentum."""
        if len(df) < 2:
            return 0
        return float(((df['Close'].iloc[-1] - df['Close'].iloc[-20]) / df['Close'].iloc[-20] * 100) if len(df) >= 20 else 0)
    
    def get_alignment(self) -> dict:
        """Get multi-timeframe alignment."""
        alignment = {}
        trends = {}
        
        for timeframe, data in self.data.items():
            if timeframe in self.analysis:
                trend = self.analysis[timeframe].get('trend')
                trends[timeframe.value] = trend
        
        aligned_up = sum(1 for t in trends.values() if t == 'uptrend')
        aligned_down = sum(1 for t in trends.values() if t == 'downtrend')
        
        alignment['trends'] = trends
        alignment['uptrend_count'] = aligned_up
        alignment['downtrend_count'] = aligned_down
        alignment['alignment_score'] = (max(aligned_up, aligned_down) / len(trends) * 100) if trends else 0
        
        return alignment


def main():
    """Test scanner."""
    scanner = TimeframeScanner("AAPL")
    print(f"Timeframe scanner initialized for {scanner.symbol}")


if __name__ == "__main__":
    main()

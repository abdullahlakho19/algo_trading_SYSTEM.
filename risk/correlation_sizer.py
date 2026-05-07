# ============================================================================
# CORRELATION-ADJUSTED POSITION SIZER - L5 RISK MANAGEMENT
# Reduces position sizes when two positions are highly correlated
# ============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class CorrelationPair:
    """Represents correlation between two symbols."""
    symbol_a: str
    symbol_b: str
    correlation: float
    lookback_periods: int
    
    @property
    def is_highly_correlated(self, threshold: float = 0.7) -> bool:
        """Check if correlation exceeds threshold."""
        return abs(self.correlation) >= threshold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol_a": self.symbol_a,
            "symbol_b": self.symbol_b,
            "correlation": self.correlation,
            "lookback_periods": self.lookback_periods,
            "is_highly_correlated": self.is_highly_correlated,
        }


@dataclass
class SizeAdjustment:
    """Represents a position size adjustment due to correlation."""
    symbol: str
    original_size: float
    adjusted_size: float
    correlated_symbols: List[str]
    adjustment_reason: str
    adjustment_factor: float  # 0.0 to 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "original_size": self.original_size,
            "adjusted_size": self.adjusted_size,
            "adjustment_factor": self.adjustment_factor,
            "correlated_symbols": self.correlated_symbols,
            "adjustment_reason": self.adjustment_reason,
        }


# ============================================================================
# CORRELATION CALCULATOR
# ============================================================================

class CorrelationMatrix:
    """Manages correlation calculations and caching."""
    
    def __init__(self, lookback_periods: int = 252):
        """
        Initialize correlation matrix.
        
        Args:
            lookback_periods: Number of periods for correlation calculation
        """
        self.lookback_periods = lookback_periods
        self.price_data: Dict[str, pd.Series] = {}  # {symbol: price_series}
        self.correlation_cache: Dict[Tuple[str, str], float] = {}
        self.cache_timestamp = {}
        self.cache_ttl = 3600  # 1 hour
    
    def add_price_data(self, symbol: str, prices: pd.Series) -> None:
        """
        Add price data for symbol.
        
        Args:
            symbol: Symbol identifier
            prices: Series of prices (datetime index)
        """
        if len(prices) < self.lookback_periods:
            logger.warning(f"Insufficient data for {symbol} (need {self.lookback_periods}, got {len(prices)})")
        
        self.price_data[symbol] = prices.tail(self.lookback_periods)
        self._invalidate_cache()
    
    def _invalidate_cache(self) -> None:
        """Invalidate correlation cache."""
        self.correlation_cache.clear()
    
    def get_correlation(
        self,
        symbol_a: str,
        symbol_b: str,
        method: str = "pearson"
    ) -> Optional[float]:
        """
        Calculate correlation between two symbols.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
            method: Correlation method ('pearson', 'spearman', 'kendall')
        
        Returns:
            Correlation coefficient (-1.0 to 1.0) or None if insufficient data
        """
        # Check cache
        cache_key = tuple(sorted([symbol_a, symbol_b]))
        if cache_key in self.correlation_cache:
            return self.correlation_cache[cache_key]
        
        # Validate data
        if symbol_a not in self.price_data or symbol_b not in self.price_data:
            logger.warning(f"Missing price data for {symbol_a} or {symbol_b}")
            return None
        
        prices_a = self.price_data[symbol_a]
        prices_b = self.price_data[symbol_b]
        
        # Align indices
        common_idx = prices_a.index.intersection(prices_b.index)
        
        if len(common_idx) < 2:
            logger.warning(f"Insufficient common data points for {symbol_a}-{symbol_b}")
            return None
        
        prices_a = prices_a[common_idx]
        prices_b = prices_b[common_idx]
        
        # Calculate returns (more stable than prices)
        returns_a = prices_a.pct_change().dropna()
        returns_b = prices_b.pct_change().dropna()
        
        if len(returns_a) < 2 or len(returns_b) < 2:
            return None
        
        try:
            correlation = returns_a.corr(returns_b, method=method)
            self.correlation_cache[cache_key] = correlation
            return correlation
        except Exception as e:
            logger.error(f"Error calculating correlation for {symbol_a}-{symbol_b}: {e}")
            return None
    
    def get_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get full correlation matrix for list of symbols.
        
        Args:
            symbols: List of symbols
        
        Returns:
            Correlation DataFrame (NxN matrix)
        """
        # Align all price series
        all_prices = pd.DataFrame()
        for symbol in symbols:
            if symbol in self.price_data:
                all_prices[symbol] = self.price_data[symbol]
        
        if all_prices.empty:
            return pd.DataFrame()
        
        # Calculate returns correlation
        returns = all_prices.pct_change().dropna()
        
        if len(returns) < 2:
            return pd.DataFrame()
        
        return returns.corr(method="pearson")


# ============================================================================
# CORRELATION-ADJUSTED SIZER
# ============================================================================

class CorrelationSizer:
    """
    Correlation-based position sizing engine.
    
    PHILOSOPHY (BlackRock Aladdin):
    "Reduce position size when two positions are highly correlated (> 0.7).
    This prevents portfolio concentration risk and reduces effective hedging."
    
    ALGORITHM:
    1. For each new position, calculate correlation with all open positions
    2. If correlation > 0.7, apply scaling factor:
       - Each high-correlation pair: multiply size by (1 - correlation_factor)
       - correlation_factor = (abs(correlation) - 0.7) / 0.3
       - This ranges from 0% reduction (corr=0.7) to 100% reduction (corr=1.0)
    3. Return adjusted position size
    
    EXAMPLE:
    >>> sizer = CorrelationSizer(correlation_threshold=0.7)
    >>> 
    >>> # Add open positions
    >>> sizer.add_position_prices("SPY", spy_prices)
    >>> sizer.add_position_prices("QQQ", qqq_prices)
    >>> 
    >>> # Calculate correlation
    >>> corr = sizer.get_correlation("SPY", "QQQ")  # 0.85
    >>> 
    >>> # Adjust size
    >>> adjusted = sizer.get_adjusted_size("SPY", intended_size=1000, open_symbols=["QQQ"])
    >>> print(f"Reduced from 1000 to {adjusted} shares")
    >>> # Output: Reduced from 1000 to 500 shares
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.7,
        lookback_periods: int = 252,
        allow_hedges: bool = False,
    ):
        """
        Initialize Correlation Sizer.
        
        Args:
            correlation_threshold: Correlation level triggering reduction
            lookback_periods: Periods for correlation calculation
            allow_hedges: If True, don't reduce size for short hedges
        """
        self.correlation_threshold = correlation_threshold
        self.allow_hedges = allow_hedges
        self.correlation_matrix = CorrelationMatrix(lookback_periods)
        
        # Track position directions for hedge detection
        self.position_directions: Dict[str, str] = {}  # {symbol: "long"/"short"}
        
        logger.info(f"✓ CorrelationSizer initialized")
        logger.info(f"  Correlation Threshold: {correlation_threshold}")
        logger.info(f"  Allow Hedges: {allow_hedges}")
    
    def add_position_prices(self, symbol: str, price_series: pd.Series) -> None:
        """
        Add price data for a symbol.
        
        Args:
            symbol: Symbol identifier
            price_series: pandas Series with datetime index
        """
        self.correlation_matrix.add_price_data(symbol, price_series)
        logger.debug(f"Added price data for {symbol} ({len(price_series)} periods)")
    
    def set_position_direction(self, symbol: str, direction: str) -> None:
        """
        Store position direction (long/short).
        
        Args:
            symbol: Symbol
            direction: "long" or "short"
        """
        if direction not in ("long", "short"):
            raise ValueError("Direction must be 'long' or 'short'")
        
        self.position_directions[symbol] = direction
    
    def get_correlation(self, symbol_a: str, symbol_b: str) -> Optional[float]:
        """
        Get correlation between two symbols.
        
        Args:
            symbol_a: First symbol
            symbol_b: Second symbol
        
        Returns:
            Correlation coefficient or None
        """
        return self.correlation_matrix.get_correlation(symbol_a, symbol_b)
    
    def get_correlated_positions(
        self,
        symbol: str,
        open_symbols: List[str],
        threshold: Optional[float] = None
    ) -> List[CorrelationPair]:
        """
        Get all open positions that are correlated with a symbol.
        
        Args:
            symbol: Target symbol
            open_symbols: List of open position symbols
            threshold: Override correlation threshold
        
        Returns:
            List of CorrelationPair objects for correlated positions
        """
        if threshold is None:
            threshold = self.correlation_threshold
        
        correlated = []
        
        for other_symbol in open_symbols:
            if other_symbol == symbol:
                continue
            
            corr = self.get_correlation(symbol, other_symbol)
            
            if corr is not None and abs(corr) >= threshold:
                pair = CorrelationPair(
                    symbol_a=symbol,
                    symbol_b=other_symbol,
                    correlation=corr,
                    lookback_periods=self.correlation_matrix.lookback_periods,
                )
                correlated.append(pair)
        
        return correlated
    
    def _calculate_adjustment_factor(self, correlation: float) -> float:
        """
        Calculate position size adjustment factor based on correlation.
        
        Formula:
            If abs(correlation) < threshold: factor = 1.0 (no reduction)
            If abs(correlation) >= threshold:
                factor = 1.0 - min((abs(correlation) - threshold) / (1.0 - threshold), 1.0)
        
        Examples:
            - correlation = 0.7: factor = 1.0 (no reduction)
            - correlation = 0.85: factor = 0.7 (30% reduction)
            - correlation = 1.0: factor = 0.0 (full reduction)
            - correlation = -0.8: same as 0.8 (use absolute value)
        
        Args:
            correlation: Correlation coefficient
        
        Returns:
            Adjustment factor (0.0 to 1.0)
        """
        abs_corr = abs(correlation)
        
        if abs_corr < self.correlation_threshold:
            return 1.0  # No adjustment needed
        
        # Linear reduction from threshold to 1.0
        # At threshold: factor = 1.0
        # At 1.0: factor = 0.0
        excess = abs_corr - self.correlation_threshold
        max_excess = 1.0 - self.correlation_threshold
        
        factor = 1.0 - (excess / max_excess)
        return max(0.0, min(1.0, factor))  # Clamp to [0.0, 1.0]
    
    def get_adjusted_size(
        self,
        symbol: str,
        intended_size: float,
        open_symbols: List[str],
        position_direction: Optional[str] = None,
    ) -> Tuple[float, SizeAdjustment]:
        """
        Calculate correlation-adjusted position size.
        
        ALGORITHM:
        1. Find all open positions correlated with new position
        2. For each correlated position (same direction):
           - Calculate correlation adjustment factor
           - Apply cumulative reduction: adjusted_size *= factor
        3. Return reduced size
        
        Args:
            symbol: Symbol to size
            intended_size: Original intended position size
            open_symbols: List of currently open position symbols
            position_direction: "long" or "short" (for hedge detection)
        
        Returns:
            (adjusted_size, SizeAdjustment object)
        
        Example:
            >>> sizer = CorrelationSizer()
            >>> adjusted, details = sizer.get_adjusted_size(
            ...     "TECH_NEW",
            ...     intended_size=1000,
            ...     open_symbols=["AAPL", "MSFT", "NVDA"]
            ... )
            >>> print(f"Adjusted from {intended_size} to {adjusted}")
        """
        # Get correlated positions
        correlated = self.get_correlated_positions(symbol, open_symbols)
        
        if not correlated:
            # No correlations detected
            return intended_size, SizeAdjustment(
                symbol=symbol,
                original_size=intended_size,
                adjusted_size=intended_size,
                correlated_symbols=[],
                adjustment_reason="No correlated positions detected",
                adjustment_factor=1.0,
            )
        
        # Apply adjustments cumulatively
        adjusted_size = float(intended_size)
        correlated_symbols_list = []
        
        for pair in correlated:
            other_symbol = pair.symbol_b if pair.symbol_a == symbol else pair.symbol_a
            correlated_symbols_list.append(other_symbol)
            
            # Check if this is a hedge
            other_direction = self.position_directions.get(other_symbol)
            is_hedge = (
                self.allow_hedges and
                position_direction and
                other_direction and
                position_direction != other_direction
            )
            
            if is_hedge:
                # Don't reduce size for hedges
                continue
            
            # Apply reduction
            factor = self._calculate_adjustment_factor(pair.correlation)
            adjusted_size *= factor
        
        # Create adjustment record
        adjustment = SizeAdjustment(
            symbol=symbol,
            original_size=intended_size,
            adjusted_size=adjusted_size,
            correlated_symbols=correlated_symbols_list,
            adjustment_reason=f"Adjusted for {len(correlated)} correlated position(s)",
            adjustment_factor=adjusted_size / intended_size if intended_size > 0 else 1.0,
        )
        
        logger.info(f"Size adjusted for {symbol}: {intended_size} → {adjusted_size} "
                   f"(factor: {adjustment.adjustment_factor:.2f})")
        
        return adjusted_size, adjustment
    
    def get_correlation_matrix(self, symbols: List[str]) -> pd.DataFrame:
        """
        Get full correlation matrix for symbols.
        
        Args:
            symbols: List of symbols
        
        Returns:
            Correlation DataFrame
        """
        return self.correlation_matrix.get_correlation_matrix(symbols)
    
    def print_correlations(self, symbols: List[str]) -> None:
        """Print correlation matrix for symbols."""
        corr_matrix = self.get_correlation_matrix(symbols)
        
        if corr_matrix.empty:
            logger.info("Correlation matrix is empty")
            return
        
        print("\n" + "=" * 70)
        print("CORRELATION MATRIX")
        print("=" * 70)
        print(corr_matrix.round(4))
        print("=" * 70 + "\n")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    # Create sample data
    dates = pd.date_range("2024-01-01", periods=252, freq="D")
    
    # Highly correlated assets
    spy_prices = pd.Series(
        np.random.randn(252).cumsum() + 400,
        index=dates,
        name="SPY"
    )
    qqq_prices = pd.Series(
        spy_prices.values + np.random.randn(252) * 20,  # ~0.9 correlation
        index=dates,
        name="QQQ"
    )
    # Low correlation asset
    eem_prices = pd.Series(
        np.random.randn(252).cumsum() + 100,
        index=dates,
        name="EEM"
    )
    
    # Initialize sizer
    sizer = CorrelationSizer(correlation_threshold=0.7)
    
    sizer.add_position_prices("SPY", spy_prices)
    sizer.add_position_prices("QQQ", qqq_prices)
    sizer.add_position_prices("EEM", eem_prices)
    
    sizer.set_position_direction("SPY", "long")
    sizer.set_position_direction("QQQ", "long")
    sizer.set_position_direction("EEM", "long")
    
    # Print correlations
    sizer.print_correlations(["SPY", "QQQ", "EEM"])
    
    # Test size adjustment
    adjusted, details = sizer.get_adjusted_size(
        symbol="SPY",
        intended_size=1000,
        open_symbols=["QQQ", "EEM"]
    )
    
    print(f"Size Adjustment: {details.to_dict()}")

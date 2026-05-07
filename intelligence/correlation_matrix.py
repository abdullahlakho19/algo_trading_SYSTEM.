# ============================================================================
# CORRELATION_MATRIX.PY - Cross-asset correlation tracking
# ============================================================================

from typing import Dict, List

import numpy as np
import pandas as pd


class CorrelationMatrix:
    """Tracks correlations between multiple assets."""
    
    def __init__(self, symbols: List[str], lookback_days: int = 60):
        """
        Initialize correlation matrix.
        
        Args:
            symbols: List of symbols to track
            lookback_days: Days to calculate correlation
        """
        self.symbols = symbols
        self.lookback_days = lookback_days
        self.price_data: Dict[str, pd.Series] = {}
        self.correlations = {}
    
    def add_price_data(self, symbol: str, prices: pd.Series) -> None:
        """Add price data for a symbol."""
        self.price_data[symbol] = prices
    
    def calculate_correlations(self) -> pd.DataFrame:
        """Calculate correlation matrix."""
        if len(self.price_data) < 2:
            return pd.DataFrame()
        
        # Create DataFrame from price data
        df = pd.DataFrame(self.price_data)
        
        # Calculate daily returns
        returns = df.pct_change().dropna()
        
        # Calculate correlation
        corr_matrix = returns.corr()
        self.correlations = corr_matrix
        
        return corr_matrix
    
    def get_correlation(self, symbol1: str, symbol2: str) -> float:
        """Get correlation between two symbols."""
        if self.correlations.empty:
            self.calculate_correlations()
        
        if symbol1 in self.correlations.index and symbol2 in self.correlations.columns:
            return float(self.correlations.loc[symbol1, symbol2])
        
        return 0.0
    
    def find_correlated_pairs(self, threshold: float = 0.7) -> List[tuple]:
        """Find pairs with correlation above threshold."""
        if self.correlations.empty:
            self.calculate_correlations()
        
        pairs = []
        for i in range(len(self.correlations.columns)):
            for j in range(i + 1, len(self.correlations.columns)):
                corr = self.correlations.iloc[i, j]
                if abs(corr) > threshold:
                    sym1 = self.correlations.columns[i]
                    sym2 = self.correlations.columns[j]
                    pairs.append((sym1, sym2, float(corr)))
        
        return pairs
    
    def find_uncorrelated_pairs(self, threshold: float = 0.3) -> List[tuple]:
        """Find pairs with low correlation (portfolio diversification)."""
        if self.correlations.empty:
            self.calculate_correlations()
        
        pairs = []
        for i in range(len(self.correlations.columns)):
            for j in range(i + 1, len(self.correlations.columns)):
                corr = self.correlations.iloc[i, j]
                if abs(corr) < threshold:
                    sym1 = self.correlations.columns[i]
                    sym2 = self.correlations.columns[j]
                    pairs.append((sym1, sym2, float(corr)))
        
        return pairs


def main():
    """Test correlation matrix."""
    print("Correlation matrix initialized")


if __name__ == "__main__":
    main()

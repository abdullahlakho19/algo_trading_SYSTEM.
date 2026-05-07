# ============================================================================
# SYNTHETIC DATA GENERATOR - Fallback for yFinance API Issues
# ============================================================================
# Generates realistic OHLCV data for model training when live feeds unavailable

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_synthetic_ohlcv(symbol: str, num_bars: int = 365, seed: int = None) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for a symbol.
    
    Args:
        symbol: Ticker symbol
        num_bars: Number of bars to generate (default 365 days)
        seed: Random seed for reproducibility
    
    Returns:
        DataFrame with OHLCV data
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Starting price varies by symbol
    base_prices = {
        "SPY": 400, "QQQ": 350, "IWM": 180, "EEM": 35, "GLD": 180, "TLT": 90,
        "AAPL": 180, "GOOGL": 140, "MSFT": 350, "EURUSD=X": 1.08
    }
    start_price = base_prices.get(symbol, 100)
    
    # Generate dates - using business days
    end_date = datetime.now()
    start_date = end_date - timedelta(days=num_bars * 2)  # Go back 2x to account for weekends
    dates = pd.bdate_range(start=start_date, end=end_date)  # Business days only
    dates = dates[-num_bars:]  # Take last num_bars
    
    # Generate price movements with realistic properties
    returns = np.random.normal(0.0005, 0.015, len(dates))  # Small positive drift, daily volatility
    prices = start_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV
    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
        'close': prices,
        'volume': np.random.uniform(10e6, 50e6, len(dates)).astype(int),
    }, index=dates)
    
    # Ensure high >= max(open, close), low <= min(open, close)
    df['high'] = df[['open', 'high', 'close']].max(axis=1)
    df['low'] = df[['open', 'low', 'close']].min(axis=1)
    
    # Reset index to make date a column
    df = df.reset_index(drop=False).rename(columns={'index': 'date'})
    
    # Lowercase columns
    df.columns = df.columns.str.lower()
    
    return df

def generate_training_dataset(symbols: list, num_bars: int = 365) -> pd.DataFrame:
    """
    Generate synthetic training dataset for multiple symbols.
    
    Args:
        symbols: List of ticker symbols
        num_bars: Number of bars per symbol
    
    Returns:
        Concatenated DataFrame with all symbols
    """
    dfs = []
    for i, symbol in enumerate(symbols):
        df = generate_synthetic_ohlcv(symbol, num_bars, seed=42 + i)
        df['symbol'] = symbol
        dfs.append(df)
    
    return pd.concat(dfs, ignore_index=True)

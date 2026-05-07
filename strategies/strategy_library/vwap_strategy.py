# ============================================================================
# VWAP_STRATEGY.PY - Volume-Weighted Average Price Strategy
# Institutional strategy with adaptive standard deviation bands
# ============================================================================

import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class VWAPBands:
    """VWAP with adaptive standard deviation bands."""
    vwap: float       # Volume-weighted average price
    upper_band: float # +N std dev
    lower_band: float # -N std dev
    std_dev: float    # Standard deviation of price from VWAP
    bandwidth: float  # Upper - Lower


class VWAPStrategy:
    """
    Volume-Weighted Average Price (VWAP) Strategy.
    
    Institutional-grade strategy using VWAP as dynamic support/resistance
    with adaptive standard deviation bands that expand/contract with volatility.
    
    Logic:
    1. Calculate VWAP: sum(price * volume) / sum(volume)
    2. Calculate deviation bands: VWAP ± (N_std_dev * historical_std_dev)
    3. Generate signals:
       - BUY: Price touches lower band + reversal confirmation
       - SELL: Price touches upper band + reversal confirmation
    4. Bands widen in high volatility (adaptive)
    5. Mean reversion bias: Price tends to revert to VWAP
    
    Appropriate for:
    - Institutional traders (large capital pools)
    - High-liquidity assets (ETFs, large-cap stocks, forex)
    - Mean reversion strategies
    - Intraday to swing trading
    
    Risk: Break of VWAP with high volume = regime change
    """
    
    def __init__(self, std_dev_multiplier: float = 2.0, lookback: int = 50):
        """
        Initialize VWAP strategy.
        
        Args:
            std_dev_multiplier: Band width in standard deviations (default 2.0)
            lookback: Periods for historical volatility calculation
        """
        self.std_dev_multiplier = std_dev_multiplier
        self.lookback = lookback
        logger.info(
            f"✓ VWAPStrategy initialized: "
            f"multiplier={std_dev_multiplier}, lookback={lookback}"
        )
    
    def calculate_vwap(self, bars: pd.DataFrame, window: int = 50) -> float:
        """
        Calculate Volume-Weighted Average Price.
        
        VWAP = sum(price * volume) / sum(volume)
        where price = (High + Low + Close) / 3
        
        Args:
            bars: OHLCV data
            window: Lookback period (default 50 bars)
            
        Returns:
            VWAP value
        """
        if len(bars) < window:
            window = len(bars)
        
        df = bars.tail(window).copy()
        
        # Typical price = (H + L + C) / 3
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Cumulative volume and cumulative typical price * volume
        df['pv'] = df['typical_price'] * df['Volume']
        
        vwap = df['pv'].sum() / df['Volume'].sum()
        return vwap
    
    def calculate_vwap_bands(
        self,
        bars: pd.DataFrame,
        vwap: float,
        window: int = 50
    ) -> VWAPBands:
        """
        Calculate VWAP with adaptive standard deviation bands.
        
        Bands expand/contract based on price volatility within the lookback.
        
        Args:
            bars: OHLCV data
            vwap: Pre-calculated VWAP
            window: Lookback for volatility
            
        Returns:
            VWAPBands object
        """
        if len(bars) < window:
            window = len(bars)
        
        df = bars.tail(window).copy()
        
        # Typical price
        df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
        
        # Standard deviation of price from VWAP (weighted by volume)
        df['deviation'] = (df['typical_price'] - vwap) ** 2
        df['dev_weighted'] = df['deviation'] * df['Volume']
        
        variance = df['dev_weighted'].sum() / df['Volume'].sum()
        std_dev = np.sqrt(variance)
        
        # Bands
        upper_band = vwap + (self.std_dev_multiplier * std_dev)
        lower_band = vwap - (self.std_dev_multiplier * std_dev)
        
        return VWAPBands(
            vwap=vwap,
            upper_band=upper_band,
            lower_band=lower_band,
            std_dev=std_dev,
            bandwidth=upper_band - lower_band
        )
    
    def detect_mean_reversion_setup(
        self,
        bars: pd.DataFrame,
        bands: VWAPBands,
        reversal_threshold: float = 0.02
    ) -> Optional[Dict]:
        """
        Detect mean reversion setup: price touches band + reversal.
        
        Args:
            bars: OHLCV data
            bands: VWAPBands object
            reversal_threshold: Min % move against extreme (0.02 = 2%)
            
        Returns:
            Setup dict if valid, None otherwise
        """
        current_close = bars['Close'].iloc[-1]
        prev_close = bars['Close'].iloc[-2] if len(bars) > 1 else current_close
        current_volume = bars['Volume'].iloc[-1]
        avg_volume = bars['Volume'].tail(20).mean()
        
        # Check if price touched upper band
        if current_close >= bands.upper_band and prev_close < bands.upper_band:
            # Price just touched upper band - watch for reversal
            # Reversal = close below open or body smaller than previous
            is_reversal = (
                current_close < bars['Open'].iloc[-1] or
                abs(current_close - bars['Open'].iloc[-1]) < 
                abs(prev_close - bars['Open'].iloc[-2])
            )
            
            if is_reversal and current_volume < avg_volume * 1.2:
                # Weak volume on touch = genuine mean reversion
                return {
                    'type': 'upper_band_reversal',
                    'direction': 0,  # SHORT / SELL
                    'entry': current_close,
                    'target': bands.vwap,
                    'stop': bands.upper_band * 1.01,  # Above band
                    'rationale': 'VWAP short: price at upper band, reversal, low volume',
                    'setup_strength': 0.75
                }
        
        # Check if price touched lower band
        if current_close <= bands.lower_band and prev_close > bands.lower_band:
            is_reversal = (
                current_close > bars['Open'].iloc[-1] or
                abs(current_close - bars['Open'].iloc[-1]) < 
                abs(prev_close - bars['Open'].iloc[-2])
            )
            
            if is_reversal and current_volume < avg_volume * 1.2:
                return {
                    'type': 'lower_band_reversal',
                    'direction': 1,  # LONG / BUY
                    'entry': current_close,
                    'target': bands.vwap,
                    'stop': bands.lower_band * 0.99,  # Below band
                    'rationale': 'VWAP long: price at lower band, reversal, low volume',
                    'setup_strength': 0.75
                }
        
        return None
    
    def detect_breakout_setup(
        self,
        bars: pd.DataFrame,
        bands: VWAPBands
    ) -> Optional[Dict]:
        """
        Detect breakout setup: price breaks band with high volume.
        
        More aggressive than mean reversion.
        
        Args:
            bars: OHLCV data
            bands: VWAPBands object
            
        Returns:
            Setup dict if valid, None otherwise
        """
        current_close = bars['Close'].iloc[-1]
        prev_high = bars['High'].tail(5).max()
        current_volume = bars['Volume'].iloc[-1]
        avg_volume = bars['Volume'].tail(20).mean()
        
        # Bullish breakout: close above upper band + high volume
        if current_close > bands.upper_band and current_volume > avg_volume * 1.5:
            momentum = (current_close - bars['Close'].iloc[-5]) / bars['Close'].iloc[-5]
            if momentum > 0.03:  # At least 3% up in 5 bars
                return {
                    'type': 'bullish_breakout',
                    'direction': 1,  # LONG
                    'entry': current_close,
                    'target': bands.upper_band * 1.05,
                    'stop': bands.vwap * 0.99,
                    'rationale': f'VWAP breakout long: close {current_close:.2f} > band {bands.upper_band:.2f}, vol {current_volume/avg_volume:.1f}x',
                    'setup_strength': 0.65
                }
        
        # Bearish breakout: close below lower band + high volume
        if current_close < bands.lower_band and current_volume > avg_volume * 1.5:
            momentum = (bars['Close'].iloc[-5] - current_close) / bars['Close'].iloc[-5]
            if momentum > 0.03:  # At least 3% down in 5 bars
                return {
                    'type': 'bearish_breakout',
                    'direction': 0,  # SHORT
                    'entry': current_close,
                    'target': bands.lower_band * 0.95,
                    'stop': bands.vwap * 1.01,
                    'rationale': f'VWAP breakout short: close {current_close:.2f} < band {bands.lower_band:.2f}, vol {current_volume/avg_volume:.1f}x',
                    'setup_strength': 0.65
                }
        
        return None
    
    def generate_signal(self, bars: pd.DataFrame) -> Optional[Dict]:
        """
        Generate VWAP strategy signal.
        
        Prioritizes mean reversion over breakout. Returns best available setup.
        
        Args:
            bars: OHLCV data
            
        Returns:
            Trade setup dict or None
        """
        if len(bars) < self.lookback:
            logger.warning(f"⚠️  Insufficient data for VWAP: {len(bars)} < {self.lookback}")
            return None
        
        try:
            # Calculate VWAP and bands
            vwap = self.calculate_vwap(bars)
            bands = self.calculate_vwap_bands(bars, vwap)
            
            # Check for mean reversion (priority)
            mr_setup = self.detect_mean_reversion_setup(bars, bands)
            if mr_setup:
                return mr_setup
            
            # Check for breakout
            bo_setup = self.detect_breakout_setup(bars, bands)
            if bo_setup:
                return bo_setup
            
            return None
        
        except Exception as e:
            logger.error(f"✗ VWAP signal generation failed: {e}")
            return None
    
    def get_vwap_metrics(self, bars: pd.DataFrame) -> Dict:
        """
        Get VWAP metrics for reporting.
        
        Args:
            bars: OHLCV data
            
        Returns:
            Dictionary with VWAP statistics
        """
        try:
            vwap = self.calculate_vwap(bars)
            bands = self.calculate_vwap_bands(bars, vwap)
            current_price = bars['Close'].iloc[-1]
            distance = current_price - vwap
            pct_above_vwap = (distance / vwap) * 100
            
            return {
                'vwap': vwap,
                'upper_band': bands.upper_band,
                'lower_band': bands.lower_band,
                'bandwidth': bands.bandwidth,
                'current_price': current_price,
                'distance_from_vwap': distance,
                'pct_above_vwap': pct_above_vwap,
                'std_dev': bands.std_dev,
            }
        except Exception as e:
            logger.error(f"✗ VWAP metrics failed: {e}")
            return {}

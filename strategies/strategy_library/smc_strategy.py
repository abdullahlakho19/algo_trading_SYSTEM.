# ============================================================================
# SMC_STRATEGY.PY - Smart Money Concepts / ICT Strategy
# Liquidity sweeps, order blocks, and fair value gaps (FVGs)
# ============================================================================

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class Liquidity Zone:
    """Institutional liquidity sink (high + low concentration area)."""
    level: float
    type: str  # 'above_high' or 'below_low'
    strength: int  # Number of touches (1-5)
    first_touch: int  # Bar index
    last_touch: int
    proximity: float  # Distance from current price (%)


@dataclass
class OrderBlock:
    """Institutional order block: bullish or bearish."""
    high: float
    low: float
    close: float
    bar_index: int
    type: str  # 'bullish' or 'bearish'
    strength: float  # 0-1 based on size and recency
    created_at: str  # Timestamp


@dataclass
class FairValueGap:
    """Fair Value Gap: swing high-low imbalance."""
    top: float
    bottom: float
    bar_index: int
    type: str  # 'bullish' or 'bearish'
    size: float  # Gap size in price units
    proximity: float  # Distance from current price


class SMCStrategy:
    """
    Smart Money Concepts (SMC) / ICT Strategy Implementation.
    
    Core Concepts:
    1. Liquidity Sweeps: Price moves beyond structure (HH, LL) to trigger stops
    2. Order Blocks: Consolidated accumulation areas before actual move
    3. Fair Value Gaps: Imbalances that price returns to fill
    4. First Support/Resistance: Pre-high/pre-low before swing extremes
    
    Trading Logic:
    - Identify liquidity level = institutional stop-hunt zone
    - Wait for sweep + order block setup
    - Enter on FVG fill or retracement into order block
    - Target: Opposite liquidity level
    
    Appropriate for:
    - 4H+ timeframes (institutional order accumulation)
    - Trending markets
    - Mean reversion within trends
    - Risk management with clear structure
    
    Edge: Institutions accumulate at key levels before directional moves.
    SMC identifies these areas statistically.
    """
    
    def __init__(self, lookback: int = 100, liquidity_threshold: float = 0.02):
        """
        Initialize SMC strategy.
        
        Args:
            lookback: Bars to analyze for structure (default 100)
            liquidity_threshold: Min price range for liquidity zone detection (2%)
        """
        self.lookback = lookback
        self.liquidity_threshold = liquidity_threshold
        logger.info(
            f"✓ SMCStrategy initialized: "
            f"lookback={lookback}, threshold={liquidity_threshold*100:.1f}%"
        )
    
    def find_structure_extremes(self, bars: pd.DataFrame) -> Tuple[float, float, int, int]:
        """
        Find recent swing highs and lows (structure).
        
        Args:
            bars: OHLCV data
            
        Returns:
            (swing_high, swing_low, swing_high_idx, swing_low_idx)
        """
        if len(bars) < 10:
            return bars['High'].max(), bars['Low'].min(), 0, 0
        
        df = bars.tail(self.lookback).copy()
        
        # Find swing high (local max)
        swing_high = df['High'].iloc[-5:].max()
        swing_high_idx = df['High'].tail(5).idxmax() - df.index[0]
        
        # Find swing low (local min)
        swing_low = df['Low'].iloc[-5:].min()
        swing_low_idx = df['Low'].tail(5).idxmin() - df.index[0]
        
        return swing_high, swing_low, swing_high_idx, swing_low_idx
    
    def identify_liquidity_zones(self, bars: pd.DataFrame) -> List[Liquidity Zone]:
        """
        Identify institutional liquidity zones.
        
        Liquidity = concentration of stops above highs and below lows.
        Created by institutional accumulation patterns.
        
        Args:
            bars: OHLCV data
            
        Returns:
            List of Liquidity Zones
        """
        liquidity_zones = []
        current_price = bars['Close'].iloc[-1]
        
        # Analyze recent swings
        df = bars.tail(self.lookback).copy()
        highs = df['High'].values
        lows = df['Low'].values
        
        # Find highest high and lowest low in window
        highest_high = highs.max()
        lowest_low = lows.min()
        
        # Zone above highest high (short-stop liquidation level)
        above_high = highest_high * (1 + self.liquidity_threshold)
        above_touches = np.sum(bars['High'] > highest_high)
        
        if above_touches > 0:
            proximity = ((above_high - current_price) / current_price) * 100
            liquidity_zones.append(Liquidity Zone(
                level=above_high,
                type='above_high',
                strength=min(5, above_touches),
                first_touch=len(bars) - np.where(bars['High'] > highest_high)[0][-1],
                last_touch=len(bars) - np.where(bars['High'] > highest_high)[0][0],
                proximity=proximity
            ))
        
        # Zone below lowest low (long-stop liquidation level)
        below_low = lowest_low * (1 - self.liquidity_threshold)
        below_touches = np.sum(bars['Low'] < lowest_low)
        
        if below_touches > 0:
            proximity = ((current_price - below_low) / current_price) * 100
            liquidity_zones.append(Liquidity Zone(
                level=below_low,
                type='below_low',
                strength=min(5, below_touches),
                first_touch=len(bars) - np.where(bars['Low'] < lowest_low)[0][-1],
                last_touch=len(bars) - np.where(bars['Low'] < lowest_low)[0][0],
                proximity=proximity
            ))
        
        return liquidity_zones
    
    def detect_order_blocks(self, bars: pd.DataFrame, lookback: int = 20) -> List[OrderBlock]:
        """
        Detect institutional order blocks.
        
        Order block = consolidated bar (small range relative to surroundings)
        before directional move. Shows institutional accumulation.
        
        Args:
            bars: OHLCV data
            lookback: Recent bars to analyze
            
        Returns:
            List of OrderBlock objects
        """
        order_blocks = []
        df = bars.tail(lookback).copy()
        
        for i in range(1, len(df) - 1):
            current_bar = df.iloc[i]
            next_bar = df.iloc[i + 1]
            
            # Current bar ATR
            current_range = current_bar['High'] - current_bar['Low']
            
            # Check if consolidation (small range)
            if current_range < (current_bar['Close'] * 0.01):  # Less than 1% range
                # Check for directional move after
                next_move = abs(next_bar['Close'] - current_bar['Close']) / current_bar['Close']
                
                if next_move > 0.015:  # At least 1.5% move after consolidation
                    ob_type = 'bullish' if next_bar['Close'] > current_bar['Close'] else 'bearish'
                    
                    strength = min(1.0, next_move / 0.03)  # Normalize to 0-1
                    
                    order_blocks.append(OrderBlock(
                        high=current_bar['High'],
                        low=current_bar['Low'],
                        close=current_bar['Close'],
                        bar_index=i,
                        type=ob_type,
                        strength=strength,
                        created_at=str(df.index[i])
                    ))
        
        return order_blocks
    
    def detect_fair_value_gaps(self, bars: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect Fair Value Gaps (FVGs).
        
        FVG = swing high-low imbalance where price "jumped over"
        a price level without trading there. Price tends to fill gaps.
        
        Args:
            bars: OHLCV data
            
        Returns:
            List of FairValueGap objects
        """
        fvgs = []
        current_price = bars['Close'].iloc[-1]
        
        if len(bars) < 5:
            return fvgs
        
        df = bars.tail(50).copy()
        
        for i in range(2, len(df) - 1):
            # Check for swing high then swing low pattern (bullish FVG)
            if (df['High'].iloc[i] < df['High'].iloc[i-2] and  # Lower high
                df['Low'].iloc[i] > df['High'].iloc[i-1]):     # Low > previous high
                
                # Bullish FVG: gap between previous high and current low
                fvg_bottom = df['High'].iloc[i-1]
                fvg_top = df['Low'].iloc[i]
                gap_size = fvg_top - fvg_bottom
                
                if gap_size > df['Close'].iloc[i] * 0.001:  # At least 0.1%
                    proximity = ((current_price - fvg_bottom) / current_price) * 100
                    fvgs.append(FairValueGap(
                        top=fvg_top,
                        bottom=fvg_bottom,
                        bar_index=i,
                        type='bullish',
                        size=gap_size,
                        proximity=proximity
                    ))
            
            # Check for swing low then swing high pattern (bearish FVG)
            if (df['Low'].iloc[i] > df['Low'].iloc[i-2] and   # Higher low
                df['High'].iloc[i] < df['Low'].iloc[i-1]):    # High < previous low
                
                # Bearish FVG: gap between current high and previous low
                fvg_bottom = df['High'].iloc[i]
                fvg_top = df['Low'].iloc[i-1]
                gap_size = fvg_top - fvg_bottom
                
                if gap_size > df['Close'].iloc[i] * 0.001:
                    proximity = ((fvg_top - current_price) / current_price) * 100
                    fvgs.append(FairValueGap(
                        top=fvg_top,
                        bottom=fvg_bottom,
                        bar_index=i,
                        type='bearish',
                        size=gap_size,
                        proximity=proximity
                    ))
        
        return fvgs
    
    def detect_liquidity_sweep(
        self,
        bars: pd.DataFrame,
        liquidity_zones: List[Liquidity Zone],
        order_blocks: List[OrderBlock]
    ) -> Optional[Dict]:
        """
        Detect liquidity sweep setup: price touches/breaks liquidity + OB nearby.
        
        Classic institutional pattern: shake out retail stops, then reverse.
        
        Args:
            bars: OHLCV data
            liquidity_zones: Pre-calculated liquidity levels
            order_blocks: Pre-calculated order blocks
            
        Returns:
            Trade setup dict if valid
        """
        if not liquidity_zones or not order_blocks:
            return None
        
        current_close = bars['Close'].iloc[-1]
        current_high = bars['High'].iloc[-1]
        current_low = bars['Low'].iloc[-1]
        current_volume = bars['Volume'].iloc[-1]
        avg_volume = bars['Volume'].tail(10).mean()
        
        for lz in liquidity_zones:
            # Check if price touched liquidity zone
            swept = False
            if lz.type == 'above_high' and current_high >= lz.level * 0.99:  # Within 1%
                swept = True
                direction = 0  # SHORT after sweep
            elif lz.type == 'below_low' and current_low <= lz.level * 1.01:
                swept = True
                direction = 1  # LONG after sweep
            else:
                continue
            
            if not swept:
                continue
            
            # Find nearest order block for entry
            nearest_ob = min(
                order_blocks,
                key=lambda ob: abs(ob.close - current_close),
                default=None
            )
            
            if nearest_ob and current_volume > avg_volume * 0.8:  # Decent volume on sweep
                ob_distance = abs(nearest_ob.close - current_close) / current_close
                if ob_distance < 0.02:  # Within 2% of OB
                    return {
                        'type': 'liquidity_sweep',
                        'direction': direction,
                        'entry': current_close,
                        'order_block_level': nearest_ob.close,
                        'liquidity_level': lz.level,
                        'target': nearest_ob.close,  # Retracement to OB
                        'stop': lz.level * 1.02 if direction == 0 else lz.level * 0.98,
                        'rationale': (
                            f'SMC Sweep: swept liq at {lz.level:.2f}, '
                            f'order block {nearest_ob.type} at {nearest_ob.close:.2f}, '
                            f'vol {current_volume/avg_volume:.1f}x'
                        ),
                        'setup_strength': 0.80
                    }
        
        return None
    
    def detect_fvg_fill(
        self,
        bars: pd.DataFrame,
        fvgs: List[FairValueGap]
    ) -> Optional[Dict]:
        """
        Detect FVG fill setup: price entering fair value gap.
        
        Price tends to fill gaps. Entry at gap boundary for mean reversion.
        
        Args:
            bars: OHLCV data
            fvgs: Pre-calculated FVGs
            
        Returns:
            Trade setup dict if valid
        """
        if not fvgs:
            return None
        
        current_close = bars['Close'].iloc[-1]
        
        for fvg in fvgs:
            # Check if price is entering the gap
            if fvg.type == 'bullish':
                # Bullish gap: price entry from top
                entering = (
                    bars['Close'].iloc[-2] > fvg.top and
                    current_close <= fvg.top and
                    current_close > fvg.bottom
                )
                if entering:
                    return {
                        'type': 'fvg_fill_bullish',
                        'direction': 1,  # LONG
                        'entry': current_close,
                        'target': fvg.bottom,  # Bottom of gap
                        'stop': fvg.top * 1.005,
                        'rationale': f'SMC FVG fill: bullish gap {fvg.bottom:.2f}-{fvg.top:.2f}, size {fvg.size:.4f}',
                        'setup_strength': 0.70
                    }
            
            elif fvg.type == 'bearish':
                # Bearish gap: price entry from bottom
                entering = (
                    bars['Close'].iloc[-2] < fvg.bottom and
                    current_close >= fvg.bottom and
                    current_close < fvg.top
                )
                if entering:
                    return {
                        'type': 'fvg_fill_bearish',
                        'direction': 0,  # SHORT
                        'entry': current_close,
                        'target': fvg.top,  # Top of gap
                        'stop': fvg.bottom * 0.995,
                        'rationale': f'SMC FVG fill: bearish gap {fvg.bottom:.2f}-{fvg.top:.2f}, size {fvg.size:.4f}',
                        'setup_strength': 0.70
                    }
        
        return None
    
    def generate_signal(self, bars: pd.DataFrame) -> Optional[Dict]:
        """
        Generate SMC strategy signal.
        
        Checks: liquidity sweep > FVG fill (priority order).
        
        Args:
            bars: OHLCV data
            
        Returns:
            Trade setup dict or None
        """
        if len(bars) < 50:
            logger.warning(f"⚠️  Insufficient data for SMC: {len(bars)} < 50")
            return None
        
        try:
            # Identify SMC structures
            liquidity_zones = self.identify_liquidity_zones(bars)
            order_blocks = self.detect_order_blocks(bars)
            fvgs = self.detect_fair_value_gaps(bars)
            
            # Priority: liquidity sweep (stronger signal)
            sweep_setup = self.detect_liquidity_sweep(bars, liquidity_zones, order_blocks)
            if sweep_setup:
                return sweep_setup
            
            # Secondary: FVG fill
            fvg_setup = self.detect_fvg_fill(bars, fvgs)
            if fvg_setup:
                return fvg_setup
            
            return None
        
        except Exception as e:
            logger.error(f"✗ SMC signal generation failed: {e}")
            return None
    
    def get_smc_metrics(self, bars: pd.DataFrame) -> Dict:
        """
        Get SMC analysis metrics for reporting.
        
        Args:
            bars: OHLCV data
            
        Returns:
            Dictionary with SMC statistics
        """
        try:
            liquidity_zones = self.identify_liquidity_zones(bars)
            order_blocks = self.detect_order_blocks(bars)
            fvgs = self.detect_fair_value_gaps(bars)
            
            return {
                'liquidity_zones': len(liquidity_zones),
                'order_blocks': len(order_blocks),
                'fvgs': len(fvgs),
                'bullish_blocks': sum(1 for ob in order_blocks if ob.type == 'bullish'),
                'bearish_blocks': sum(1 for ob in order_blocks if ob.type == 'bearish'),
            }
        except Exception as e:
            logger.error(f"✗ SMC metrics failed: {e}")
            return {}

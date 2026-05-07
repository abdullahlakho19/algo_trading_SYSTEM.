# ============================================================================
# STRUCTURE ENGINE - L2 INTELLIGENCE LAYER
# Real-time market structure detection (HH, HL, LH, LL, BoS, CHoCH)
# ============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS & DATA MODELS
# ============================================================================

class StructureType(Enum):
    """Market structure types."""
    HIGHER_HIGH = "HH"      # New high above previous swing high
    HIGHER_LOW = "HL"       # New low above previous swing low (bullish)
    LOWER_HIGH = "LH"       # New high below previous swing high (bearish)
    LOWER_LOW = "LL"        # New low below previous swing low
    BREAK_OF_STRUCTURE = "BoS"      # Breaks recent support/resistance
    CHANGE_OF_CHARACTER = "CHoCH"   # Shift from bullish to bearish or vice versa


class Trend(Enum):
    """Market trend direction."""
    UPTREND = "UPTREND"
    DOWNTREND = "DOWNTREND"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class SwingPoint:
    """
    Represents a swing high or swing low in the market.
    
    Attributes:
        type: 'high' or 'low'
        price: Price level of the swing
        bar_index: Index of the bar where swing occurred
        timestamp: Timestamp of the swing
        confirmed: Whether swing has been confirmed by subsequent bars
    """
    type: str  # 'high' or 'low'
    price: float
    bar_index: int
    timestamp: datetime
    confirmed: bool = False
    
    def __hash__(self):
        return hash((self.type, self.price, self.bar_index))


@dataclass
class StructureSignal:
    """
    Market structure signal detected by the engine.
    
    Attributes:
        signal_type: Type of structure (HH, HL, LH, LL, BoS, CHoCH)
        price: Price level of the signal
        bar_index: Bar index where signal occurred
        confidence: Confidence level 0.0-1.0
        details: Additional context about the signal
        timestamp: When signal was detected
    """
    signal_type: StructureType
    price: float
    bar_index: int
    confidence: float
    details: Dict
    timestamp: datetime


class StructureEngine:
    """
    Real-time Market Structure Detection Engine.
    
    Identifies market structure without visual charting by analyzing:
    - Swing highs and lows at multiple timeframes
    - Breaks of previous swings (BoS)
    - Changes in direction (CHoCH)
    - Higher highs/lows (uptrend)
    - Lower highs/lows (downtrend)
    
    Mathematical approach:
    1. Identify swing points (local extrema)
    2. Confirm swings after N bars pass without new extreme
    3. Compare current swings to previous swings
    4. Generate structure signals based on comparisons
    """
    
    def __init__(
        self,
        swing_length: int = 5,
        confirmation_bars: int = 2,
        use_hl2: bool = False
    ):
        """
        Initialize Structure Engine.
        
        Args:
            swing_length: Number of bars to left/right for swing identification
            confirmation_bars: Bars needed to confirm a swing
            use_hl2: Use (High+Low)/2 for swings, else use High/Low directly
        """
        self.swing_length = swing_length
        self.confirmation_bars = confirmation_bars
        self.use_hl2 = use_hl2
        
        # Cache for swing points
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.structure_signals: List[StructureSignal] = []
        
        logger.info(f"✓ StructureEngine initialized (swing_length={swing_length})")
    
    # ========================================================================
    # SWING IDENTIFICATION
    # ========================================================================
    
    def find_swing_highs(self, df: pd.DataFrame) -> np.ndarray:
        """
        Find swing high points (local maxima).
        
        A swing high is a price level where:
        - High is >= all prices within swing_length bars to left
        - High is >= all prices within swing_length bars to right
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Boolean array indicating swing high positions
        """
        highs = df['High'].values
        n = len(highs)
        is_swing_high = np.zeros(n, dtype=bool)
        
        for i in range(self.swing_length, n - self.swing_length):
            # Check if current high is >= all surrounding bars
            left_max = np.max(highs[i - self.swing_length:i])
            right_max = np.max(highs[i + 1:i + self.swing_length + 1])
            current_high = highs[i]
            
            # Swing high must be >= left and right sides
            if current_high >= left_max and current_high >= right_max:
                is_swing_high[i] = True
        
        return is_swing_high
    
    def find_swing_lows(self, df: pd.DataFrame) -> np.ndarray:
        """
        Find swing low points (local minima).
        
        A swing low is a price level where:
        - Low is <= all prices within swing_length bars to left
        - Low is <= all prices within swing_length bars to right
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Boolean array indicating swing low positions
        """
        lows = df['Low'].values
        n = len(lows)
        is_swing_low = np.zeros(n, dtype=bool)
        
        for i in range(self.swing_length, n - self.swing_length):
            # Check if current low is <= all surrounding bars
            left_min = np.min(lows[i - self.swing_length:i])
            right_min = np.min(lows[i + 1:i + self.swing_length + 1])
            current_low = lows[i]
            
            # Swing low must be <= left and right sides
            if current_low <= left_min and current_low <= right_min:
                is_swing_low[i] = True
        
        return is_swing_low
    
    def extract_swing_points(self, df: pd.DataFrame) -> Tuple[List[SwingPoint], List[SwingPoint]]:
        """
        Extract and confirm swing points from price data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Tuple of (swing_highs, swing_lows) as lists of SwingPoint objects
        """
        df = df.copy()
        swing_high_indices = self.find_swing_highs(df)
        swing_low_indices = self.find_swing_lows(df)
        
        swing_highs = []
        swing_lows = []
        
        # Extract swing highs
        for idx in np.where(swing_high_indices)[0]:
            price = df['High'].iloc[idx]
            timestamp = df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp(int(idx))
            
            # Confirm swing if N bars have passed
            confirmed = (len(df) - idx - 1) >= self.confirmation_bars
            
            swing_highs.append(SwingPoint(
                type='high',
                price=float(price),
                bar_index=int(idx),
                timestamp=timestamp,
                confirmed=confirmed
            ))
        
        # Extract swing lows
        for idx in np.where(swing_low_indices)[0]:
            price = df['Low'].iloc[idx]
            timestamp = df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp(int(idx))
            
            # Confirm swing if N bars have passed
            confirmed = (len(df) - idx - 1) >= self.confirmation_bars
            
            swing_lows.append(SwingPoint(
                type='low',
                price=float(price),
                bar_index=int(idx),
                timestamp=timestamp,
                confirmed=confirmed
            ))
        
        self.swing_highs = swing_highs
        self.swing_lows = swing_lows
        
        logger.info(f"✓ Identified {len(swing_highs)} swing highs and {len(swing_lows)} swing lows")
        
        return swing_highs, swing_lows
    
    # ========================================================================
    # STRUCTURE DETECTION: HH, HL, LH, LL
    # ========================================================================
    
    def detect_higher_high(
        self,
        current_high: float,
        prev_swing_high: SwingPoint
    ) -> bool:
        """
        Detect Higher High (HH): New swing high > previous swing high.
        
        Bullish signal - continuation of uptrend with increasing momentum.
        """
        return current_high > prev_swing_high.price
    
    def detect_higher_low(
        self,
        current_low: float,
        prev_swing_low: SwingPoint
    ) -> bool:
        """
        Detect Higher Low (HL): New swing low > previous swing low.
        
        Bullish signal - support is rising, showing buying pressure.
        """
        return current_low > prev_swing_low.price
    
    def detect_lower_high(
        self,
        current_high: float,
        prev_swing_high: SwingPoint
    ) -> bool:
        """
        Detect Lower High (LH): New swing high < previous swing high.
        
        Bearish signal - resistance is falling, showing selling pressure.
        """
        return current_high < prev_swing_high.price
    
    def detect_lower_low(
        self,
        current_low: float,
        prev_swing_low: SwingPoint
    ) -> bool:
        """
        Detect Lower Low (LL): New swing low < previous swing low.
        
        Bearish signal - continuation of downtrend with increasing momentum.
        """
        return current_low < prev_swing_low.price
    
    # ========================================================================
    # BREAK OF STRUCTURE (BoS)
    # ========================================================================
    
    def detect_break_of_structure(
        self,
        current_price: float,
        current_bar_high: float,
        current_bar_low: float,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Optional[StructureSignal]:
        """
        Detect Break of Structure (BoS).
        
        BoS occurs when:
        - In uptrend: Price breaks below the last swing low with conviction
        - In downtrend: Price breaks above the last swing high with conviction
        
        Indicates potential trend reversal or impulse phase.
        
        Args:
            current_price: Current market price
            current_bar_high: Current bar high
            current_bar_low: Current bar low
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            
        Returns:
            StructureSignal if BoS detected, None otherwise
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        last_swing_high = max(swing_highs, key=lambda x: x.bar_index)
        last_swing_low = min(swing_lows, key=lambda x: x.bar_index)
        
        # Downtrend BoS: Price closes above last swing high
        if current_bar_high > last_swing_high.price and current_bar_low >= last_swing_high.price:
            confidence = min(1.0, (current_bar_high - last_swing_high.price) / (last_swing_high.price * 0.01))
            
            return StructureSignal(
                signal_type=StructureType.BREAK_OF_STRUCTURE,
                price=last_swing_high.price,
                bar_index=len(swing_highs),
                confidence=confidence,
                details={
                    'direction': 'upside_break',
                    'structure_type': 'downtrend_BoS',
                    'break_distance_pct': ((current_bar_high - last_swing_high.price) / last_swing_high.price) * 100
                },
                timestamp=datetime.now()
            )
        
        # Uptrend BoS: Price closes below last swing low
        if current_bar_low < last_swing_low.price and current_bar_high <= last_swing_low.price:
            confidence = min(1.0, (last_swing_low.price - current_bar_low) / (last_swing_low.price * 0.01))
            
            return StructureSignal(
                signal_type=StructureType.BREAK_OF_STRUCTURE,
                price=last_swing_low.price,
                bar_index=len(swing_lows),
                confidence=confidence,
                details={
                    'direction': 'downside_break',
                    'structure_type': 'uptrend_BoS',
                    'break_distance_pct': ((last_swing_low.price - current_bar_low) / last_swing_low.price) * 100
                },
                timestamp=datetime.now()
            )
        
        return None
    
    # ========================================================================
    # CHANGE OF CHARACTER (CHoCH)
    # ========================================================================
    
    def detect_change_of_character(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Optional[StructureSignal]:
        """
        Detect Change of Character (CHoCH).
        
        CHoCH indicates shift from bullish to bearish structure or vice versa:
        - Bullish CHoCH: HH + HL -> LH (shift to bearish)
        - Bearish CHoCH: LL + LH -> HH (shift to bullish)
        
        Args:
            swing_highs: List of swing high points
            swing_lows: List of swing low points
            
        Returns:
            StructureSignal if CHoCH detected, None otherwise
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return None
        
        # Get last 3 of each type
        recent_highs = sorted(swing_highs, key=lambda x: x.bar_index)[-3:]
        recent_lows = sorted(swing_lows, key=lambda x: x.bar_index)[-3:]
        
        if len(recent_highs) < 2 or len(recent_lows) < 2:
            return None
        
        # ====================================================================
        # BULLISH CHoCH: HH + HL followed by LH (shift to bearish structure)
        # ====================================================================
        
        # Check for HH (each high > previous)
        if all(recent_highs[i].price > recent_highs[i - 1].price for i in range(1, len(recent_highs))):
            # Check for HL (each low > previous)
            if all(recent_lows[i].price > recent_lows[i - 1].price for i in range(1, len(recent_lows))):
                # Now check if we're about to form LH (next high would be lower)
                # This is detected when price action suggests a potential lower high
                
                last_high = recent_highs[-1]
                prev_high = recent_highs[-2]
                
                return StructureSignal(
                    signal_type=StructureType.CHANGE_OF_CHARACTER,
                    price=last_high.price,
                    bar_index=last_high.bar_index,
                    confidence=0.75,
                    details={
                        'from_structure': 'uptrend (HH+HL)',
                        'to_structure': 'downtrend (potential LH)',
                        'swing_high_price': last_high.price,
                        'previous_high': prev_high.price
                    },
                    timestamp=datetime.now()
                )
        
        # ====================================================================
        # BEARISH CHoCH: LL + LH followed by HH (shift to bullish structure)
        # ====================================================================
        
        # Check for LL (each low < previous)
        if all(recent_lows[i].price < recent_lows[i - 1].price for i in range(1, len(recent_lows))):
            # Check for LH (each high < previous)
            if all(recent_highs[i].price < recent_highs[i - 1].price for i in range(1, len(recent_highs))):
                # Now potential for HH
                
                last_low = recent_lows[-1]
                prev_low = recent_lows[-2]
                
                return StructureSignal(
                    signal_type=StructureType.CHANGE_OF_CHARACTER,
                    price=last_low.price,
                    bar_index=last_low.bar_index,
                    confidence=0.75,
                    details={
                        'from_structure': 'downtrend (LL+LH)',
                        'to_structure': 'uptrend (potential HH)',
                        'swing_low_price': last_low.price,
                        'previous_low': prev_low.price
                    },
                    timestamp=datetime.now()
                )
        
        return None
    
    # ========================================================================
    # TREND DETERMINATION
    # ========================================================================
    
    def determine_trend(
        self,
        swing_highs: List[SwingPoint],
        swing_lows: List[SwingPoint]
    ) -> Tuple[Trend, Dict]:
        """
        Determine current market trend using swing structure.
        
        Args:
            swing_highs: List of confirmed swing highs
            swing_lows: List of confirmed swing lows
            
        Returns:
            Tuple of (Trend, details_dict)
        """
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return Trend.UNCERTAIN, {'reason': 'insufficient_swings'}
        
        # Get last 2 of each
        recent_highs = sorted(swing_highs, key=lambda x: x.bar_index)[-2:]
        recent_lows = sorted(swing_lows, key=lambda x: x.bar_index)[-2:]
        
        # Uptrend: HH + HL
        if recent_highs[1].price > recent_highs[0].price and recent_lows[1].price > recent_lows[0].price:
            return Trend.UPTREND, {
                'structure': 'HH + HL',
                'last_high': recent_highs[1].price,
                'last_low': recent_lows[1].price
            }
        
        # Downtrend: LL + LH
        if recent_lows[1].price < recent_lows[0].price and recent_highs[1].price < recent_highs[0].price:
            return Trend.DOWNTREND, {
                'structure': 'LL + LH',
                'last_high': recent_highs[1].price,
                'last_low': recent_lows[1].price
            }
        
        return Trend.UNCERTAIN, {'reason': 'mixed_structure'}
    
    # ========================================================================
    # MAIN ANALYSIS FUNCTION
    # ========================================================================
    
    def analyze(self, df: pd.DataFrame) -> Dict:
        """
        Complete structural analysis of price data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            Dictionary with all structural analysis results
        """
        # Extract swings
        swing_highs, swing_lows = self.extract_swing_points(df)
        
        # Determine trend
        trend, trend_details = self.determine_trend(swing_highs, swing_lows)
        
        # Detect CHoCH
        choch = self.detect_change_of_character(swing_highs, swing_lows)
        
        # Detect BoS
        if len(df) > 0:
            current_bar = df.iloc[-1]
            bos = self.detect_break_of_structure(
                current_price=current_bar['Close'],
                current_bar_high=current_bar['High'],
                current_bar_low=current_bar['Low'],
                swing_highs=swing_highs,
                swing_lows=swing_lows
            )
        else:
            bos = None
        
        return {
            'swing_highs': swing_highs,
            'swing_lows': swing_lows,
            'trend': trend,
            'trend_details': trend_details,
            'change_of_character': choch,
            'break_of_structure': bos,
            'timestamp': datetime.now()
        }

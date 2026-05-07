# ============================================================================
# VOLUME PROFILE ENGINE - L2 INTELLIGENCE LAYER
# Calculates Point of Control (POC), Value Area High (VAH), Value Area Low (VAL)
# ============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class VolumeProfileResult:
    """
    Result of volume profile analysis.
    
    Attributes:
        poc: Point of Control - price level with highest volume
        poc_volume: Volume at Point of Control
        vah: Value Area High - upper boundary of 70% volume
        val: Value Area Low - lower boundary of 70% volume
        profile: Dict mapping price levels to volume
        cumulative_volume: Running sum of volume from lowest to highest
        volume_percentage: Dict mapping price levels to volume %
    """
    poc: float
    poc_volume: float
    vah: float
    val: float
    profile: Dict[float, float]
    cumulative_volume: np.ndarray
    volume_percentage: Dict[float, float]
    timestamp: datetime


class VolumeProfile:
    """
    Volume Profile Calculator for Market Microstructure Analysis.
    
    Calculates bid-ask volume distribution across price levels, identifying:
    - Point of Control (POC): Price level with maximum volume
    - Value Area: Range containing 70% of volume
    - Volume At Price (VAP): Volume at each price level
    
    This is used to identify:
    - Support/resistance levels based on actual volume
    - High volume nodes where price tends to interact
    - Low volume nodes (gaps) where price accelerates through
    - Initial Balance Range (IBR) for institutional entry points
    """
    
    def __init__(self, price_precision: int = 2, min_volume_threshold: float = 0.0):
        """
        Initialize Volume Profile Calculator.
        
        Args:
            price_precision: Number of decimal places for price bucketing (2 = $0.01)
            min_volume_threshold: Minimum volume to include in profile (filters noise)
        """
        self.price_precision = price_precision
        self.min_volume_threshold = min_volume_threshold
        self.profile_cache: Dict[str, VolumeProfileResult] = {}
        logger.info(f"✓ VolumeProfile initialized (precision: {price_precision})")
    
    # ========================================================================
    # CORE VOLUME PROFILE CALCULATION
    # ========================================================================
    
    def calculate(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume',
        high_column: str = 'High',
        low_column: str = 'Low',
        intrabar_levels: int = 10
    ) -> VolumeProfileResult:
        """
        Calculate volume profile from OHLCV data.
        
        For each bar, volume is distributed across the high-low range:
        - If bar is bullish (Close > Open): More weight on upper half
        - If bar is bearish (Close < Open): More weight on lower half
        - Volume buckets are created at each price level
        
        Args:
            df: DataFrame with OHLCV data
            price_column: Column name for price
            volume_column: Column name for volume
            high_column: Column name for high price
            low_column: Column name for low price
            intrabar_levels: Number of price levels to subdivide each bar
            
        Returns:
            VolumeProfileResult with POC, VAH, VAL, and full profile
        """
        df = df.copy().dropna(subset=[volume_column, high_column, low_column, price_column])
        
        if df.empty:
            logger.warning("⚠️  Empty dataframe provided to volume profile")
            raise ValueError("Cannot calculate volume profile from empty dataframe")
        
        # Initialize price-to-volume mapping
        profile = {}
        
        # ====================================================================
        # DISTRIBUTE VOLUME ACROSS INTRABAR PRICE LEVELS
        # ====================================================================
        
        for idx, row in df.iterrows():
            high = row[high_column]
            low = row[low_column]
            close = row[price_column]
            volume = row[volume_column]
            
            # Skip invalid bars
            if high <= low or volume <= 0:
                continue
            
            # Generate price levels within the bar
            price_levels = np.linspace(low, high, intrabar_levels)
            
            # Distribute volume based on close position relative to open/high/low
            # Closer to close = more volume (price discovery)
            open_price = row.get('Open', (high + low) / 2)
            
            # Calculate volume weights for each price level
            is_bullish = close >= open_price
            
            if is_bullish:
                # Bullish bars: concentrate volume in upper half
                # Distance from low normalized to [0, 1]
                distances = (price_levels - low) / (high - low + 1e-9)
                # Square the distance to emphasize upper half
                weights = distances ** 2
            else:
                # Bearish bars: concentrate volume in lower half
                distances = (high - price_levels) / (high - low + 1e-9)
                weights = distances ** 2
            
            # Normalize weights to sum to 1
            weights = weights / (weights.sum() + 1e-9)
            
            # Distribute volume according to weights
            volume_distribution = volume * weights
            
            # Add to profile (bucket by price_precision)
            for price_level, vol in zip(price_levels, volume_distribution):
                # Round to price precision
                bucketed_price = round(price_level, self.price_precision)
                profile[bucketed_price] = profile.get(bucketed_price, 0) + vol
        
        # Filter out noise below threshold
        profile = {p: v for p, v in profile.items() if v >= self.min_volume_threshold}
        
        if not profile:
            raise ValueError("No valid volume profile generated after filtering")
        
        # ====================================================================
        # CALCULATE POINT OF CONTROL (POC)
        # ====================================================================
        
        # POC is the price level with highest volume
        poc = max(profile, key=profile.get)
        poc_volume = profile[poc]
        
        # ====================================================================
        # CALCULATE VALUE AREA (70% Volume)
        # ====================================================================
        
        # Sort prices by volume (descending)
        sorted_prices = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        
        # Accumulate volume until we reach 70% of total
        total_volume = sum(profile.values())
        target_volume = total_volume * 0.70
        
        value_area_prices = []
        accumulated_volume = 0
        
        for price, vol in sorted_prices:
            value_area_prices.append(price)
            accumulated_volume += vol
            if accumulated_volume >= target_volume:
                break
        
        # Value Area High (VAH) and Value Area Low (VAL)
        vah = max(value_area_prices)
        val = min(value_area_prices)
        
        # ====================================================================
        # CALCULATE VOLUME PERCENTAGES & CUMULATIVE VOLUME
        # ====================================================================
        
        volume_percentage = {p: (v / total_volume) * 100 for p, v in profile.items()}
        
        # Cumulative volume from lowest to highest price
        sorted_profile_prices = sorted(profile.keys())
        cumulative_volume = np.cumsum([profile[p] for p in sorted_profile_prices])
        
        result = VolumeProfileResult(
            poc=poc,
            poc_volume=poc_volume,
            vah=vah,
            val=val,
            profile=profile,
            cumulative_volume=cumulative_volume,
            volume_percentage=volume_percentage,
            timestamp=datetime.now()
        )
        
        logger.info(f"✓ Volume Profile calculated:")
        logger.info(f"  - POC: ${poc:.{self.price_precision}f} (Volume: {poc_volume:,.0f})")
        logger.info(f"  - VAH: ${vah:.{self.price_precision}f}")
        logger.info(f"  - VAL: ${val:.{self.price_precision}f}")
        logger.info(f"  - Value Area Width: ${(vah - val):.{self.price_precision}f}")
        logger.info(f"  - Total Volume: {total_volume:,.0f}")
        
        return result
    
    # ========================================================================
    # VOLUME PROFILE ANALYSIS
    # ========================================================================
    
    def get_high_volume_nodes(
        self,
        profile: Dict[float, float],
        percentile: float = 75.0
    ) -> List[Tuple[float, float]]:
        """
        Identify high volume nodes (support/resistance levels).
        
        Args:
            profile: Volume profile dictionary
            percentile: Volume percentile threshold (75 = top 25% most liquid levels)
            
        Returns:
            List of (price, volume) tuples for high volume nodes
        """
        volumes = np.array(list(profile.values()))
        threshold = np.percentile(volumes, percentile)
        
        high_volume_nodes = [
            (price, vol) for price, vol in profile.items()
            if vol >= threshold
        ]
        
        return sorted(high_volume_nodes, key=lambda x: x[0])
    
    def get_low_volume_nodes(
        self,
        profile: Dict[float, float],
        percentile: float = 25.0
    ) -> List[Tuple[float, float]]:
        """
        Identify low volume nodes (areas of acceleration).
        
        Args:
            profile: Volume profile dictionary
            percentile: Volume percentile threshold (25 = bottom 25% least liquid levels)
            
        Returns:
            List of (price, volume) tuples for low volume nodes
        """
        volumes = np.array(list(profile.values()))
        threshold = np.percentile(volumes, percentile)
        
        low_volume_nodes = [
            (price, vol) for price, vol in profile.items()
            if vol <= threshold
        ]
        
        return sorted(low_volume_nodes, key=lambda x: x[0])
    
    def get_value_area_width(self, vah: float, val: float) -> float:
        """
        Calculate Value Area width (VAH - VAL).
        
        Narrow value area = High conviction (tight trading range)
        Wide value area = Low conviction (mixed sentiment)
        
        Args:
            vah: Value Area High
            val: Value Area Low
            
        Returns:
            Width of value area in price units
        """
        return vah - val
    
    def get_poc_distance_to_price(self, poc: float, current_price: float) -> float:
        """
        Calculate distance of current price from Point of Control.
        
        Positive = Price above POC
        Negative = Price below POC
        
        Args:
            poc: Point of Control price
            current_price: Current market price
            
        Returns:
            Distance from POC (positive/negative)
        """
        return current_price - poc
    
    def get_volume_profile_levels(
        self,
        profile: Dict[float, float],
        num_levels: int = 5
    ) -> List[float]:
        """
        Get top N most significant volume levels (potential S/R).
        
        Args:
            profile: Volume profile dictionary
            num_levels: Number of top levels to return
            
        Returns:
            List of prices with highest volume
        """
        sorted_prices = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        return [price for price, _ in sorted_prices[:num_levels]]
    
    # ========================================================================
    # INSTITUTIONAL FOOTPRINT DETECTION
    # ========================================================================
    
    def detect_auction_ranges(
        self,
        df: pd.DataFrame,
        profile: VolumeProfileResult,
        tolerance: float = 0.02
    ) -> Dict[str, list]:
        """
        Detect and identify market auction ranges using volume profile.
        
        Auctions are price ranges where market participants traded heavily.
        Multiple auctions indicate competing institutional interest at those levels.
        
        Args:
            df: OHLCV DataFrame
            profile: VolumeProfileResult from calculate()
            tolerance: % tolerance for price consolidation (2% = ±2%)
            
        Returns:
            Dict with 'ranges' (list of auction ranges) and 'confidence' scores
        """
        high_volume_levels = self.get_high_volume_nodes(profile.profile, percentile=70)
        
        if not high_volume_levels:
            return {'ranges': [], 'confidence': []}
        
        auction_ranges = []
        prices = [p for p, v in high_volume_levels]
        
        current_cluster = [prices[0]]
        
        for price in prices[1:]:
            # If price is within tolerance of current cluster, add it
            cluster_avg = np.mean(current_cluster)
            if abs(price - cluster_avg) / cluster_avg <= tolerance:
                current_cluster.append(price)
            else:
                # Start new cluster
                if current_cluster:
                    auction_ranges.append({
                        'low': min(current_cluster),
                        'high': max(current_cluster),
                        'center': np.mean(current_cluster),
                        'volume': sum(profile.profile.get(p, 0) for p in current_cluster)
                    })
                current_cluster = [price]
        
        # Don't forget last cluster
        if current_cluster:
            auction_ranges.append({
                'low': min(current_cluster),
                'high': max(current_cluster),
                'center': np.mean(current_cluster),
                'volume': sum(profile.profile.get(p, 0) for p in current_cluster)
            })
        
        # Sort by volume descending
        auction_ranges.sort(key=lambda x: x['volume'], reverse=True)
        
        # Calculate confidence based on volume concentration
        confidence_scores = [
            (r['volume'] / sum(rng['volume'] for rng in auction_ranges)) * 100
            for r in auction_ranges
        ]
        
        return {
            'ranges': auction_ranges,
            'confidence': confidence_scores
        }
    
    def detect_breakout_zones(
        self,
        profile: VolumeProfileResult,
        sensitivity: float = 0.5
    ) -> Dict[str, list]:
        """
        Identify low-volume breakout zones where price accelerates.
        
        Args:
            profile: VolumeProfileResult
            sensitivity: Sensitivity for zone detection (0.0-1.0, higher = more sensitive)
            
        Returns:
            Dict with breakout zones and acceleration potential
        """
        low_volume_nodes = self.get_low_volume_nodes(profile.profile, percentile=20 + (sensitivity * 30))
        
        breakout_zones = []
        prices = sorted([p for p, _ in low_volume_nodes])
        
        # Group consecutive low volume levels
        current_zone = [prices[0]] if prices else []
        
        for price in prices[1:]:
            if abs(price - current_zone[-1]) <= (profile.vah - profile.val) * 0.05:
                current_zone.append(price)
            else:
                if current_zone:
                    avg_vol = np.mean([profile.profile.get(p, 0) for p in current_zone])
                    breakout_zones.append({
                        'low': min(current_zone),
                        'high': max(current_zone),
                        'avg_volume': avg_vol,
                        'acceleration_potential': 1.0 - (avg_vol / np.mean(list(profile.profile.values())))
                    })
                current_zone = [price]
        
        return {'zones': breakout_zones}

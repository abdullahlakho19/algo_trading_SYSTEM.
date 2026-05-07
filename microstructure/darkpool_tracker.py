# ============================================================================
# DARKPOOL TRACKER - L2 MICROSTRUCTURE LAYER
# Institutional bias detection from COT reports, block trades, and market data
# ============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# ENUMERATIONS & DATA MODELS
# ============================================================================

class InstitutionalAction(Enum):
    """Types of institutional footprints."""
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    POSITIONING = "positioning"
    LIQUIDATION = "liquidation"
    UNKNOWN = "unknown"


@dataclass
class InstitutionalSignal:
    """
    Institutional activity signal.
    
    Attributes:
        action_type: Type of institutional activity
        strength: Signal strength 0.0-1.0
        price_level: Price where activity detected
        volume_involved: Estimated volume
        timestamp: When signal detected
        sources: List of data sources confirming signal
    """
    action_type: InstitutionalAction
    strength: float
    price_level: float
    volume_involved: float
    timestamp: datetime
    sources: List[str]


@dataclass
class InstitutionalBiasScore:
    """
    Institutional bias score aggregating all signals.
    
    Attributes:
        bias_score: Net bias -1 (strong sell) to +1 (strong buy)
        long_bias: Long positioning bias 0-1
        short_bias: Short positioning bias 0-1
        accumulation_score: Accumulation activity 0-1
        distribution_score: Distribution activity 0-1
        conviction_level: Overall conviction 0-1
        signals_count: Number of confirmed signals
        timestamp: When calculated
    """
    bias_score: float
    long_bias: float
    short_bias: float
    accumulation_score: float
    distribution_score: float
    conviction_level: float
    signals_count: int
    timestamp: datetime


class DarkpoolTracker:
    """
    Institutional Footprint Tracker using Multiple Data Sources.
    
    Analyzes:
    - COT (Commitment of Traders) reports (futures positioning)
    - Block trade flow (large institutional orders)
    - Order book imbalance persistence
    - Volume patterns (absorption, distribution)
    - Price action confirmation
    
    Mathematical approach:
    1. Gather institutional indicators from multiple sources
    2. Normalize each signal to -1 to +1 scale
    3. Weight signals by conviction/reliability
    4. Aggregate into institutional bias score
    5. Detect divergences between bias and price
    """
    
    def __init__(self, lookback_days: int = 20, min_signals: int = 2):
        """
        Initialize Darkpool Tracker.
        
        Args:
            lookback_days: Number of days to analyze for patterns
            min_signals: Minimum signals required for high conviction
        """
        self.lookback_days = lookback_days
        self.min_signals = min_signals
        
        # Historical signals cache
        self.signals_history: List[InstitutionalSignal] = []
        
        logger.info(f"✓ DarkpoolTracker initialized (lookback={lookback_days} days)")
    
    # ========================================================================
    # COT REPORT ANALYSIS
    # ========================================================================
    
    def analyze_cot_positioning(
        self,
        commercials_net: float,
        large_spec_net: float,
        small_spec_net: float,
        open_interest: float
    ) -> Dict:
        """
        Analyze Commitment of Traders (COT) report positioning.
        
        COT reports show positioning of:
        - Commercials: Hedgers (producers/consumers) - trend-following
        - Large Specs: Hedge funds, large traders - trend-following
        - Small Specs: Retail traders - often dead wrong at extremes
        
        Args:
            commercials_net: Commercial net position (contracts)
            large_spec_net: Large speculator net position (contracts)
            small_spec_net: Small speculator net position (contracts)
            open_interest: Total open interest (contracts)
            
        Returns:
            Dict with bullish/bearish bias signals from COT
        """
        # Normalize to percentage of open interest
        comm_pct = (commercials_net / open_interest) * 100 if open_interest > 0 else 0
        large_spec_pct = (large_spec_net / open_interest) * 100 if open_interest > 0 else 0
        small_spec_pct = (small_spec_net / open_interest) * 100 if open_interest > 0 else 0
        
        # Interpretation:
        # - Commercials short = bullish (hedging commercial output)
        # - Large specs long = bullish trend-following
        # - Small specs long = contrarian bearish (retail usually wrong at extremes)
        
        signals = []
        scores = []
        
        # ====================================================================
        # COMMERCIAL POSITIONING
        # ====================================================================
        
        # Commercials are hedgers - when they're short, it's bullish
        # (They're hedging future production, so market needs to rally to buy it)
        if commercials_net < 0:
            signals.append('commercials_short_bias')
            # Extreme shorts are more bullish
            comm_signal = min(1.0, abs(comm_pct) / 30.0)  # Extreme at ±30%
            scores.append(comm_signal)
        else:
            signals.append('commercials_long_bias')
            comm_signal = -min(1.0, abs(comm_pct) / 30.0)
            scores.append(comm_signal)
        
        # ====================================================================
        # LARGE SPECULATOR POSITIONING (Smart Money)
        # ====================================================================
        
        # Large specs are trend-following and generally correct
        if large_spec_net > 0:
            signals.append('large_specs_long_bias')
            large_spec_signal = min(1.0, abs(large_spec_pct) / 50.0)  # Extreme at ±50%
            scores.append(large_spec_signal)
        else:
            signals.append('large_specs_short_bias')
            large_spec_signal = -min(1.0, abs(large_spec_pct) / 50.0)
            scores.append(large_spec_signal)
        
        # ====================================================================
        # SMALL SPECULATOR POSITIONING (Retail - Contrarian)
        # ====================================================================
        
        # Small specs are often wrong at extremes - use as contrarian indicator
        if small_spec_net > 0:
            signals.append('small_specs_long_extreme')
            # When retail is extremely long (>40%), this is bearish (tops-out)
            small_spec_signal = -min(1.0, abs(small_spec_pct) / 40.0)
            scores.append(small_spec_signal)
        else:
            signals.append('small_specs_short_extreme')
            # When retail is extremely short (<-40%), this is bullish (reversal)
            small_spec_signal = min(1.0, abs(small_spec_pct) / 40.0)
            scores.append(small_spec_signal)
        
        # ====================================================================
        # NET SCORE
        # ====================================================================
        
        net_score = np.mean(scores) if scores else 0.0
        
        # Clip to -1 to +1
        net_score = np.clip(net_score, -1.0, 1.0)
        
        return {
            'net_score': net_score,
            'commercial_signal': comm_signal,
            'large_spec_signal': large_spec_signal,
            'small_spec_signal': small_spec_signal,
            'signals': signals,
            'commercials_net': commercials_net,
            'large_specs_net': large_spec_net,
            'small_specs_net': small_spec_net
        }
    
    # ========================================================================
    # BLOCK TRADE ANALYSIS
    # ========================================================================
    
    def analyze_block_trade_flow(
        self,
        block_trades: List[Dict],
        lookback_hours: int = 24
    ) -> Dict:
        """
        Analyze institutional block trade flow patterns.
        
        Large block trades indicate institutional activity:
        - Accumulation: Blocks on dips with upside price action
        - Distribution: Blocks on rallies with downside price action
        - Positioning: Sustained block activity in one direction
        
        Args:
            block_trades: List of block trade dicts with volume, direction, price
            lookback_hours: Hours to look back for analysis
            
        Returns:
            Dict with block trade analysis and institutional bias
        """
        if not block_trades:
            return {
                'net_score': 0.0,
                'buy_blocks': 0,
                'sell_blocks': 0,
                'buy_volume': 0.0,
                'sell_volume': 0.0,
                'institutional_bias': 'neutral'
            }
        
        # Filter recent blocks
        recent_blocks = block_trades[-5:] if len(block_trades) > 5 else block_trades
        
        # Count and sum volumes by direction
        buy_blocks = sum(1 for b in recent_blocks if b.get('direction') == 'buy')
        sell_blocks = sum(1 for b in recent_blocks if b.get('direction') == 'sell')
        
        buy_volume = sum(b.get('volume', 0) for b in recent_blocks if b.get('direction') == 'buy')
        sell_volume = sum(b.get('volume', 0) for b in recent_blocks if b.get('direction') == 'sell')
        
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return {'net_score': 0.0, 'institutional_bias': 'neutral'}
        
        # Calculate bias: +1 = all buying, -1 = all selling
        volume_bias = (buy_volume - sell_volume) / total_volume
        tick_bias = (buy_blocks - sell_blocks) / (buy_blocks + sell_blocks + 1e-9)
        
        # Combine volume and tick bias
        net_score = (volume_bias * 0.7 + tick_bias * 0.3)
        
        if net_score > 0.3:
            institutional_bias = 'accumulation'
        elif net_score < -0.3:
            institutional_bias = 'distribution'
        else:
            institutional_bias = 'neutral'
        
        return {
            'net_score': net_score,
            'buy_blocks': buy_blocks,
            'sell_blocks': sell_blocks,
            'buy_volume': buy_volume,
            'sell_volume': sell_volume,
            'institutional_bias': institutional_bias,
            'volume_bias': volume_bias,
            'tick_bias': tick_bias
        }
    
    # ========================================================================
    # VOLUME CLUSTER ANALYSIS
    # ========================================================================
    
    def detect_volume_clusters(
        self,
        df: pd.DataFrame,
        volume_column: str = 'Volume',
        lookback: int = 20
    ) -> Dict:
        """
        Detect sustained volume clusters indicating institutional activity.
        
        Volume clusters = Multiple periods of elevated volume in same direction.
        This suggests institutional accumulation/distribution (not just volatility).
        
        Args:
            df: OHLCV DataFrame
            volume_column: Column name for volume
            lookback: Number of bars to analyze
            
        Returns:
            Dict with cluster analysis and bias
        """
        df_slice = df.tail(lookback).copy()
        
        # Calculate volume statistics
        mean_volume = df_slice[volume_column].mean()
        std_volume = df_slice[volume_column].std()
        
        # Define cluster: volume > mean + 0.5*std
        threshold = mean_volume + (0.5 * std_volume)
        
        df_slice['is_high_volume'] = df_slice[volume_column] > threshold
        df_slice['direction'] = np.where(df_slice['Close'] > df_slice['Open'], 1, -1)
        
        # Identify clusters (consecutive high volume bars in same direction)
        df_slice['direction_change'] = df_slice['direction'].diff().fillna(0)
        
        # Group by cluster
        df_slice['cluster_id'] = (df_slice['is_high_volume'] != df_slice['is_high_volume'].shift()).cumsum()
        
        clusters = []
        for cluster_id, group in df_slice.groupby('cluster_id'):
            if group['is_high_volume'].iloc[0]:  # Only interested in high-volume clusters
                direction = group['direction'].iloc[0]
                cluster_volume = group[volume_column].sum()
                cluster_length = len(group)
                avg_price_move = abs(group['Close'].iloc[-1] - group['Close'].iloc[0])
                
                clusters.append({
                    'direction': 'buy' if direction > 0 else 'sell',
                    'volume': cluster_volume,
                    'length': cluster_length,
                    'avg_price_move': avg_price_move,
                    'efficiency': avg_price_move / (cluster_volume + 1e-9)  # Price per unit volume
                })
        
        # Analyze cluster patterns
        if not clusters:
            return {'net_score': 0.0, 'cluster_pattern': 'none'}
        
        buy_clusters = [c for c in clusters if c['direction'] == 'buy']
        sell_clusters = [c for c in clusters if c['direction'] == 'sell']
        
        buy_volume_total = sum(c['volume'] for c in buy_clusters)
        sell_volume_total = sum(c['volume'] for c in sell_clusters)
        
        volume_bias = (buy_volume_total - sell_volume_total) / max(buy_volume_total + sell_volume_total, 1e-9)
        
        # Persistence: more clusters in one direction = stronger bias
        persistence = abs(len(buy_clusters) - len(sell_clusters)) / (len(buy_clusters) + len(sell_clusters) + 1e-9)
        
        # Efficiency: large moves with less volume = smart accumulation
        avg_efficiency_buy = np.mean([c['efficiency'] for c in buy_clusters]) if buy_clusters else 0
        avg_efficiency_sell = np.mean([c['efficiency'] for c in sell_clusters]) if sell_clusters else 0
        
        efficiency_bias = (avg_efficiency_buy - avg_efficiency_sell) / max(abs(avg_efficiency_buy) + abs(avg_efficiency_sell), 1e-9)
        
        net_score = (volume_bias * 0.5 + persistence * 0.3 + efficiency_bias * 0.2)
        
        if len(buy_clusters) > len(sell_clusters):
            cluster_pattern = 'accumulation'
        elif len(sell_clusters) > len(buy_clusters):
            cluster_pattern = 'distribution'
        else:
            cluster_pattern = 'balanced'
        
        return {
            'net_score': np.clip(net_score, -1.0, 1.0),
            'cluster_pattern': cluster_pattern,
            'buy_clusters': len(buy_clusters),
            'sell_clusters': len(sell_clusters),
            'volume_bias': volume_bias,
            'persistence': persistence
        }
    
    # ========================================================================
    # ORDER BOOK ANALYSIS (Level 2 Data)
    # ========================================================================
    
    def analyze_level2_footprint(
        self,
        bid_volumes: List[float],
        ask_volumes: List[float],
        bid_prices: List[float],
        ask_prices: List[float],
        levels: int = 5
    ) -> Dict:
        """
        Analyze Level 2 order book for institutional footprints.
        
        Institutional activity shows as:
        - Iceberg orders: Large orders hidden, releasing in tranches
        - Wall building: Large orders not meant to fill, just to influence price
        - Absorption: Orders getting eaten up without price moving
        
        Args:
            bid_volumes: List of bid volumes at each price level
            ask_volumes: List of ask volumes at each price level
            bid_prices: List of bid prices
            ask_prices: List of ask prices
            levels: Number of price levels to analyze (typically 5-10)
            
        Returns:
            Dict with institutional footprint analysis
        """
        levels = min(levels, len(bid_volumes), len(ask_volumes))
        
        total_bid = sum(bid_volumes[:levels])
        total_ask = sum(ask_volumes[:levels])
        
        # Get top levels (closest to spread)
        top_bid = bid_volumes[0] if bid_volumes else 0
        top_ask = ask_volumes[0] if ask_volumes else 0
        
        # Imbalance at best bid/ask
        top_level_imbalance = (top_bid - top_ask) / max(top_bid + top_ask, 1e-9)
        
        # Depth skew: If one side is significantly deeper
        depth_skew = (total_bid - total_ask) / max(total_bid + total_ask, 1e-9)
        
        # Wall detection: Massive order at one level
        bid_max = max(bid_volumes[:levels]) if bid_volumes else 0
        ask_max = max(ask_volumes[:levels]) if ask_volumes else 0
        
        bid_avg = total_bid / levels if levels > 0 else 0
        ask_avg = total_ask / levels if levels > 0 else 0
        
        # Wall score: 1.0 = definite wall, 0.0 = no wall
        bid_wall_score = min(1.0, (bid_max / bid_avg - 1.0) / 3.0) if bid_avg > 0 else 0
        ask_wall_score = min(1.0, (ask_max / ask_avg - 1.0) / 3.0) if ask_avg > 0 else 0
        
        # Determine institutional action
        if ask_wall_score > bid_wall_score:
            action = 'resistance_wall'  # Sellers defending
            bias = -ask_wall_score
        elif bid_wall_score > ask_wall_score:
            action = 'support_wall'  # Buyers defending
            bias = bid_wall_score
        else:
            action = 'balanced'
            bias = (top_level_imbalance + depth_skew) / 2.0
        
        return {
            'action': action,
            'bias': np.clip(bias, -1.0, 1.0),
            'top_level_imbalance': top_level_imbalance,
            'depth_skew': depth_skew,
            'bid_wall_score': bid_wall_score,
            'ask_wall_score': ask_wall_score,
            'total_bid_depth': total_bid,
            'total_ask_depth': total_ask
        }
    
    # ========================================================================
    # AGGREGATE INSTITUTIONAL BIAS SCORE
    # ========================================================================
    
    def calculate_institutional_bias(
        self,
        cot_data: Optional[Dict] = None,
        block_trades: Optional[List[Dict]] = None,
        df: Optional[pd.DataFrame] = None,
        level2_data: Optional[Dict] = None
    ) -> InstitutionalBiasScore:
        """
        Calculate aggregate institutional bias score from multiple sources.
        
        Args:
            cot_data: COT positioning data from analyze_cot_positioning()
            block_trades: List of block trades
            df: OHLCV DataFrame for volume cluster analysis
            level2_data: Level 2 order book data
            
        Returns:
            InstitutionalBiasScore
        """
        scores = []
        signal_count = 0
        
        long_bias_scores = []
        short_bias_scores = []
        
        # ====================================================================
        # COT COMPONENT
        # ====================================================================
        
        if cot_data:
            cot_score = cot_data.get('net_score', 0.0)
            scores.append(cot_score)
            signal_count += 1
            
            if cot_score > 0:
                long_bias_scores.append(cot_score)
            else:
                short_bias_scores.append(abs(cot_score))
        
        # ====================================================================
        # BLOCK TRADE COMPONENT
        # ====================================================================
        
        block_analysis = self.analyze_block_trade_flow(block_trades or [])
        block_score = block_analysis.get('net_score', 0.0)
        if block_trades:
            scores.append(block_score)
            signal_count += 1
            
            if block_score > 0:
                long_bias_scores.append(block_score)
            else:
                short_bias_scores.append(abs(block_score))
        
        # ====================================================================
        # VOLUME CLUSTER COMPONENT
        # ====================================================================
        
        if df is not None and len(df) > 0:
            volume_analysis = self.detect_volume_clusters(df)
            volume_score = volume_analysis.get('net_score', 0.0)
            scores.append(volume_score)
            signal_count += 1
            
            if volume_score > 0:
                long_bias_scores.append(volume_score)
            else:
                short_bias_scores.append(abs(volume_score))
        
        # ====================================================================
        # LEVEL 2 COMPONENT
        # ====================================================================
        
        if level2_data:
            level2_score = level2_data.get('bias', 0.0)
            scores.append(level2_score)
            signal_count += 1
            
            if level2_score > 0:
                long_bias_scores.append(level2_score)
            else:
                short_bias_scores.append(abs(level2_score))
        
        # ====================================================================
        # AGGREGATE SCORES
        # ====================================================================
        
        # Main bias score: -1 (strong sell) to +1 (strong buy)
        bias_score = np.mean(scores) if scores else 0.0
        
        # Long/Short biases (0 to 1 scale)
        long_bias = np.mean(long_bias_scores) if long_bias_scores else 0.0
        short_bias = np.mean(short_bias_scores) if short_bias_scores else 0.0
        
        # Accumulation/Distribution scores
        accumulation_score = max(0.0, bias_score) if bias_score > 0 else 0.0
        distribution_score = max(0.0, abs(bias_score)) if bias_score < 0 else 0.0
        
        # Conviction level: Higher when multiple signals align
        agreement = 1.0 - (np.std(scores) if scores else 1.0)
        conviction_level = (abs(bias_score) * 0.7 + max(0, agreement) * 0.3)
        
        return InstitutionalBiasScore(
            bias_score=np.clip(bias_score, -1.0, 1.0),
            long_bias=np.clip(long_bias, 0.0, 1.0),
            short_bias=np.clip(short_bias, 0.0, 1.0),
            accumulation_score=np.clip(accumulation_score, 0.0, 1.0),
            distribution_score=np.clip(distribution_score, 0.0, 1.0),
            conviction_level=np.clip(conviction_level, 0.0, 1.0),
            signals_count=signal_count,
            timestamp=datetime.now()
        )

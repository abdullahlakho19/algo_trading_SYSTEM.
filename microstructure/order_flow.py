# ============================================================================
# ORDER FLOW ENGINE - L2 MICROSTRUCTURE LAYER
# Analyzes bid/ask delta, order book imbalance, and buying/selling pressure
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
class OrderFlowMetrics:
    """
    Order flow analysis metrics.
    
    Attributes:
        bid_volume: Total volume at bid side
        ask_volume: Total volume at ask side
        bid_ask_ratio: bid_volume / ask_volume
        bid_ask_delta: bid_volume - ask_volume (signed)
        buying_pressure: Normalized buying pressure 0-100
        selling_pressure: Normalized selling pressure 0-100
        net_flow: Net buying/selling pressure (-1 to +1)
        imbalance_ratio: Ratio of larger side to smaller side
        accumulation_phase: Boolean - true if accumulation behavior detected
    """
    bid_volume: float
    ask_volume: float
    bid_ask_ratio: float
    bid_ask_delta: float
    buying_pressure: float
    selling_pressure: float
    net_flow: float
    imbalance_ratio: float
    accumulation_phase: bool


@dataclass
class TickFlowAnalysis:
    """
    Tick-by-tick order flow analysis.
    
    Attributes:
        buy_ticks: Number of buying ticks
        sell_ticks: Number of selling ticks
        buy_volume: Total volume of buying ticks
        sell_volume: Total volume of selling ticks
        tick_imbalance_ratio: buy_ticks / sell_ticks
        volume_imbalance_ratio: buy_volume / sell_volume
        cumulative_delta: Running sum of (buy_vol - sell_vol)
        flow_quality: Quality/conviction of flow (0-1)
    """
    buy_ticks: int
    sell_ticks: int
    buy_volume: float
    sell_volume: float
    tick_imbalance_ratio: float
    volume_imbalance_ratio: float
    cumulative_delta: np.ndarray
    flow_quality: float


class OrderFlow:
    """
    Order Flow Analysis Engine for Market Microstructure.
    
    Analyzes:
    - Bid/Ask volume imbalance
    - Cumulative delta (buy volume - sell volume)
    - Order flow divergence (OFD)
    - Large block trades (institutional activity)
    - Absorption (how market absorbs aggressive orders)
    - Tick direction and volume profile
    
    Mathematical approach:
    1. Classify ticks as buy/sell based on price movement
    2. Accumulate volume by side
    3. Calculate imbalance metrics
    4. Detect anomalies and institutional footprints
    """
    
    def __init__(
        self,
        lookback_period: int = 20,
        imbalance_threshold: float = 1.25,
        block_trade_size_multiplier: float = 10.0
    ):
        """
        Initialize Order Flow Analysis Engine.
        
        Args:
            lookback_period: Number of bars for rolling calculations (20 = 20-bar MA)
            imbalance_threshold: Ratio threshold for significant imbalance (1.25 = 25% skew)
            block_trade_size_multiplier: Volume threshold for block trade (10x avg volume)
        """
        self.lookback_period = lookback_period
        self.imbalance_threshold = imbalance_threshold
        self.block_trade_size_multiplier = block_trade_size_multiplier
        logger.info(f"✓ OrderFlow initialized (lookback={lookback_period})")
    
    # ========================================================================
    # TICK CLASSIFICATION
    # ========================================================================
    
    def classify_ticks(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify each bar's volume as buy or sell based on price direction.
        
        Rules:
        - If Close > Open: Volume classified as "buy" (uptick)
        - If Close < Open: Volume classified as "sell" (downtick)
        - If Close = Open: Volume split 50/50 (equal ticks)
        
        Args:
            df: OHLCV DataFrame
            price_column: Column name for price
            volume_column: Column name for volume
            
        Returns:
            Tuple of (buy_volumes, sell_volumes) arrays
        """
        df = df.copy()
        
        buy_volume = np.zeros(len(df))
        sell_volume = np.zeros(len(df))
        
        for i, row in df.iterrows():
            volume = row[volume_column]
            close = row['Close']
            open_price = row['Open']
            
            if close > open_price:
                # Uptick = buying
                buy_volume[i] = volume
                sell_volume[i] = 0
            elif close < open_price:
                # Downtick = selling
                buy_volume[i] = 0
                sell_volume[i] = volume
            else:
                # Equal ticks = split
                buy_volume[i] = volume / 2
                sell_volume[i] = volume / 2
        
        return buy_volume, sell_volume
    
    # ========================================================================
    # ORDER FLOW METRICS
    # ========================================================================
    
    def calculate_order_flow_metrics(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume',
        lookback: Optional[int] = None
    ) -> OrderFlowMetrics:
        """
        Calculate comprehensive order flow metrics.
        
        Args:
            df: OHLCV DataFrame
            price_column: Column name for price
            volume_column: Column name for volume
            lookback: Number of bars to analyze (default: entire df)
            
        Returns:
            OrderFlowMetrics object
        """
        if lookback is None:
            lookback = len(df)
        
        df_slice = df.tail(lookback).copy()
        buy_vol, sell_vol = self.classify_ticks(df_slice, price_column, volume_column)
        
        total_buy = np.sum(buy_vol)
        total_sell = np.sum(sell_vol)
        total_volume = total_buy + total_sell
        
        # Avoid division by zero
        if total_sell == 0:
            bid_ask_ratio = float('inf') if total_buy > 0 else 1.0
        else:
            bid_ask_ratio = total_buy / total_sell
        
        bid_ask_delta = total_buy - total_sell
        
        # Normalized pressure (0-100)
        if total_volume > 0:
            buying_pressure = (total_buy / total_volume) * 100
            selling_pressure = (total_sell / total_volume) * 100
        else:
            buying_pressure = 50.0
            selling_pressure = 50.0
        
        # Net flow (-1 to +1)
        if total_volume > 0:
            net_flow = (total_buy - total_sell) / total_volume
        else:
            net_flow = 0.0
        
        # Imbalance detection
        if bid_ask_ratio > 1:
            imbalance_ratio = bid_ask_ratio
        elif bid_ask_ratio < 1:
            imbalance_ratio = 1 / bid_ask_ratio
        else:
            imbalance_ratio = 1.0
        
        # Accumulation phase: sustained imbalance
        accumulation_phase = imbalance_ratio > self.imbalance_threshold
        
        return OrderFlowMetrics(
            bid_volume=total_buy,
            ask_volume=total_sell,
            bid_ask_ratio=bid_ask_ratio,
            bid_ask_delta=bid_ask_delta,
            buying_pressure=buying_pressure,
            selling_pressure=selling_pressure,
            net_flow=net_flow,
            imbalance_ratio=imbalance_ratio,
            accumulation_phase=accumulation_phase
        )
    
    # ========================================================================
    # CUMULATIVE DELTA
    # ========================================================================
    
    def calculate_cumulative_delta(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume'
    ) -> np.ndarray:
        """
        Calculate cumulative delta (running sum of buy volume - sell volume).
        
        Cumulative delta shows net buying/selling pressure over time.
        - Rising delta = sustained buying
        - Falling delta = sustained selling
        - Divergence between price and delta = potential reversal
        
        Args:
            df: OHLCV DataFrame
            price_column: Column name for price
            volume_column: Column name for volume
            
        Returns:
            Array of cumulative delta values
        """
        buy_vol, sell_vol = self.classify_ticks(df, price_column, volume_column)
        delta = buy_vol - sell_vol
        cumulative_delta = np.cumsum(delta)
        
        return cumulative_delta
    
    # ========================================================================
    # TICK FLOW ANALYSIS
    # ========================================================================
    
    def analyze_tick_flow(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume',
        lookback: Optional[int] = None
    ) -> TickFlowAnalysis:
        """
        Analyze tick-by-tick flow patterns.
        
        Args:
            df: OHLCV DataFrame
            price_column: Column name for price
            volume_column: Column name for volume
            lookback: Number of bars to analyze
            
        Returns:
            TickFlowAnalysis object
        """
        if lookback is None:
            lookback = len(df)
        
        df_slice = df.tail(lookback).copy()
        buy_vol, sell_vol = self.classify_ticks(df_slice, price_column, volume_column)
        
        # Count ticks
        buy_ticks = np.count_nonzero(buy_vol)
        sell_ticks = np.count_nonzero(sell_vol)
        
        # Volume totals
        buy_volume = np.sum(buy_vol)
        sell_volume = np.sum(sell_vol)
        
        # Ratios
        if sell_ticks > 0:
            tick_imbalance_ratio = buy_ticks / sell_ticks
        else:
            tick_imbalance_ratio = float('inf') if buy_ticks > 0 else 1.0
        
        if sell_volume > 0:
            volume_imbalance_ratio = buy_volume / sell_volume
        else:
            volume_imbalance_ratio = float('inf') if buy_volume > 0 else 1.0
        
        # Cumulative delta
        cumulative_delta = self.calculate_cumulative_delta(df_slice, price_column, volume_column)
        
        # Flow quality (1.0 = extreme imbalance, 0.5 = balanced)
        avg_imbalance = (abs(tick_imbalance_ratio - 1.0) + abs(volume_imbalance_ratio - 1.0)) / 2
        flow_quality = min(1.0, avg_imbalance / 2.0)
        
        return TickFlowAnalysis(
            buy_ticks=buy_ticks,
            sell_ticks=sell_ticks,
            buy_volume=buy_volume,
            sell_volume=sell_volume,
            tick_imbalance_ratio=tick_imbalance_ratio,
            volume_imbalance_ratio=volume_imbalance_ratio,
            cumulative_delta=cumulative_delta,
            flow_quality=flow_quality
        )
    
    # ========================================================================
    # LARGE BLOCK DETECTION
    # ========================================================================
    
    def detect_block_trades(
        self,
        df: pd.DataFrame,
        volume_column: str = 'Volume',
        multiplier: Optional[float] = None
    ) -> List[Dict]:
        """
        Detect large institutional block trades.
        
        A block trade is defined as volume > (average_volume * multiplier).
        These indicate institutional involvement and often precede price moves.
        
        Args:
            df: OHLCV DataFrame
            volume_column: Column name for volume
            multiplier: Volume multiplier threshold (default: self.block_trade_size_multiplier)
            
        Returns:
            List of detected block trades with metadata
        """
        if multiplier is None:
            multiplier = self.block_trade_size_multiplier
        
        df = df.copy()
        avg_volume = df[volume_column].rolling(self.lookback_period).mean()
        threshold = avg_volume * multiplier
        
        block_trades = []
        
        for i, row in df.iterrows():
            volume = row[volume_column]
            avg_vol = avg_volume.iloc[i] if pd.notna(avg_volume.iloc[i]) else df[volume_column].mean()
            threshold_vol = avg_vol * multiplier
            
            if volume > threshold_vol:
                direction = 'buy' if row['Close'] > row['Open'] else 'sell'
                block_trades.append({
                    'bar_index': i,
                    'timestamp': df.index[i] if hasattr(df.index[i], 'timestamp') else i,
                    'volume': volume,
                    'average_volume': avg_vol,
                    'volume_multiple': volume / avg_vol,
                    'price': row['Close'],
                    'direction': direction,
                    'high': row['High'],
                    'low': row['Low']
                })
        
        logger.info(f"✓ Detected {len(block_trades)} block trades")
        return block_trades
    
    # ========================================================================
    # ABSORPTION ANALYSIS
    # ========================================================================
    
    def analyze_absorption(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume',
        lookback: Optional[int] = None
    ) -> Dict:
        """
        Analyze how market absorbs aggressive order flow.
        
        Absorption occurs when volume is high but price movement is minimal,
        indicating accumulation/distribution by large players.
        
        Args:
            df: OHLCV DataFrame
            price_column: Column name for price
            volume_column: Column name for volume
            lookback: Number of bars to analyze
            
        Returns:
            Dict with absorption metrics
        """
        if lookback is None:
            lookback = self.lookback_period
        
        df_slice = df.tail(lookback).copy()
        
        # Calculate price range (volatility)
        df_slice['range'] = df_slice['High'] - df_slice['Low']
        df_slice['range_pct'] = (df_slice['range'] / df_slice['Close']) * 100
        
        # Calculate volume metrics
        avg_volume = df_slice[volume_column].mean()
        avg_range = df_slice['range'].mean()
        avg_range_pct = df_slice['range_pct'].mean()
        
        # Absorption = High volume + Low range (price not moving much despite volume)
        # This suggests professional accumulation/distribution
        
        latest = df_slice.iloc[-1]
        volume_z_score = (latest[volume_column] - avg_volume) / (df_slice[volume_column].std() + 1e-9)
        range_z_score = (latest['range'] - avg_range) / (df_slice['range'].std() + 1e-9)
        
        # Absorption score: High volume (positive z) with low range (negative z)
        absorption_score = max(0, volume_z_score - range_z_score) / 2.0
        
        # Probability of absorption (0-1)
        absorption_probability = min(1.0, absorption_score / 2.0)
        
        return {
            'absorption_probability': absorption_probability,
            'absorption_score': absorption_score,
            'latest_volume': latest[volume_column],
            'latest_range': latest['range'],
            'average_volume': avg_volume,
            'average_range': avg_range,
            'average_range_pct': avg_range_pct,
            'volume_z_score': volume_z_score,
            'range_z_score': range_z_score
        }
    
    # ========================================================================
    # ORDER FLOW DIVERGENCE (OFD)
    # ========================================================================
    
    def detect_order_flow_divergence(
        self,
        df: pd.DataFrame,
        price_column: str = 'Close',
        volume_column: str = 'Volume',
        lookback: Optional[int] = None
    ) -> Optional[Dict]:
        """
        Detect Order Flow Divergence (OFD).
        
        OFD occurs when:
        - Price makes new high but cumulative delta makes new low (bearish OFD)
        - Price makes new low but cumulative delta makes new high (bullish OFD)
        
        This indicates weakening momentum and potential reversal.
        
        Args:
            df: OHLCV DataFrame
            price_column: Column name for price
            volume_column: Column name for volume
            lookback: Number of bars to analyze
            
        Returns:
            Dict with OFD details if detected, None otherwise
        """
        if lookback is None:
            lookback = self.lookback_period
        
        df_slice = df.tail(lookback).copy()
        
        # Calculate cumulative delta
        cumulative_delta = self.calculate_cumulative_delta(df_slice, price_column, volume_column)
        prices = df_slice[price_column].values
        
        # Find recent extremes
        price_max = np.max(prices[-5:]) if len(prices) >= 5 else np.max(prices)
        price_min = np.min(prices[-5:]) if len(prices) >= 5 else np.min(prices)
        
        delta_max = np.max(cumulative_delta[-5:]) if len(cumulative_delta) >= 5 else np.max(cumulative_delta)
        delta_min = np.min(cumulative_delta[-5:]) if len(cumulative_delta) >= 5 else np.min(cumulative_delta)
        
        current_price = prices[-1]
        current_delta = cumulative_delta[-1]
        
        # Bearish OFD: New price high with old delta high
        if current_price > price_max and current_delta < delta_max:
            return {
                'type': 'bearish_ofd',
                'current_price': current_price,
                'price_extreme': price_max,
                'current_delta': current_delta,
                'delta_extreme': delta_max,
                'confidence': min(1.0, abs(current_delta - delta_max) / (abs(delta_max) + 1e-9))
            }
        
        # Bullish OFD: New price low with old delta low
        if current_price < price_min and current_delta > delta_min:
            return {
                'type': 'bullish_ofd',
                'current_price': current_price,
                'price_extreme': price_min,
                'current_delta': current_delta,
                'delta_extreme': delta_min,
                'confidence': min(1.0, abs(current_delta - delta_min) / (abs(delta_min) + 1e-9))
            }
        
        return None

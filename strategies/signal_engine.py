# ============================================================================
# SIGNAL_ENGINE.PY - Master Signal Generator & Confluence Validator
# Requires 3+ independent confluences before signal validation
# ============================================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
import pandas as pd

from intelligence.volume_profile import VolumeProfile
from intelligence.structure_engine import StructureEngine
from microstructure.order_flow import OrderFlow
from aiml.ensemble_voter import EnsembleVoter, EnsembleVote
from aiml.signal_decay import SignalDecay, SignalType, SignalDecayTracker

logger = logging.getLogger(__name__)


class ConfluenceType(Enum):
    """Independent confluence sources for signal validation."""
    AI_ENSEMBLE = "ai_ensemble"        # ML model voting
    VOLUME_PROFILE = "volume_profile"  # POC, VAH, VAL levels
    STRUCTURE = "structure"            # Higher Highs/Lows, BoS, CHoCH
    ORDER_FLOW = "order_flow"          # Bid/ask imbalance, OFD
    MOMENTUM = "momentum"              # Price acceleration, RSI
    VOLATILITY = "volatility"          # Band squeeze, expansion
    REGIME = "regime"                  # Trending vs ranging


@dataclass
class Confluence:
    """Single confluence voting signal."""
    source: ConfluenceType
    direction: int  # 1 = bullish, 0 = bearish
    strength: float  # 0-1 confidence
    rationale: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class SignalVote:
    """Master signal generated from multiple confluences."""
    signal_id: str
    direction: int  # 1 = long, 0 = short
    confluences: List[Confluence]
    confluence_count: int
    consensus_strength: float  # Average strength across confluences
    recommendation: bool  # Should signal be executed?
    entry_price: float
    stop_loss: float
    take_profit: float
    rationale: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class SignalEngine:
    """
    Master Signal Generator with Confluence Validation.
    
    Requires at least 3 independent confluences from different sources
    before approving a signal for execution. Prevents false signals and
    ensures institutional-grade multi-factor confirmation.
    
    Confluence Sources:
    1. AI Ensemble: XGBoost + RandomForest + GradientBoost voting
    2. Volume Profile: POC/VAH/VAL price levels
    3. Market Structure: Higher Highs/Lows, Break of Structure
    4. Order Flow: Bid/ask imbalance, OFD divergence
    5. Momentum: RSI, MACD confluence
    6. Volatility: Band squeeze, expansion
    7. Regime: Trending confirmation
    
    Minimum Confluences: 3 (hard requirement)
    Target Confluences: 4-5 (institutional standard)
    """
    
    def __init__(self, min_confluences: int = 3, target_confluences: int = 4):
        """
        Initialize signal engine.
        
        Args:
            min_confluences: Minimum confluences required (default 3)
            target_confluences: Target for high-quality signals (default 4)
        """
        self.min_confluences = min_confluences
        self.target_confluences = target_confluences
        
        # Hard safety gate thresholds [REC #03, #05, #01, #07]
        self.MIN_VOLUME_THRESHOLD = 1000000.0  # Min avg volume last 20 bars
        self.MIN_ADX_TRENDING = 22.0            # ADX required if market trending
        
        # Initialize sub-engines
        self.ensemble_voter = EnsembleVoter()
        self.volume_profile = VolumeProfile()
        self.structure_engine = StructureEngine()
        self.order_flow = OrderFlow()
        self.signal_tracker = SignalDecayTracker()
        
        self.signal_counter = 0
        logger.info(
            f"✓ SignalEngine initialized [STRICT QUANT STANDARD]: "
            f"min {min_confluences}, target {target_confluences} confluences | "
            f"4 hard safety gates enforced"
        )
    
    
    # ========================================================================
    # HARD SAFETY GATES [REC #03, #05, #01, #07]
    # ========================================================================
    
    def _check_minimum_volume(self, bars: pd.DataFrame, symbol: str = None) -> bool:
        """
        [REC #03] Minimum Volume Filter - Hard Gate.
        Reject if avg volume last 20 bars < MIN_VOLUME_THRESHOLD.
        
        Args:
            bars: OHLCV DataFrame
            symbol: Symbol for logging
            
        Returns:
            True if volume passes, False if rejected
        """
        try:
            if 'volume' not in bars.columns or len(bars) < 20:
                logger.warning(f"⚠️  Volume data insufficient")
                return False
            
            avg_volume = bars['volume'].tail(20).mean()
            
            if avg_volume < self.MIN_VOLUME_THRESHOLD:
                logger.warning(
                    f"❌ [GATE #03] Volume rejected: {symbol if symbol else 'UNKNOWN'} | "
                    f"Avg volume {avg_volume:,.0f} < threshold {self.MIN_VOLUME_THRESHOLD:,.0f}"
                )
                return False
            
            logger.debug(f"✓ [GATE #03] Volume passed: {avg_volume:,.0f}")
            return True
        except Exception as e:
            logger.error(f"❌ Volume check failed: {e}")
            return False
    
    def _check_session_filter(self, symbol: str, current_time: datetime = None) -> bool:
        """
        [REC #05] Session Filter - Hard Gate.
        Reject Forex trades outside London/NY overlap (13:00-17:00 UTC).
        
        Args:
            symbol: Trading pair (e.g., EUR/USD)
            current_time: Current time (defaults to now)
            
        Returns:
            True if in valid session, False if rejected
        """
        try:
            # Only apply to Forex (contains / like EUR/USD)
            if "/" not in str(symbol):
                return True  # Not Forex, skip filter
            
            current_time = current_time or datetime.utcnow()
            hour = current_time.hour
            minute = current_time.minute
            
            # London/NY overlap: 13:00-17:00 UTC
            # London: 13:00-17:00 UTC (afternoon)
            # NY: ~08:00-12:00 UTC (morning), overlaps 13:00-17:00 with London afternoon
            start_hour = 13
            end_hour = 17
            
            in_overlap = (start_hour <= hour < end_hour)
            
            if not in_overlap:
                logger.warning(
                    f"❌ [GATE #05] Session rejected: {symbol} | "
                    f"Current UTC {hour:02d}:{minute:02d} outside "
                    f"London/NY overlap [13:00-17:00]"
                )
                return False
            
            logger.debug(f"✓ [GATE #05] Session passed: {symbol} in overlap window")
            return True
        except Exception as e:
            logger.error(f"❌ Session check failed: {e}")
            return False
    
    def _check_multi_timeframe_context(self, bars_15m: pd.DataFrame, bars_1h: pd.DataFrame = None, bars_4h: pd.DataFrame = None) -> bool:
        """
        [REC #01] Multi-Timeframe Context - Hard Gate.
        Require macro trend (1H/4H) to match entry trend (15m).
        
        Args:
            bars_15m: 15-minute OHLCV data
            bars_1h: 1-hour OHLCV data (optional, falls back to deriving from 15m)
            bars_4h: 4-hour OHLCV data (optional)
            
        Returns:
            True if trends align, False if rejected
        """
        try:
            # Determine 15m trend
            if 'sma_20_slope' not in bars_15m.columns:
                logger.warning(f"⚠️  15m SMA slope missing, deriving...")
                bars_15m['sma_20'] = bars_15m['close'].rolling(20).mean()
                bars_15m['sma_20_slope'] = bars_15m['sma_20'].diff()
            
            trend_15m = bars_15m['sma_20_slope'].iloc[-1]
            entry_trend_bullish = trend_15m > 0.0001
            entry_trend_bearish = trend_15m < -0.0001
            
            if not (entry_trend_bullish or entry_trend_bearish):
                logger.debug("ℹ️  [GATE #01] 15m trend neutral, no macro confirmation required")
                return True
            
            # Check 1H macro trend
            macro_aligned = False
            if bars_1h is not None:
                if 'sma_20_slope' not in bars_1h.columns:
                    bars_1h['sma_20'] = bars_1h['close'].rolling(20).mean()
                    bars_1h['sma_20_slope'] = bars_1h['sma_20'].diff()
                
                trend_1h = bars_1h['sma_20_slope'].iloc[-1]
                if (entry_trend_bullish and trend_1h > 0.0001) or (entry_trend_bearish and trend_1h < -0.0001):
                    macro_aligned = True
                    logger.debug(f"✓ 1H trend matches entry: {'BULL' if entry_trend_bullish else 'BEAR'}")
            
            # Check 4H macro trend if available
            if bars_4h is not None and not macro_aligned:
                if 'sma_20_slope' not in bars_4h.columns:
                    bars_4h['sma_20'] = bars_4h['close'].rolling(20).mean()
                    bars_4h['sma_20_slope'] = bars_4h['sma_20'].diff()
                
                trend_4h = bars_4h['sma_20_slope'].iloc[-1]
                if (entry_trend_bullish and trend_4h > 0.0001) or (entry_trend_bearish and trend_4h < -0.0001):
                    macro_aligned = True
                    logger.debug(f"✓ 4H trend matches entry: {'BULL' if entry_trend_bullish else 'BEAR'}")
            
            if not macro_aligned:
                logger.warning(
                    f"❌ [GATE #01] Multi-timeframe rejected: "
                    f"15m trend ({'BULL' if entry_trend_bullish else 'BEAR'}) "
                    f"does not align with macro (1H/4H)"
                )
                return False
            
            logger.debug(f"✓ [GATE #01] Multi-timeframe aligned")
            return True
        except Exception as e:
            logger.error(f"❌ Multi-timeframe check failed: {e}")
            return False
    
    def _check_adx_filter(self, bars: pd.DataFrame, regime: str = "trending") -> bool:
        """
        [REC #07] ADX Filter - Hard Gate.
        If market regime is trending, require ADX > 22.
        
        Args:
            bars: OHLCV DataFrame with ADX
            regime: Market regime ('trending', 'ranging', etc)
            
        Returns:
            True if passes, False if rejected
        """
        try:
            if regime != "trending":
                logger.debug(f"ℹ️  [GATE #07] Market regime '{regime}', ADX check optional")
                return True
            
            if 'adx' not in bars.columns or len(bars) < 14:
                logger.warning(f"⚠️  ADX data insufficient")
                return False
            
            adx = bars['adx'].iloc[-1]
            
            if adx < self.MIN_ADX_TRENDING:
                logger.warning(
                    f"❌ [GATE #07] ADX rejected (trending regime): "
                    f"ADX {adx:.1f} < required {self.MIN_ADX_TRENDING}"
                )
                return False
            
            logger.debug(f"✓ [GATE #07] ADX passed (trending): {adx:.1f}")
            return True
        except Exception as e:
            logger.error(f"❌ ADX check failed: {e}")
            return False
    
    # ========================================================================
    # CONFLUENCE VALIDATORS
    # ========================================================================
    
    def check_ai_ensemble(self, features: np.ndarray) -> Optional[Confluence]:
        """
        Check AI Ensemble confluence.
        
        Returns confluence if confidence > 60%.
        """
        try:
            vote: EnsembleVote = self.ensemble_voter.vote(features)
            
            if vote.confidence < 0.60:
                return None
            
            direction_str = "bullish" if vote.direction == 1 else "bearish"
            return Confluence(
                source=ConfluenceType.AI_ENSEMBLE,
                direction=vote.direction,
                strength=vote.confidence,
                rationale=(
                    f"AI ensemble {direction_str}: "
                    f"XGBoost {vote.xgboost_prob:.1%}, "
                    f"RF {vote.rf_prob:.1%}, "
                    f"GB {vote.gb_prob:.1%} | "
                    f"Anomaly score: {vote.anomaly_score:.2f}"
                )
            )
        except Exception as e:
            logger.warning(f"⚠️  AI ensemble check failed: {e}")
            return None
    
    def check_volume_profile(self, bars: pd.DataFrame) -> Optional[Confluence]:
        """
        Check Volume Profile confluence.
        
        Confluence if price is in value area (between VAL and VAH).
        """
        try:
            vp_result = self.volume_profile.calculate(bars)
            
            if vp_result is None:
                return None
            
            current_price = bars['Close'].iloc[-1]
            vp_val = vp_result.val
            vp_vah = vp_result.vah
            
            # Check if price is in established value area
            in_value_area = vp_val <= current_price <= vp_vah
            
            if not in_value_area:
                return None
            
            # Determine direction from price position
            poc = vp_result.poc
            direction = 1 if current_price > poc else 0
            strength = 0.7  # Volume profile confluence is moderately strong
            
            return Confluence(
                source=ConfluenceType.VOLUME_PROFILE,
                direction=direction,
                strength=strength,
                rationale=(
                    f"Volume Profile confirms: "
                    f"Price {current_price:.2f} in value area "
                    f"[{vp_val:.2f}, {vp_vah:.2f}], POC {poc:.2f}"
                )
            )
        except Exception as e:
            logger.warning(f"⚠️  Volume profile check failed: {e}")
            return None
    
    def check_structure(self, bars: pd.DataFrame) -> Optional[Confluence]:
        """
        Check Market Structure confluence.
        
        Confluence if higher high/low or break of structure detected.
        """
        try:
            structure_result = self.structure_engine.analyze(bars)
            
            if not structure_result:
                return None
            
            # Check for structure signals
            has_hh = structure_result.get('higher_high', False)
            has_hl = structure_result.get('higher_low', False)
            has_lh = structure_result.get('lower_high', False)
            has_ll = structure_result.get('lower_low', False)
            
            # Determine direction
            bullish_signals = int(has_hh) + int(has_hl)
            bearish_signals = int(has_lh) + int(has_ll)
            
            if bullish_signals == 0 and bearish_signals == 0:
                return None
            
            direction = 1 if bullish_signals > bearish_signals else 0
            strength = 0.75  # Structure signals are strong
            
            return Confluence(
                source=ConfluenceType.STRUCTURE,
                direction=direction,
                strength=strength,
                rationale=(
                    f"Structure: HH={has_hh}, HL={has_hl}, "
                    f"LH={has_lh}, LL={has_ll}, "
                    f"Trend: {structure_result.get('trend', 'UNKNOWN')}"
                )
            )
        except Exception as e:
            logger.warning(f"⚠️  Structure check failed: {e}")
            return None
    
    def check_order_flow(self, bars: pd.DataFrame) -> Optional[Confluence]:
        """
        Check Order Flow confluence.
        
        Confluence if bid/ask imbalance or OFD detected.
        """
        try:
            flow_result = self.order_flow.analyze(bars)
            
            if not flow_result:
                return None
            
            # Check for order flow signals
            net_flow = flow_result.get('net_flow', 0)  # -1 to +1
            ofd = flow_result.get('order_flow_divergence', False)
            
            # Only trigger if strong imbalance
            if abs(net_flow) < 0.4:
                return None
            
            direction = 1 if net_flow > 0 else 0
            strength = min(0.85, abs(net_flow))
            
            ofd_str = " + OFD" if ofd else ""
            return Confluence(
                source=ConfluenceType.ORDER_FLOW,
                direction=direction,
                strength=strength,
                rationale=(
                    f"Order Flow: net flow {net_flow:.2f}{ofd_str}, "
                    f"bid/ask imbalance detected"
                )
            )
        except Exception as e:
            logger.warning(f"⚠️  Order flow check failed: {e}")
            return None
    
    def check_momentum(self, bars: pd.DataFrame) -> Optional[Confluence]:
        """
        Check Momentum confluence.
        
        RSI overbought/oversold or MACD crossover.
        """
        try:
            if 'rsi' not in bars.columns or 'macd' not in bars.columns:
                return None
            
            rsi = bars['rsi'].iloc[-1]
            macd = bars['macd'].iloc[-1]
            macd_signal = bars['macd_signal'].iloc[-1]
            
            # RSI check: overbought (>70) or oversold (<30)
            rsi_bullish = rsi < 30  # Oversold = potential bounce
            rsi_bearish = rsi > 70  # Overbought = potential reversal
            
            # MACD check: crossover
            macd_bullish = macd > macd_signal
            
            if not (rsi_bullish or rsi_bearish or macd_bullish):
                return None
            
            direction = 1 if (rsi_bullish or macd_bullish) else 0
            strength = 0.65
            
            return Confluence(
                source=ConfluenceType.MOMENTUM,
                direction=direction,
                strength=strength,
                rationale=(
                    f"Momentum: RSI {rsi:.1f}, MACD {macd:.4f} "
                    f"vs Signal {macd_signal:.4f}"
                )
            )
        except Exception as e:
            logger.warning(f"⚠️  Momentum check failed: {e}")
            return None
    
    def check_volatility(self, bars: pd.DataFrame) -> Optional[Confluence]:
        """
        Check Volatility confluence.
        
        Band squeeze (low volatility) suggests expansion coming.
        """
        try:
            if 'volatility' not in bars.columns or 'bb_position' not in bars.columns:
                return None
            
            volatility = bars['volatility'].iloc[-1]
            vol_ma = bars['volatility'].rolling(20).mean().iloc[-1]
            bb_position = bars['bb_position'].iloc[-1]
            
            # Band squeeze: current volatility < 70% of MA
            is_squeeze = volatility < (vol_ma * 0.7)
            
            if not is_squeeze:
                return None
            
            # Direction based on price position within bands
            direction = 1 if bb_position > 0.5 else 0
            strength = 0.60
            
            return Confluence(
                source=ConfluenceType.VOLATILITY,
                direction=direction,
                strength=strength,
                rationale=(
                    f"Volatility squeeze detected: "
                    f"{volatility:.4f} vs MA {vol_ma:.4f}, "
                    f"BB position {bb_position:.2f}"
                )
            )
        except Exception as e:
            logger.warning(f"⚠️  Volatility check failed: {e}")
            return None
    
    def check_regime(self, bars: pd.DataFrame) -> Optional[Confluence]:
        """
        Check Regime confluence.
        
        Trend or range confirmation.
        """
        try:
            if 'sma_20_slope' not in bars.columns or 'rsi' not in bars.columns:
                return None
            
            sma_slope = bars['sma_20_slope'].iloc[-1]
            rsi = bars['rsi'].iloc[-1]
            
            # Trending regime: MA slope > 0.001 and RSI away from center
            is_trending_up = sma_slope > 0.001 and rsi > 55
            is_trending_down = sma_slope < -0.001 and rsi < 45
            
            if not (is_trending_up or is_trending_down):
                return None
            
            direction = 1 if is_trending_up else 0
            strength = 0.70
            
            return Confluence(
                source=ConfluenceType.REGIME,
                direction=direction,
                strength=strength,
                rationale=(
                    f"Regime trending {'up' if direction == 1 else 'down'}: "
                    f"SMA slope {sma_slope:.5f}, RSI {rsi:.1f}"
                )
            )
        except Exception as e:
            logger.warning(f"⚠️  Regime check failed: {e}")
            return None
    
    # ========================================================================
    # SIGNAL GENERATION
    # ========================================================================
    
    def generate_signal(
        self,
        bars: pd.DataFrame,
        features: np.ndarray,
        entry_price: float,
        stop_loss: float = None,
        take_profit: float = None,
        symbol: str = None,
        bars_1h: pd.DataFrame = None,
        bars_4h: pd.DataFrame = None,
        regime: str = "trending"
    ) -> Optional[SignalVote]:
        """
        Generate a signal from confluence validators.
        
        [STRICT QUANT STANDARD] First applies 4 hard safety gates:
        1. Minimum Volume Filter (REC #03)
        2. Session Filter (REC #05) 
        3. Multi-Timeframe Context (REC #01)
        4. ADX Filter (REC #07)
        
        If ANY gate fails, returns None immediately (trade rejected).
        
        Then runs all 7 confluence checks and aggregates votes.
        Only approves if >= min_confluences agree.
        
        Args:
            bars: OHLCV data (must contain engineered features)
            features: Pre-engineered feature vector
            entry_price: Proposed entry price
            stop_loss: Optional SL price
            take_profit: Optional TP price
            symbol: Trading symbol (for logging/session check)
            bars_1h: 1-hour OHLCV data for MTF check
            bars_4h: 4-hour OHLCV data for MTF check
            regime: Market regime for ADX check
            
        Returns:
            SignalVote if approved (>= min confluences), None otherwise
        """
        # ====================================================================
        # [HARD SAFETY GATES] - Return None if ANY fail
        # ====================================================================
        logger.info("\n" + "="*70)
        logger.info("[HARD SAFETY GATES] Applying 4 institutional safety gates...")
        logger.info("="*70)
        
        # GATE #03: Minimum Volume
        if not self._check_minimum_volume(bars, symbol):
            logger.error("🛑 SIGNAL REJECTED by Gate #03 (Volume)")
            return None
        
        # GATE #05: Session Filter
        if not self._check_session_filter(symbol):
            logger.error("🛑 SIGNAL REJECTED by Gate #05 (Session)")
            return None
        
        # GATE #01: Multi-Timeframe Context
        if not self._check_multi_timeframe_context(bars, bars_1h, bars_4h):
            logger.error("🛑 SIGNAL REJECTED by Gate #01 (Multi-TF)")
            return None
        
        # GATE #07: ADX Filter
        if not self._check_adx_filter(bars, regime):
            logger.error("🛑 SIGNAL REJECTED by Gate #07 (ADX)")
            return None
        
        logger.info("✓ All 4 hard safety gates PASSED\n")
        
        # ====================================================================
        # [CONFLUENCE CHECKS] - Run after gates pass
        # ====================================================================
        logger.info("🔍 Running confluence checks...")
        
        confluences = []
        
        # Run all 7 checks
        checks = [
            self.check_ai_ensemble(features),
            self.check_volume_profile(bars),
            self.check_structure(bars),
            self.check_order_flow(bars),
            self.check_momentum(bars),
            self.check_volatility(bars),
            self.check_regime(bars),
        ]
        
        # Filter valid confluences
        for confluence in checks:
            if confluence is not None:
                confluences.append(confluence)
        
        logger.info(f"  {len(confluences)}/{len(checks)} confluences passed")
        
        # Check minimum threshold
        if len(confluences) < self.min_confluences:
            logger.warning(
                f"❌ Signal rejected: only {len(confluences)} confluences "
                f"(need {self.min_confluences})"
            )
            return None
        
        # Final recommendation based on consensus
        bullish_count = sum(1 for c in confluences if c.direction == 1)
        bearish_count = sum(1 for c in confluences if c.direction == 0)
        direction = 1 if bullish_count > bearish_count else 0
        
        consensus_strength = np.mean([c.strength for c in confluences])
        
        # Self-critique: quality check
        quality_score = (len(confluences) - self.min_confluences) / 4.0  # 0-1
        is_high_quality = len(confluences) >= self.target_confluences
        
        self.signal_counter += 1
        signal_id = f"SIG_{self.signal_counter:06d}"
        
        # Create signal
        signal = SignalVote(
            signal_id=signal_id,
            direction=direction,
            confluences=confluences,
            confluence_count=len(confluences),
            consensus_strength=consensus_strength,
            recommendation=is_high_quality,
            entry_price=entry_price,
            stop_loss=stop_loss or (entry_price * 0.99),
            take_profit=take_profit or (entry_price * 1.02),
            rationale=[c.rationale for c in confluences],
            timestamp=datetime.now()
        )
        
        quality_str = "🟢 HIGH" if is_high_quality else "🟡 MEDIUM"
        logger.info(
            f"✅ Signal approved: {signal_id} | "
            f"{direction_str[direction]}, "
            f"{len(confluences)} confluences, "
            f"quality: {quality_str}"
        )
        
        return signal
    
    def approve_for_quantitative_layer(self, signal: SignalVote) -> bool:
        """
        Determine if signal should be passed to Quant layer for stress testing.
        
        Quant layer will perform additional checks (MC, Black-Scholes, etc).
        This layer only ensures multi-factor confluence.
        
        Args:
            signal: SignalVote object
            
        Returns:
            True if should pass to Quant, False if hold
        """
        if signal is None:
            return False
        
        if signal.confluence_count < self.min_confluences:
            return False
        
        # Track in decay tracker
        decay_signal = SignalDecay(
            signal_id=signal.signal_id,
            signal_type=SignalType.CONFLUENCE,
            direction=signal.direction,
            generated_at=signal.timestamp,
            entry_price=signal.entry_price,
            initial_confidence=signal.consensus_strength,
            ttl_seconds=3600
        )
        self.signal_tracker.add_signal(decay_signal)
        
        return True


# Helper mapping
direction_str = {1: "LONG", 0: "SHORT"}

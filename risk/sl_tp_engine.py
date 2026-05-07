# ============================================================================
# STOP LOSS / TAKE PROFIT ENGINE - L5 RISK MANAGEMENT
# Calculates mathematically optimal exit targets with 1:2+ R/R ratio
# ============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Optional, Dict, List, Tuple, Any, Callable
from dataclasses import dataclass
from enum import Enum
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class ExitStrategy(Enum):
    """Exit strategy enumeration."""
    ATR_BASED = "atr_based"  # Using Average True Range
    SUPPORT_RESISTANCE = "support_resistance"  # Key levels
    VOLATILITY_BASED = "volatility_based"  # Using implied/historical volatility
    FIXED_POINTS = "fixed_points"  # Fixed number of points
    CHANDELIER = "chandelier"  # Chandelier stop


@dataclass
class SLTPLevels:
    """Stop Loss / Take Profit level pair."""
    entry_price: float
    stop_loss: float
    take_profit: float
    strategy: ExitStrategy
    
    # Optional enrichment
    atr: Optional[float] = None
    support: Optional[float] = None
    resistance: Optional[float] = None
    
    @property
    def risk_amount(self) -> float:
        """Calculate risk in dollars (per unit)."""
        return abs(self.entry_price - self.stop_loss)
    
    @property
    def reward_amount(self) -> float:
        """Calculate reward in dollars (per unit)."""
        return abs(self.take_profit - self.entry_price)
    
    @property
    def risk_reward_ratio(self) -> float:
        """
        Calculate R/R ratio.
        
        Formula: Reward / Risk
        Example: If risk=$10 and reward=$20, ratio=2.0 (1:2)
        """
        if self.risk_amount == 0:
            return 0.0
        return self.reward_amount / self.risk_amount
    
    @property
    def is_valid(self) -> bool:
        """Check if levels form valid trade setup."""
        # SL and TP on opposite sides of entry
        if self.entry_price > self.stop_loss and self.entry_price < self.take_profit:
            return True  # Long valid
        if self.entry_price < self.stop_loss and self.entry_price > self.take_profit:
            return True  # Short valid
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "strategy": self.strategy.value,
            "risk_amount": self.risk_amount,
            "reward_amount": self.reward_amount,
            "risk_reward_ratio": self.risk_reward_ratio,
            "is_valid": self.is_valid,
            "atr": self.atr,
            "support": self.support,
            "resistance": self.resistance,
        }


# ============================================================================
# EXIT CALCULATORS (Abstract Base + Implementations)
# ============================================================================

class BaseExitCalculator(ABC):
    """Base class for exit level calculators."""
    
    def __init__(self, min_risk_reward_ratio: float = 1.5):
        """
        Initialize calculator.
        
        Args:
            min_risk_reward_ratio: Minimum required R/R ratio (e.g., 1.5 = 1:1.5)
        """
        self.min_rr_ratio = min_risk_reward_ratio
    
    @abstractmethod
    def calculate(
        self,
        entry_price: float,
        direction: str,
        **kwargs
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP levels.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            **kwargs: Additional parameters
        
        Returns:
            SLTPLevels or None if calculation fails
        """
        pass
    
    def _validate_ratio(self, sl: float, tp: float, entry: float) -> bool:
        """Validate that trade meets minimum R/R ratio."""
        risk = abs(entry - sl)
        reward = abs(tp - entry)
        
        if risk == 0:
            return False
        
        ratio = reward / risk
        return ratio >= self.min_rr_ratio


class ATRBasedCalculator(BaseExitCalculator):
    """
    ATR-based stop loss and take profit (used by Renaissance Tech).
    
    THEORY:
    - ATR (Average True Range) measures market volatility
    - High volatility: wider SL/TP
    - Low volatility: tighter SL/TP
    - Auto-adjusts to market conditions
    
    FORMULA (Long):
    - Stop Loss = Entry - (ATR_multiplier * ATR)
    - Take Profit = Entry + (2 * ATR_multiplier * ATR)  [for 1:2 R/R]
    
    FORMULA (Short):
    - Stop Loss = Entry + (ATR_multiplier * ATR)
    - Take Profit = Entry - (2 * ATR_multiplier * ATR)
    """
    
    def __init__(self, min_risk_reward_ratio: float = 1.5):
        super().__init__(min_risk_reward_ratio)
        self.name = "ATR-Based"
    
    def calculate(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP based on ATR.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            atr: Current ATR value
            atr_multiplier: Multiplier (default 2.0 = 2x ATR distance)
        
        Returns:
            SLTPLevels or None
        """
        risk_distance = atr_multiplier * atr
        
        if direction == "long":
            sl = entry_price - risk_distance
            tp = entry_price + (2 * risk_distance)  # 1:2 ratio
        elif direction == "short":
            sl = entry_price + risk_distance
            tp = entry_price - (2 * risk_distance)
        else:
            return None
        
        levels = SLTPLevels(
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            strategy=ExitStrategy.ATR_BASED,
            atr=atr,
        )
        
        # Validate
        if not self._validate_ratio(sl, tp, entry_price):
            logger.warning(f"ATR-based levels don't meet R/R ratio: {levels.risk_reward_ratio:.2f}")
            return None
        
        return levels


class VolatilityBasedCalculator(BaseExitCalculator):
    """
    Volatility-based stop loss (used by BlackRock).
    
    Adjusts stops based on implied volatility of the instrument.
    
    FORMULA:
    - Volatility Distance = Entry_Price * (Volatility % / 100) * Vol_Multiplier
    - Stop Loss = Entry ± Vol_Distance
    - Take Profit = Entry ± (2 * Vol_Distance)  [for 1:2]
    """
    
    def __init__(self, min_risk_reward_ratio: float = 1.5):
        super().__init__(min_risk_reward_ratio)
        self.name = "Volatility-Based"
    
    def calculate(
        self,
        entry_price: float,
        direction: str,
        volatility_pct: float,
        vol_multiplier: float = 0.5,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP based on volatility.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            volatility_pct: Historical/implied volatility as percentage
            vol_multiplier: Multiplier for volatility distance (default 0.5)
        
        Returns:
            SLTPLevels or None
        """
        vol_distance = entry_price * (volatility_pct / 100) * vol_multiplier
        
        if direction == "long":
            sl = entry_price - vol_distance
            tp = entry_price + (2 * vol_distance)
        elif direction == "short":
            sl = entry_price + vol_distance
            tp = entry_price - (2 * vol_distance)
        else:
            return None
        
        levels = SLTPLevels(
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            strategy=ExitStrategy.VOLATILITY_BASED,
        )
        
        if not self._validate_ratio(sl, tp, entry_price):
            logger.warning(f"Vol-based levels don't meet R/R ratio: {levels.risk_reward_ratio:.2f}")
            return None
        
        return levels


class SupportResistanceCalculator(BaseExitCalculator):
    """
    Support/Resistance based stops (used by Citadel).
    
    Places stops at key technical levels.
    
    STRATEGY:
    - Long: SL just below support, TP at resistance
    - Short: SL just above resistance, TP at support
    - Ensures stops are at meaningful levels
    """
    
    def __init__(self, min_risk_reward_ratio: float = 1.5):
        super().__init__(min_risk_reward_ratio)
        self.name = "Support/Resistance"
    
    def calculate(
        self,
        entry_price: float,
        direction: str,
        support: float,
        resistance: float,
        buffer_pct: float = 0.1,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP using support/resistance.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            support: Support level
            resistance: Resistance level
            buffer_pct: Buffer % below/above levels
        
        Returns:
            SLTPLevels or None
        """
        buffer_dist = entry_price * (buffer_pct / 100)
        
        if direction == "long":
            sl = support - buffer_dist
            tp = resistance + buffer_dist
        elif direction == "short":
            sl = resistance + buffer_dist
            tp = support - buffer_dist
        else:
            return None
        
        levels = SLTPLevels(
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            strategy=ExitStrategy.SUPPORT_RESISTANCE,
            support=support,
            resistance=resistance,
        )
        
        if not self._validate_ratio(sl, tp, entry_price):
            logger.warning(f"S/R levels don't meet R/R ratio: {levels.risk_reward_ratio:.2f}")
            # Adjust TP to meet R/R
            risk = abs(entry_price - sl)
            if direction == "long":
                tp = entry_price + (self.min_rr_ratio * risk)
            else:
                tp = entry_price - (self.min_rr_ratio * risk)
            
            levels.take_profit = tp
        
        return levels


class FixedPointsCalculator(BaseExitCalculator):
    """
    Fixed points/pips stop loss (simple, used for tests).
    
    Allows manual specification of risk/reward distances.
    """
    
    def __init__(self, min_risk_reward_ratio: float = 1.5):
        super().__init__(min_risk_reward_ratio)
        self.name = "Fixed Points"
    
    def calculate(
        self,
        entry_price: float,
        direction: str,
        risk_points: float,
        reward_points: Optional[float] = None,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP with fixed points.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            risk_points: Points to risk
            reward_points: Points for reward (auto-calculated if None)
        
        Returns:
            SLTPLevels
        """
        if reward_points is None:
            reward_points = risk_points * self.min_rr_ratio
        
        if direction == "long":
            sl = entry_price - risk_points
            tp = entry_price + reward_points
        elif direction == "short":
            sl = entry_price + risk_points
            tp = entry_price - reward_points
        else:
            return None
        
        levels = SLTPLevels(
            entry_price=entry_price,
            stop_loss=sl,
            take_profit=tp,
            strategy=ExitStrategy.FIXED_POINTS,
        )
        
        return levels


# ============================================================================
# MASTER SL/TP ENGINE
# ============================================================================

class SLTPEngine:
    """
    Stop Loss / Take Profit calculation engine.
    
    PURPOSE (from PDF):
    "[cite_start]Calculate mathematically optimal stop loss and take profit
    targets per trade, ensuring a minimum 1:2 R/R ratio[cite: 30, 47]."
    
    PROVIDES:
    - Multiple calculation methods (ATR, volatility, support/resistance, fixed)
    - Automatic R/R ratio validation
    - Adjustments to meet minimum thresholds
    - Integration with risk management
    
    USAGE:
    >>> engine = SLTPEngine(min_rr_ratio=1.5)
    >>> 
    >>> levels = engine.calculate_atr_based(
    ...     entry_price=150.0,
    ...     direction="long",
    ...     atr=2.5,
    ...     atr_multiplier=2.0
    ... )
    >>> print(f"SL: {levels.stop_loss}, TP: {levels.take_profit}")
    >>> print(f"R/R Ratio: {levels.risk_reward_ratio:.2f}:1")
    """
    
    def __init__(self, min_rr_ratio: float = 1.5):
        """
        Initialize SL/TP Engine.
        
        Args:
            min_rr_ratio: Minimum risk/reward ratio (e.g., 1.5 = trade only 1:1.5+)
        """
        self.min_rr_ratio = min_rr_ratio
        
        # Calculators
        self.atr_calc = ATRBasedCalculator(min_rr_ratio)
        self.vol_calc = VolatilityBasedCalculator(min_rr_ratio)
        self.sr_calc = SupportResistanceCalculator(min_rr_ratio)
        self.fixed_calc = FixedPointsCalculator(min_rr_ratio)
        
        logger.info(f"✓ SLTPEngine initialized")
        logger.info(f"  Minimum R/R Ratio: {min_rr_ratio}:1")
    
    # ========================================================================
    # CALCULATION METHODS
    # ========================================================================
    
    def calculate_atr_based(
        self,
        entry_price: float,
        direction: str,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP using ATR (Average True Range).
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            atr: Current ATR
            atr_multiplier: Multiplier (default 2.0)
        
        Returns:
            SLTPLevels or None
        """
        return self.atr_calc.calculate(entry_price, direction, atr, atr_multiplier)
    
    def calculate_volatility_based(
        self,
        entry_price: float,
        direction: str,
        volatility_pct: float,
        vol_multiplier: float = 0.5,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP using volatility.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            volatility_pct: Volatility as percentage
            vol_multiplier: Multiplier (default 0.5)
        
        Returns:
            SLTPLevels or None
        """
        return self.vol_calc.calculate(
            entry_price, direction, volatility_pct, vol_multiplier
        )
    
    def calculate_support_resistance(
        self,
        entry_price: float,
        direction: str,
        support: float,
        resistance: float,
        buffer_pct: float = 0.1,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP using key levels.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            support: Support level
            resistance: Resistance level
            buffer_pct: Buffer % (default 0.1%)
        
        Returns:
            SLTPLevels or None
        """
        return self.sr_calc.calculate(
            entry_price, direction, support, resistance, buffer_pct
        )
    
    def calculate_fixed_points(
        self,
        entry_price: float,
        direction: str,
        risk_points: float,
        reward_points: Optional[float] = None,
    ) -> Optional[SLTPLevels]:
        """
        Calculate SL/TP with fixed points/pips.
        
        Args:
            entry_price: Entry price
            direction: "long" or "short"
            risk_points: Risk points
            reward_points: Reward points (auto-calc if None)
        
        Returns:
            SLTPLevels
        """
        return self.fixed_calc.calculate(
            entry_price, direction, risk_points, reward_points
        )
    
    # ========================================================================
    # ADJUSTMENT & VALIDATION
    # ========================================================================
    
    def ensure_minimum_rr(self, levels: SLTPLevels) -> SLTPLevels:
        """
        Ensure levels meet minimum R/R ratio.
        
        If not, adjusts TP upward to meet requirement.
        
        Args:
            levels: SLTPLevels to adjust
        
        Returns:
            Adjusted SLTPLevels
        """
        if levels.risk_reward_ratio >= self.min_rr_ratio:
            return levels  # Already meets requirement
        
        # Adjust TP
        risk = abs(levels.entry_price - levels.stop_loss)
        required_reward = risk * self.min_rr_ratio
        
        if levels.entry_price < levels.take_profit:  # Long
            levels.take_profit = levels.entry_price + required_reward
        else:  # Short
            levels.take_profit = levels.entry_price - required_reward
        
        logger.info(f"Adjusted TP to meet minimum R/R: {levels.risk_reward_ratio:.2f}:1")
        
        return levels
    
    def scale_risk(
        self,
        levels: SLTPLevels,
        risk_pct: float,
        account_size: float,
    ) -> Dict[str, float]:
        """
        Calculate position size based on risk percentage.
        
        FORMULA:
        - Risk Amount = Account_Size * (Risk_Pct / 100)
        - Position_Size = Risk_Amount / (Entry - SL)
        
        Args:
            levels: SL/TP levels
            risk_pct: Risk as % of account (e.g., 1.0 = 1%)
            account_size: Total account size
        
        Returns:
            Dict with position_size, risk_amount, max_position_value
        """
        risk_amount = account_size * (risk_pct / 100)
        risk_distance = abs(levels.entry_price - levels.stop_loss)
        
        if risk_distance == 0:
            return {
                "position_size": 0,
                "risk_amount": 0,
                "max_position_value": 0,
            }
        
        position_size = risk_amount / risk_distance
        max_position_value = position_size * levels.entry_price
        expected_profit = position_size * (levels.take_profit - levels.entry_price)
        
        return {
            "position_size": position_size,
            "risk_amount": risk_amount,
            "expected_profit": expected_profit,
            "max_position_value": max_position_value,
            "risk_distance": risk_distance,
        }
    
    def partial_tp_levels(
        self,
        entry_price: float,
        stop_loss: float,
        quantity: float,
        direction: str,
    ) -> Dict[str, Any]:
        """
        PARTIAL TAKE-PROFIT (REC #12): Calculate 50/50 exit strategy.
        
        Exit structure:
        - 50% position at exactly 1:1 Risk/Reward ratio
        - 50% position at exactly 1:2 Risk/Reward ratio
        
        This allows early profit-taking on half while protecting upside on remainder.
        
        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            quantity: Total position quantity
            direction: "long" or "short"
        
        Returns:
            Dict with:
                - tp_partial: 1:1 R/R exit price for 50% of position
                - qty_partial: 50% of quantity
                - tp_full: 1:2 R/R exit price for remaining 50%
                - qty_full: Remaining 50% of quantity
                - total_expected_profit: Combined expected profit from both exits
        """
        risk_distance = abs(entry_price - stop_loss)
        
        if risk_distance == 0:
            logger.error("Cannot calculate partial TP: Stop loss equals entry price")
            return {}
        
        # Calculate both TP levels based on direction
        if direction == "long":
            tp_partial = entry_price + risk_distance  # 1:1 R/R
            tp_full = entry_price + (2 * risk_distance)  # 1:2 R/R
        elif direction == "short":
            tp_partial = entry_price - risk_distance  # 1:1 R/R
            tp_full = entry_price - (2 * risk_distance)  # 1:2 R/R
        else:
            logger.error(f"Invalid direction: {direction}")
            return {}
        
        # Split position 50/50
        qty_partial = quantity / 2
        qty_full = quantity - qty_partial  # Use subtraction to ensure exact sum
        
        # Calculate expected profit from each exit
        profit_partial = qty_partial * abs(tp_partial - entry_price)
        profit_full = qty_full * abs(tp_full - entry_price)
        total_expected_profit = profit_partial + profit_full
        
        # Verify valid setup
        if direction == "long":
            is_valid = stop_loss < entry_price < tp_partial < tp_full
        else:
            is_valid = tp_full < tp_partial < entry_price < stop_loss
        
        result = {
            "tp_partial": tp_partial,
            "qty_partial": qty_partial,
            "tp_full": tp_full,
            "qty_full": qty_full,
            "total_expected_profit": total_expected_profit,
            "profit_partial": profit_partial,
            "profit_full": profit_full,
            "is_valid": is_valid,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "risk_distance": risk_distance,
            "direction": direction,
        }
        
        if is_valid:
            logger.info(
                f"[REC #12 PARTIAL TP] {direction.upper()} position split: "
                f"50% (@1:1 = {tp_partial:.2f}) + 50% (@1:2 = {tp_full:.2f}) | "
                f"Expected profit: ${total_expected_profit:,.2f}"
            )
        else:
            logger.warning(f"Partial TP setup is INVALID: {result}")
        
        return result
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def print_levels(self, levels: SLTPLevels, account_size: float = 100000, risk_pct: float = 1.0) -> None:
        """Print detailed SL/TP analysis."""
        position_info = self.scale_risk(levels, risk_pct, account_size)
        
        print("\n" + "=" * 70)
        print("STOP LOSS / TAKE PROFIT ANALYSIS")
        print("=" * 70)
        
        direction = "LONG" if levels.entry_price < levels.take_profit else "SHORT"
        
        print(f"\nTRADE SETUP: {direction}")
        print(f"  Entry Price:        ${levels.entry_price:>15,.2f}")
        print(f"  Stop Loss:          ${levels.stop_loss:>15,.2f}")
        print(f"  Take Profit:        ${levels.take_profit:>15,.2f}")
        print(f"  Strategy:           {levels.strategy.value:>15}")
        
        print(f"\nRISK/REWARD METRICS:")
        print(f"  Risk Distance:      ${levels.risk_amount:>15,.2f}")
        print(f"  Reward Distance:    ${levels.reward_amount:>15,.2f}")
        print(f"  Risk/Reward Ratio:  {levels.risk_reward_ratio:>15.2f}:1")
        print(f"  Valid Setup:        {'✓ YES' if levels.is_valid else '✗ NO':>15}")
        
        print(f"\nPOSITION SIZING (${account_size:,.0f} account, {risk_pct}% risk):")
        print(f"  Position Size:      {position_info['position_size']:>15.2f} shares")
        print(f"  Risk Amount:        ${position_info['risk_amount']:>15,.2f}")
        print(f"  Expected Profit:    ${position_info['expected_profit']:>15,.2f}")
        print(f"  Max Position Value: ${position_info['max_position_value']:>15,.2f}")
        
        print("=" * 70 + "\n")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    engine = SLTPEngine(min_rr_ratio=1.5)
    
    # Test ATR-based
    print("\n=== ATR-Based Calculation ===")
    levels_atr = engine.calculate_atr_based(
        entry_price=150.0,
        direction="long",
        atr=2.5,
        atr_multiplier=2.0
    )
    if levels_atr:
        engine.print_levels(levels_atr, account_size=100000, risk_pct=1.0)
    
    # Test Fixed Points
    print("\n=== Fixed Points Calculation ===")
    levels_fixed = engine.calculate_fixed_points(
        entry_price=150.0,
        direction="long",
        risk_points=5.0
    )
    if levels_fixed:
        engine.print_levels(levels_fixed, account_size=100000, risk_pct=1.0)
    
    # Test Support/Resistance
    print("\n=== Support/Resistance Calculation ===")
    levels_sr = engine.calculate_support_resistance(
        entry_price=150.0,
        direction="long",
        support=140.0,
        resistance=165.0
    )
    if levels_sr:
        engine.print_levels(levels_sr, account_size=100000, risk_pct=1.0)

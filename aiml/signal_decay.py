# ============================================================================
# SIGNAL_DECAY.PY - Signal Lifecycle & Validity Tracker
# Measures how long a generated signal remains valid in current market conditions
# ============================================================================

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from enum import Enum

import pandas as pd

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Classification of signal types."""
    DIRECTIONAL = "directional"  # Trend-based signals
    REVERSAL = "reversal"        # Mean-reversion signals
    BREAKOUT = "breakout"        # Volatility-based signals
    CONFLUENCE = "confluence"    # Multi-factor signals


@dataclass
class SignalDecay:
    """Tracks the lifecycle and validity of a trading signal."""
    
    signal_id: str
    signal_type: SignalType
    direction: int  # 1 = long, 0 = short
    generated_at: datetime
    entry_price: float
    initial_confidence: float  # 0-1
    
    # Validity parameters
    ttl_seconds: int = 3600  # Time-to-live (default 1 hour)
    consecutive_breakages: int = 0  # Times price crossed against signal
    max_allowed_breakages: int = 3  # Threshold before invalidation
    
    # State tracking
    is_expired: bool = False
    is_invalidated: bool = False
    is_active: bool = True
    current_confidence: float = field(init=False)
    last_checked: datetime = field(default_factory=datetime.now)
    
    def __post_init__(self):
        """Initialize current confidence."""
        self.current_confidence = self.initial_confidence
    
    def check_ttl_expired(self, current_time: datetime) -> bool:
        """
        Check if signal has exceeded time-to-live.
        
        Args:
            current_time: Current time
            
        Returns:
            True if TTL expired, False otherwise
        """
        age = (current_time - self.generated_at).total_seconds()
        self.is_expired = age > self.ttl_seconds
        
        if self.is_expired and self.is_active:
            logger.info(f"⏰ Signal {self.signal_id} expired after {age:.0f}s")
            self.is_active = False
        
        return self.is_expired
    
    def apply_confidence_decay(self, current_time: datetime, decay_rate: float = 0.01) -> float:
        """
        Apply exponential decay to signal confidence over time.
        
        Confidence decays as signal ages, reflecting increasing uncertainty
        in validating the original market condition.
        
        Formula: confidence = initial_confidence * exp(-decay_rate * age_seconds)
        
        Args:
            current_time: Current time
            decay_rate: Decay factor (0.01 = 1% per minute)
            
        Returns:
            Updated confidence score (0-1)
        """
        age_minutes = (current_time - self.generated_at).total_seconds() / 60
        decay_factor = 2.71828 ** (-decay_rate * age_minutes)
        self.current_confidence = max(0.0, self.initial_confidence * decay_factor)
        self.last_checked = current_time
        
        return self.current_confidence
    
    def register_breakage(self, current_price: float, current_time: datetime) -> bool:
        """
        Register a price breakage against signal direction.
        
        If price moves opposite to signal direction, increment breach counter.
        After N breaches, invalidate the signal.
        
        Args:
            current_price: Current market price
            current_time: Current time
            
        Returns:
            True if signal now invalidated, False otherwise
        """
        # Check if price moved opposite to signal
        signal_direction = "up" if self.direction == 1 else "down"
        if self.direction == 1 and current_price < self.entry_price:
            # Bullish signal but price down
            self.consecutive_breakages += 1
        elif self.direction == 0 and current_price > self.entry_price:
            # Bearish signal but price up
            self.consecutive_breakages += 1
        else:
            # Price moved with signal - reset counter
            self.consecutive_breakages = max(0, self.consecutive_breakages - 1)
        
        # Check invalidation threshold
        if self.consecutive_breakages >= self.max_allowed_breakages:
            self.is_invalidated = True
            self.is_active = False
            logger.warning(
                f"❌ Signal {self.signal_id} invalidated: "
                f"{self.consecutive_breakages} breaches at {current_price}"
            )
            return True
        
        return False
    
    def update_confidence_on_adverse_move(self, current_price: float) -> float:
        """
        Reduce confidence if price has moved significantly against signal.
        
        Args:
            current_price: Current market price
            
        Returns:
            New confidence score
        """
        if not self.is_active:
            return self.current_confidence
        
        # Calculate adverse move %
        adverse_pct = abs(current_price - self.entry_price) / self.entry_price
        
        # Each 1% adverse move reduces confidence by 5%
        confidence_reduction = adverse_pct * 5
        self.current_confidence = max(0.0, self.current_confidence - confidence_reduction)
        
        return self.current_confidence
    
    def status(self, current_time: datetime = None) -> Dict:
        """
        Get current signal status.
        
        Args:
            current_time: Current time (defaults to now)
            
        Returns:
            Dictionary with signal state
        """
        current_time = current_time or datetime.now()
        
        return {
            'signal_id': self.signal_id,
            'signal_type': self.signal_type.value,
            'is_active': self.is_active,
            'is_expired': self.is_expired,
            'is_invalidated': self.is_invalidated,
            'current_confidence': self.current_confidence,
            'age_seconds': (current_time - self.generated_at).total_seconds(),
            'ttl_seconds': self.ttl_seconds,
            'consecutive_breakages': self.consecutive_breakages,
            'generated_at': self.generated_at.isoformat(),
        }


class SignalDecayTracker:
    """
    Manages a portfolio of signals and their decay lifecycles.
    
    Optimized for 24/7 operation:
    - O(1) signal lookup by ID
    - O(N) batch status updates
    - Automatic cleanup of expired signals
    """
    
    def __init__(self, max_signals: int = 1000):
        """
        Initialize signal tracker.
        
        Args:
            max_signals: Maximum concurrent signals (for memory planning)
        """
        self.signals: Dict[str, SignalDecay] = {}
        self.max_signals = max_signals
        logger.info(f"✓ SignalDecayTracker initialized (capacity: {max_signals})")
    
    def add_signal(self, signal: SignalDecay) -> None:
        """
        Register a new signal for tracking.
        
        Args:
            signal: SignalDecay object
        """
        if len(self.signals) >= self.max_signals:
            logger.warning(f"⚠️  Signal tracker at capacity ({self.max_signals})")
            # Clean up oldest expired signals
            self.cleanup_expired()
        
        self.signals[signal.signal_id] = signal
        logger.info(f"📍 Signal {signal.signal_id} registered ({len(self.signals)} active)")
    
    def get_signal(self, signal_id: str) -> Optional[SignalDecay]:
        """Get signal by ID."""
        return self.signals.get(signal_id)
    
    def remove_signal(self, signal_id: str) -> bool:
        """Remove signal from tracking."""
        if signal_id in self.signals:
            del self.signals[signal_id]
            return True
        return False
    
    def update_all(self, current_time: datetime, price_data: Dict[str, float] = None) -> None:
        """
        Update all signals' decay status and validity.
        
        Fast bulk operation for monitoring all active signals.
        
        Args:
            current_time: Current time
            price_data: Optional dict of {signal_id: current_price}
        """
        expired_count = 0
        invalidated_count = 0
        
        for signal_id, signal in list(self.signals.items()):
            # Check TTL
            if signal.check_ttl_expired(current_time):
                expired_count += 1
                continue
            
            # Apply confidence decay
            signal.apply_confidence_decay(current_time)
            
            # Update on price if available
            if price_data and signal_id in price_data:
                current_price = price_data[signal_id]
                
                # Check for breaches
                if signal.register_breakage(current_price, current_time):
                    invalidated_count += 1
                
                # Update confidence based on adverse move
                signal.update_confidence_on_adverse_move(current_price)
        
        if expired_count > 0 or invalidated_count > 0:
            logger.info(
                f"🔄 Signal update: {expired_count} expired, "
                f"{invalidated_count} invalidated"
            )
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired or invalidated signals.
        
        Returns:
            Number of signals removed
        """
        to_remove = [
            signal_id for signal_id, signal in self.signals.items()
            if signal.is_expired or signal.is_invalidated
        ]
        
        for signal_id in to_remove:
            del self.signals[signal_id]
        
        if to_remove:
            logger.info(f"🧹 Cleaned up {len(to_remove)} inactive signals")
        
        return len(to_remove)
    
    def get_active_signals(self, current_time: datetime = None) -> List[SignalDecay]:
        """Get all active (non-expired, non-invalidated) signals."""
        current_time = current_time or datetime.now()
        return [
            signal for signal in self.signals.values()
            if (signal.is_active and 
                not signal.check_ttl_expired(current_time) and 
                not signal.is_invalidated)
        ]
    
    def get_signals_by_type(self, signal_type: SignalType) -> List[SignalDecay]:
        """Get all signals of a specific type."""
        return [
            signal for signal in self.signals.values()
            if signal.signal_type == signal_type
        ]
    
    def get_portfolio_confidence(self) -> float:
        """
        Calculate average confidence across all active signals.
        
        Returns:
            Mean confidence (0-1)
        """
        active = self.get_active_signals()
        if not active:
            return 0.5
        return np.mean([s.current_confidence for s in active])
    
    def get_status_report(self, current_time: datetime = None) -> Dict:
        """
        Generate portfolio status report.
        
        Returns:
            Dictionary with aggregate statistics
        """
        current_time = current_time or datetime.now()
        active = self.get_active_signals(current_time)
        
        return {
            'total_signals': len(self.signals),
            'active_signals': len(active),
            'expired_signals': sum(1 for s in self.signals.values() if s.is_expired),
            'invalidated_signals': sum(1 for s in self.signals.values() if s.is_invalidated),
            'avg_confidence': np.mean([s.current_confidence for s in active]) if active else 0,
            'bullish_signals': sum(1 for s in active if s.direction == 1),
            'bearish_signals': sum(1 for s in active if s.direction == 0),
            'timestamp': current_time.isoformat(),
        }


import numpy as np

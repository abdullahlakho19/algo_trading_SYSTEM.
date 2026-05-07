# ============================================================================
# CIRCUIT BREAKER - L5 RISK MANAGEMENT
# Automatically pauses ALL trading when loss limits are breached
# ============================================================================

import logging
from typing import Optional, Dict, List, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

class CircuitBreakerStatus(Enum):
    """Circuit breaker status enumeration."""
    ARMED = "armed"  # Normal operation, ready to trade
    TRIGGERED = "triggered"  # Loss limit breached, trading paused
    COOLDOWN = "cooldown"  # Waiting before re-arming
    MANUAL_OVERRIDE = "manual_override"  # Manually disabled by user


@dataclass
class CBEvent:
    """Circuit breaker event record."""
    timestamp: datetime
    event_type: str  # "breach_daily" | "breach_weekly" | "breach_drawdown" | "reset" | "armed"
    status: CircuitBreakerStatus
    triggered_by: Optional[str] = None  # Which metric triggered it
    metric_value: Optional[float] = None  # Value of triggering metric
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type,
            "status": self.status.value,
            "triggered_by": self.triggered_by,
            "metric_value": self.metric_value,
            "message": self.message,
        }


# ============================================================================
# CIRCUIT BREAKER ENGINE
# ============================================================================

class CircuitBreaker:
    """
    Circuit breaker system for automated risk mitigation.
    
    PURPOSE (from PDF):
    "Pause ALL trading when daily loss limit is breached."
    
    WORKFLOW:
    1. Monitor daily/weekly P&L from PortfolioRisk
    2. If daily loss > threshold: TRIGGER circuit breaker
    3. All trading PAUSED (no new orders allowed)
    4. After cooldown period: Optionally re-arm (if broker allows)
    5. On daily reset: Automatic re-arm if within limits
    
    STATES:
    - ARMED: Normal operation, accepting trades
    - TRIGGERED: Trading paused, loss limit breached
    - COOLDOWN: Waiting before automatic re-arm
    - MANUAL_OVERRIDE: User has disabled CB
    
    REAL-WORLD APPLICATION:
    In production markets, circuit breakers are standard practice:
    - NYSE: Market-wide halts at -10%, -20%, -30% of index
    - CBOE: Halt if VIX moves >50%
    - Institution order: Halt entire fund if daily loss > 2%
    
    This implementation follows institutional best practices.
    """
    
    def __init__(
        self,
        daily_loss_threshold_pct: float = 2.0,
        weekly_loss_threshold_pct: float = 5.0,
        drawdown_threshold_pct: float = 10.0,
        cooldown_minutes: int = 60,
        auto_rearm_daily_reset: bool = True,
    ):
        """
        Initialize Circuit Breaker.
        
        Args:
            daily_loss_threshold_pct: Daily loss % to trigger CB
            weekly_loss_threshold_pct: Weekly loss % to trigger CB
            drawdown_threshold_pct: Drawdown % to trigger CB
            cooldown_minutes: Minutes to wait before auto re-arm
            auto_rearm_daily_reset: Auto-arm after market close
        """
        self.daily_loss_threshold = daily_loss_threshold_pct
        self.weekly_loss_threshold = weekly_loss_threshold_pct
        self.drawdown_threshold = drawdown_threshold_pct
        self.cooldown_duration = timedelta(minutes=cooldown_minutes)
        self.auto_rearm_daily_reset = auto_rearm_daily_reset
        
        # State tracking
        self.status = CircuitBreakerStatus.ARMED
        self.is_trading_allowed = True
        self.triggered_at: Optional[datetime] = None
        self.cooldown_until: Optional[datetime] = None
        
        # Event history
        self.events: List[CBEvent] = []
        
        # Callbacks
        self.on_trigger_callbacks: List[Callable] = []
        self.on_rearm_callbacks: List[Callable] = []
        
        # Logging
        initial_event = CBEvent(
            timestamp=datetime.now(),
            event_type="armed",
            status=CircuitBreakerStatus.ARMED,
            message="Circuit breaker initialized and armed",
        )
        self.events.append(initial_event)
        
        logger.info(f"✓ CircuitBreaker initialized")
        logger.info(f"  Daily Loss Limit: {daily_loss_threshold_pct}%")
        logger.info(f"  Weekly Loss Limit: {weekly_loss_threshold_pct}%")
        logger.info(f"  Drawdown Limit: {drawdown_threshold_pct}%")
        logger.info(f"  Cooldown Duration: {cooldown_minutes} minutes")
    
    # ========================================================================
    # MONITORING & TRIGGERING
    # ========================================================================
    
    def check_daily_loss(self, daily_loss_pct: float) -> bool:
        """
        Check if daily loss exceeds threshold.
        
        Args:
            daily_loss_pct: Current daily loss as percentage
        
        Returns:
            True if threshold breached, False otherwise
        """
        is_breached = daily_loss_pct <= -self.daily_loss_threshold
        
        if is_breached and self.is_trading_allowed:
            self._trigger(
                triggered_by="daily_loss",
                metric_value=daily_loss_pct,
                message=f"Daily loss limit exceeded: {daily_loss_pct:.2f}%"
            )
        
        return is_breached
    
    def check_weekly_loss(self, weekly_loss_pct: float) -> bool:
        """
        Check if weekly loss exceeds threshold.
        
        Args:
            weekly_loss_pct: Current weekly loss as percentage
        
        Returns:
            True if threshold breached, False otherwise
        """
        is_breached = weekly_loss_pct <= -self.weekly_loss_threshold
        
        if is_breached and self.is_trading_allowed:
            self._trigger(
                triggered_by="weekly_loss",
                metric_value=weekly_loss_pct,
                message=f"Weekly loss limit exceeded: {weekly_loss_pct:.2f}%"
            )
        
        return is_breached
    
    def check_max_drawdown(self, drawdown_pct: float) -> bool:
        """
        Check if maximum drawdown exceeds threshold.
        
        Args:
            drawdown_pct: Current drawdown as percentage
        
        Returns:
            True if threshold breached, False otherwise
        """
        is_breached = drawdown_pct >= self.drawdown_threshold
        
        if is_breached and self.is_trading_allowed:
            self._trigger(
                triggered_by="max_drawdown",
                metric_value=drawdown_pct,
                message=f"Maximum drawdown limit exceeded: {drawdown_pct:.2f}%"
            )
        
        return is_breached
    
    def _trigger(
        self,
        triggered_by: str,
        metric_value: Optional[float] = None,
        message: str = ""
    ) -> None:
        """
        Trigger circuit breaker.
        
        Args:
            triggered_by: Metric that triggered CB
            metric_value: Value of the metric
            message: Event message
        """
        if self.status == CircuitBreakerStatus.TRIGGERED:
            # Already triggered, don't re-trigger
            return
        
        self.status = CircuitBreakerStatus.TRIGGERED
        self.is_trading_allowed = False
        self.triggered_at = datetime.now()
        self.cooldown_until = self.triggered_at + self.cooldown_duration
        
        # Record event
        event = CBEvent(
            timestamp=self.triggered_at,
            event_type=f"breach_{triggered_by}",
            status=CircuitBreakerStatus.TRIGGERED,
            triggered_by=triggered_by,
            metric_value=metric_value,
            message=message,
        )
        self.events.append(event)
        
        logger.error(f"🔴 CIRCUIT BREAKER TRIGGERED: {message}")
        logger.error(f"   Status: TRADING PAUSED")
        logger.error(f"   Cooldown until: {self.cooldown_until.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute callbacks
        for callback in self.on_trigger_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in CB trigger callback: {e}")
    
    # ========================================================================
    # RECOVERY & RE-ARMING
    # ========================================================================
    
    def check_cooldown_expired(self) -> bool:
        """
        Check if cooldown period has expired.
        
        Returns:
            True if cooldown has expired
        """
        if self.cooldown_until is None:
            return False
        
        return datetime.now() >= self.cooldown_until
    
    def attempt_rearm(self) -> bool:
        """
        Attempt to re-arm circuit breaker after cooldown.
        
        Returns:
            True if successfully re-armed, False if still in cooldown
        """
        if self.status != CircuitBreakerStatus.TRIGGERED:
            return True  # Already armed
        
        if not self.check_cooldown_expired():
            minutes_remaining = (
                (self.cooldown_until - datetime.now()).total_seconds() / 60
            )
            logger.warning(f"Circuit breaker still in cooldown ({minutes_remaining:.1f} min remaining)")
            return False
        
        # Cooldown expired, re-arm
        self._rearm()
        return True
    
    def _rearm(self) -> None:
        """Re-arm circuit breaker (automatic after cooldown)."""
        self.status = CircuitBreakerStatus.ARMED
        self.is_trading_allowed = True
        self.triggered_at = None
        self.cooldown_until = None
        
        # Record event
        event = CBEvent(
            timestamp=datetime.now(),
            event_type="reset",
            status=CircuitBreakerStatus.ARMED,
            message="Circuit breaker re-armed after cooldown",
        )
        self.events.append(event)
        
        logger.info(f"🟢 CIRCUIT BREAKER RE-ARMED")
        logger.info(f"   Status: TRADING RESUMED")
        logger.info(f"   Timestamp: {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Execute callbacks
        for callback in self.on_rearm_callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Error in CB rearm callback: {e}")
    
    def daily_reset(self) -> None:
        """
        Called at market close to reset daily loss tracking and auto-rearm if needed.
        
        This should be called by the main trading loop at EOD.
        """
        if not self.auto_rearm_daily_reset:
            return
        
        if self.status == CircuitBreakerStatus.TRIGGERED:
            logger.info("Daily reset: Attempting to re-arm circuit breaker...")
            self.attempt_rearm()
        
        # Log daily summary
        logger.info(f"Daily CB Status: {self.status.value} | Trading Allowed: {self.is_trading_allowed}")
    
    # ========================================================================
    # MANUAL CONTROL
    # ========================================================================
    
    def manual_override(self, reason: str = "User override") -> None:
        """
        Manually disable circuit breaker (requires user confirmation).
        
        WARNING: This bypasses automatic risk controls. Use carefully.
        
        Args:
            reason: Reason for override (logged for audit trail)
        """
        self.status = CircuitBreakerStatus.MANUAL_OVERRIDE
        self.is_trading_allowed = True
        
        logger.warning(f"🔴 CIRCUIT BREAKER MANUAL OVERRIDE")
        logger.warning(f"   Reason: {reason}")
        logger.warning(f"   This disables automatic risk protection!")
        logger.warning(f"   Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        event = CBEvent(
            timestamp=datetime.now(),
            event_type="manual_override",
            status=CircuitBreakerStatus.MANUAL_OVERRIDE,
            message=f"Manual override: {reason}",
        )
        self.events.append(event)
    
    def disarm_override(self) -> None:
        """Re-enable automatic circuit breaker after manual override."""
        if self.status == CircuitBreakerStatus.MANUAL_OVERRIDE:
            self._rearm()
            logger.info("Manual override disarmed, circuit breaker re-enabled")
    
    # ========================================================================
    # QUERY & REPORTING
    # ========================================================================
    
    def is_trading_paused(self) -> bool:
        """Check if trading is currently paused."""
        return not self.is_trading_allowed
    
    def get_status(self) -> str:
        """Get human-readable status string."""
        if self.status == CircuitBreakerStatus.ARMED:
            return "🟢 ARMED (trading allowed)"
        elif self.status == CircuitBreakerStatus.TRIGGERED:
            remaining = (
                (self.cooldown_until - datetime.now()).total_seconds() / 60
                if self.cooldown_until
                else 0
            )
            return f"🔴 TRIGGERED (paused, {remaining:.1f} min to rearm)"
        elif self.status == CircuitBreakerStatus.COOLDOWN:
            return "🟡 COOLDOWN (waiting to rearm)"
        else:  # MANUAL_OVERRIDE
            return "🟠 MANUAL OVERRIDE (risk controls disabled)"
    
    def get_last_event(self) -> Optional[CBEvent]:
        """Get the most recent circuit breaker event."""
        return self.events[-1] if self.events else None
    
    def get_events(self, limit: int = 10) -> List[CBEvent]:
        """Get recent events."""
        return self.events[-limit:]
    
    def register_trigger_callback(self, callback: Callable[[CBEvent], None]) -> None:
        """
        Register callback to be called when CB is triggered.
        
        Args:
            callback: Function(event) called on trigger
        """
        self.on_trigger_callbacks.append(callback)
    
    def register_rearm_callback(self, callback: Callable[[CBEvent], None]) -> None:
        """
        Register callback to be called when CB is re-armed.
        
        Args:
            callback: Function(event) called on rearm
        """
        self.on_rearm_callbacks.append(callback)
    
    # ========================================================================
    # REPORTING
    # ========================================================================
    
    def print_status(self) -> None:
        """Print circuit breaker status."""
        print("\n" + "=" * 70)
        print("CIRCUIT BREAKER STATUS")
        print("=" * 70)
        print(f"\nStatus: {self.get_status()}")
        print(f"\nConfiguration:")
        print(f"  Daily Loss Limit: {self.daily_loss_threshold}%")
        print(f"  Weekly Loss Limit: {self.weekly_loss_threshold}%")
        print(f"  Drawdown Limit: {self.drawdown_threshold}%")
        print(f"  Cooldown Duration: {self.cooldown_duration.total_seconds() / 60:.0f} minutes")
        
        if self.status == CircuitBreakerStatus.TRIGGERED:
            print(f"\nTriggered Information:")
            print(f"  Triggered At: {self.triggered_at.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Cooldown Until: {self.cooldown_until.strftime('%Y-%m-%d %H:%M:%S')}")
            remaining = (self.cooldown_until - datetime.now()).total_seconds() / 60
            print(f"  Remaining: {remaining:.1f} minutes")
        
        if self.events:
            print(f"\nRecent Events:")
            for event in self.events[-5:]:
                print(f"  [{event.timestamp.strftime('%H:%M:%S')}] "
                      f"{event.event_type}: {event.message}")
        
        print("=" * 70 + "\n")


# ============================================================================
# STANDALONE TESTING
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
    )
    
    # Create circuit breaker
    cb = CircuitBreaker(
        daily_loss_threshold_pct=2.0,
        weekly_loss_threshold_pct=5.0,
        drawdown_threshold_pct=10.0,
        cooldown_minutes=5,  # 5 min for testing
    )
    
    cb.print_status()
    
    # Simulate trading conditions
    print("\n--- Scenario 1: Normal Trading ---")
    is_breached = cb.check_daily_loss(-0.5)  # Only -0.5% loss
    print(f"Daily loss -0.5%: Breached={is_breached}, CB Status={cb.get_status()}")
    
    print("\n--- Scenario 2: Loss Limit Breached ---")
    is_breached = cb.check_daily_loss(-2.5)  # -2.5% loss (exceeds 2%)
    print(f"Daily loss -2.5%: Breached={is_breached}, CB Status={cb.get_status()}")
    
    print("\n--- Scenario 3: Cooldown Check ---")
    can_rearm = cb.check_cooldown_expired()
    print(f"Cooldown expired immediately: {can_rearm}")
    
    # Print final status
    cb.print_status()

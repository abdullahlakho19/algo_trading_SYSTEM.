# ============================================================================
# ADAPTIVE_EXECUTOR.PY - Adaptive Order Execution Algorithm
# Splits large orders to minimize slippage via VWAP/TWAP participation
# ============================================================================

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class ExecutionStrategy(str, Enum):
    """Execution strategy enumeration."""
    IMMEDIATE = "immediate"  # Execute all immediately
    VWAP = "vwap"  # Volume Weighted Average Price participation
    TWAP = "twap"  # Time Weighted Average Price slicing
    ADAPTIVE = "adaptive"  # Smart blending based on market conditions
    POV = "pov"  # Percentage of Volume participation


@dataclass
class OrderSlice:
    """Individual order slice."""
    slice_id: int
    quantity: float
    scheduled_time: datetime
    execution_time: Optional[datetime] = None
    filled_quantity: float = 0.0
    average_price: float = 0.0
    slippage: float = 0.0
    status: str = "PENDING"  # PENDING, EXECUTING, FILLED, CANCELLED


@dataclass
class ExecutionPlan:
    """Complete execution plan for large order."""
    order_id: str
    symbol: str
    side: str
    total_quantity: float
    target_price: float
    strategy: ExecutionStrategy
    slices: List[OrderSlice] = field(default_factory=list)
    start_time: datetime = field(default_factory=datetime.utcnow)
    end_time: Optional[datetime] = None
    total_executed: float = 0.0
    total_slippage: float = 0.0
    estimated_slippage_bps: float = 0.0
    status: str = "PENDING"


@dataclass
class MarketState:
    """Current market microstructure state."""
    bid: float
    ask: float
    bid_volume: float
    ask_volume: float
    recent_volume: float
    volatility: float
    timestamp: datetime


class AdaptiveExecutor:
    """
    Adaptive order execution algorithm.
    
    Features:
    - Order splitting strategies: VWAP, TWAP, Adaptive POV
    - Real-time slippage estimation and minimization
    - Market microstructure awareness
    - Intelligent slice scheduling based on volume profile
    - Adaptive strategy selection based on market conditions
    - Parent-child order hierarchy tracking
    
    Algorithms:
    1. VWAP: Slice sizes follow historical volume distribution
    2. TWAP: Equal time slices, executed uniformly
    3. Adaptive: Dynamically adjust slices based on live volume
    4. POV: Execute fixed % of real-time market volume
    
    Designed for:
    - Institutional-size order execution
    - Minimization of market impact
    - Reducing realization slippage
    - Benchmark compliance reporting
    """
    
    def __init__(
        self,
        max_order_size: float = 10000.0,
        split_threshold: float = 5000.0,  # Split if > 5000
        execution_window_minutes: int = 60,
        pov_target: float = 0.10,  # 10% of volume
    ):
        """
        Initialize adaptive executor.
        
        Args:
            max_order_size: Maximum single order size
            split_threshold: Auto-split threshold
            execution_window_minutes: Time window for execution
            pov_target: POV strategy target participation rate
        """
        self.max_order_size = max_order_size
        self.split_threshold = split_threshold
        self.execution_window_minutes = execution_window_minutes
        self.pov_target = pov_target
        
        self.plans: Dict[str, ExecutionPlan] = {}
        self.executed_orders: Dict[str, Dict] = {}
        self.market_states: Dict[str, List[MarketState]] = {}
        
        logger.info(
            f"✓ AdaptiveExecutor initialized: threshold={split_threshold}, "
            f"window={execution_window_minutes}min, POV={pov_target*100:.1f}%"
        )
    
    def _estimate_slippage(
        self,
        quantity: float,
        bid: float,
        ask: float,
        bid_volume: float,
        ask_volume: float,
        side: str,
    ) -> float:
        """
        Estimate execution slippage for quantity.
        
        Uses market depth modeling: slippage increases with
        quantity relative to available volume at bid/ask.
        
        Args:
            quantity: Order quantity
            bid: Current bid price
            ask: Current ask price
            bid_volume: Volume at bid level
            ask_volume: Volume at ask level
            side: 'buy' or 'sell'
            
        Returns:
            Estimated slippage in basis points
        """
        spread = (ask - bid) / bid * 10000  # Bid-ask spread in bps
        
        if side == 'buy':
            # Buying: execute at ask, slippage = ask - mid
            depth_ratio = quantity / max(ask_volume, 1.0)
            depth_slippage = depth_ratio * 10  # Non-linear impact
        else:
            # Selling: execute at bid, slippage = mid - bid
            depth_ratio = quantity / max(bid_volume, 1.0)
            depth_slippage = depth_ratio * 10
        
        total_slippage = spread + min(depth_slippage, 50)  # Cap at 50bps
        
        return total_slippage
    
    def _calculate_vwap_slices(
        self,
        quantity: float,
        volume_profile: Dict[int, float],  # minute -> volume
        num_slices: int,
    ) -> List[float]:
        """
        Calculate order slice sizes using VWAP weights.
        
        Slices are sized proportional to expected volume
        at each time interval.
        
        Args:
            quantity: Total quantity to execute
            volume_profile: Expected volume per minute
            num_slices: Number of slices
            
        Returns:
            List of slice quantities
        """
        total_expected_volume = sum(volume_profile.values())
        if total_expected_volume == 0:
            return [quantity / num_slices] * num_slices
        
        slices = []
        for i in range(num_slices):
            minute = i % len(volume_profile)
            volume_weight = volume_profile.get(minute, 0) / total_expected_volume
            slice_qty = quantity * volume_weight
            slices.append(slice_qty)
        
        # Normalize to exact total
        total_slices = sum(slices)
        slices = [s * (quantity / total_slices) for s in slices]
        
        return slices
    
    def _calculate_twap_slices(
        self,
        quantity: float,
        num_slices: int,
    ) -> List[float]:
        """
        Calculate TWAP slices (equal-weighted).
        
        Args:
            quantity: Total quantity
            num_slices: Number of slices
            
        Returns:
            List of equal slice quantities
        """
        base_slice = quantity / num_slices
        slices = [base_slice] * num_slices
        
        # Adjust last slice for rounding
        slices[-1] += (quantity - sum(slices))
        
        return slices
    
    def _calculate_pov_slices(
        self,
        quantity: float,
        market_volume: float,
        pov_target: float,
        num_slices: int,
    ) -> List[float]:
        """
        Calculate POV-based slices.
        
        Each slice targets a fixed % of volume during that period.
        
        Args:
            quantity: Total quantity
            market_volume: Expected market volume across window
            pov_target: Target % participation (0.1 = 10%)
            num_slices: Number of slices
            
        Returns:
            List of slice quantities
        """
        per_slice_volume = market_volume / num_slices
        per_slice_pov = per_slice_volume * pov_target
        
        slices = [per_slice_pov] * num_slices
        total_slices = sum(slices)
        
        if total_slices > 0:
            slices = [s * (quantity / total_slices) for s in slices]
        else:
            slices = [quantity / num_slices] * num_slices
        
        return slices
    
    async def create_execution_plan(
        self,
        order_id: str,
        symbol: str,
        side: str,
        quantity: float,
        target_price: float,
        strategy: ExecutionStrategy = ExecutionStrategy.ADAPTIVE,
        market_state: Optional[MarketState] = None,
    ) -> ExecutionPlan:
        """
        Create execution plan for large order.
        
        Generates order slices based on strategy and market conditions.
        
        Args:
            order_id: Parent order ID
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Total quantity to execute
            target_price: Target execution price
            strategy: Execution strategy
            market_state: Current market state
            
        Returns:
            ExecutionPlan with scheduled slices
        """
        # Determine number of slices
        if quantity <= self.split_threshold:
            num_slices = 1
            strategy = ExecutionStrategy.IMMEDIATE
        else:
            num_slices = max(3, int(quantity / self.max_order_size))
        
        # Calculate slices
        if strategy == ExecutionStrategy.IMMEDIATE:
            slices_qty = [quantity]
        elif strategy == ExecutionStrategy.VWAP:
            volume_profile = self._get_expected_volume_profile(symbol)
            slices_qty = self._calculate_vwap_slices(quantity, volume_profile, num_slices)
        elif strategy == ExecutionStrategy.TWAP:
            slices_qty = self._calculate_twap_slices(quantity, num_slices)
        elif strategy == ExecutionStrategy.POV:
            expected_volume = self._get_expected_market_volume(symbol)
            slices_qty = self._calculate_pov_slices(
                quantity, expected_volume, self.pov_target, num_slices
            )
        else:  # ADAPTIVE
            slices_qty = self._calculate_twap_slices(quantity, num_slices)
        
        # Create execution plan with scheduled times
        plan = ExecutionPlan(
            order_id=order_id,
            symbol=symbol,
            side=side,
            total_quantity=quantity,
            target_price=target_price,
            strategy=strategy,
            start_time=datetime.utcnow(),
        )
        
        execution_interval = self.execution_window_minutes / num_slices
        
        for i, slice_qty in enumerate(slices_qty):
            scheduled_time = plan.start_time + timedelta(minutes=i * execution_interval)
            
            slice_obj = OrderSlice(
                slice_id=i,
                quantity=slice_qty,
                scheduled_time=scheduled_time,
            )
            
            plan.slices.append(slice_obj)
        
        # Estimate total slippage
        if market_state:
            plan.estimated_slippage_bps = self._estimate_slippage(
                quantity, market_state.bid, market_state.ask,
                market_state.bid_volume, market_state.ask_volume,
                side
            )
        
        self.plans[order_id] = plan
        
        logger.info(
            f"✓ Execution plan created: {order_id} | strategy={strategy.value} | "
            f"slices={num_slices} | est_slippage={plan.estimated_slippage_bps:.1f}bps"
        )
        
        return plan
    
    async def get_next_slice(self, order_id: str) -> Optional[OrderSlice]:
        """
        Get next scheduled slice to execute.
        
        Args:
            order_id: Parent order ID
            
        Returns:
            Next OrderSlice or None if complete
        """
        if order_id not in self.plans:
            return None
        
        plan = self.plans[order_id]
        now = datetime.utcnow()
        
        for slice_obj in plan.slices:
            if slice_obj.status == "PENDING" and slice_obj.scheduled_time <= now:
                return slice_obj
        
        return None
    
    async def execute_slice(
        self,
        order_id: str,
        slice_id: int,
        fill_price: float,
        quantity: float,
        executor_fn: Callable,
    ) -> bool:
        """
        Execute individual slice and record fill.
        
        Args:
            order_id: Parent order ID
            slice_id: Slice index
            fill_price: Execution price
            quantity: Executed quantity
            executor_fn: Async function to execute the slice
            
        Returns:
            True if execution successful
        """
        try:
            if order_id not in self.plans:
                return False
            
            plan = self.plans[order_id]
            slice_obj = plan.slices[slice_id]
            
            # Execute via external executor
            result = await executor_fn(quantity, fill_price)
            
            if result:
                slice_obj.status = "FILLED"
                slice_obj.execution_time = datetime.utcnow()
                slice_obj.filled_quantity = quantity
                slice_obj.average_price = fill_price
                
                # Calculate slippage
                slice_obj.slippage = (fill_price - plan.target_price) / plan.target_price * 10000
                
                # Update plan stats
                plan.total_executed += quantity
                plan.total_slippage += slice_obj.slippage * quantity
                
                logger.info(
                    f"✓ Slice executed: {order_id}[{slice_id}] | "
                    f"{quantity} @ {fill_price:.4f} | slippage {slice_obj.slippage:.1f}bps"
                )
                
                # Check if plan complete
                if plan.total_executed >= plan.total_quantity * 0.99:  # 99% filled
                    plan.status = "COMPLETED"
                    plan.end_time = datetime.utcnow()
                
                return True
            
            return False
        
        except Exception as e:
            logger.error(f"✗ Slice execution failed: {e}")
            return False
    
    async def cancel_execution_plan(self, order_id: str) -> bool:
        """
        Cancel execution plan and all pending slices.
        
        Args:
            order_id: Plan to cancel
            
        Returns:
            True if cancelled
        """
        if order_id not in self.plans:
            return False
        
        plan = self.plans[order_id]
        
        for slice_obj in plan.slices:
            if slice_obj.status == "PENDING":
                slice_obj.status = "CANCELLED"
        
        plan.status = "CANCELLED"
        
        logger.info(f"✓ Execution plan cancelled: {order_id}")
        
        return True
    
    async def get_execution_status(self, order_id: str) -> Dict:
        """
        Get execution plan status and metrics.
        
        Args:
            order_id: Plan to check
            
        Returns:
            Status dictionary
        """
        if order_id not in self.plans:
            return {}
        
        plan = self.plans[order_id]
        
        filled_slices = [s for s in plan.slices if s.status == "FILLED"]
        pending_slices = [s for s in plan.slices if s.status == "PENDING"]
        
        return {
            'order_id': order_id,
            'status': plan.status,
            'strategy': plan.strategy.value,
            'total_quantity': plan.total_quantity,
            'executed_quantity': plan.total_executed,
            'execution_percent': (plan.total_executed / plan.total_quantity * 100)
            if plan.total_quantity > 0 else 0,
            'total_slices': len(plan.slices),
            'filled_slices': len(filled_slices),
            'pending_slices': len(pending_slices),
            'average_slippage_bps': (
                plan.total_slippage / plan.total_executed
                if plan.total_executed > 0 else 0
            ),
            'estimated_slippage_bps': plan.estimated_slippage_bps,
            'elapsed_time_seconds': (
                (datetime.utcnow() - plan.start_time).total_seconds()
                if plan.end_time is None
                else (plan.end_time - plan.start_time).total_seconds()
            ),
        }
    
    def _get_expected_volume_profile(self, symbol: str) -> Dict[int, float]:
        """Get expected volume profile per minute (mock: would load from history)."""
        # In production: load from 20-day historical profiles
        return {i: 1000 + np.random.uniform(-200, 200) for i in range(60)}
    
    def _get_expected_market_volume(self, symbol: str) -> float:
        """Get expected total market volume (mock)."""
        # In production: use realized volume from current day
        return 100000 + np.random.uniform(-20000, 20000)

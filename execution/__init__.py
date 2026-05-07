# ============================================================================
# EXECUTION MODULE - Layer 6: Execution Engine
# ============================================================================
# Institutional-grade order execution with risk management and compliance
# Stocks and Forex only (Crypto removed)

from execution.adaptive_executor import (
    AdaptiveExecutor,
    ExecutionPlan,
    ExecutionStrategy,
    OrderSlice,
)
from execution.alpaca_executor import (
    AlpacaBracketOrder,
    AlpacaExecutor,
    OrderSide as AlpacaOrderSide,
    OrderStatus as AlpacaOrderStatus,
)
from execution.order_manager import (
    CancellationReason,
    MissedTradeLog,
    MissedTradeProtocol,
    OrderManager,
    TradeOrder,
    TradeState,
)
from execution.paper_simulator import (
    Fill,
    OrderStatus as PaperOrderStatus,
    PaperSimulator,
    Position,
    PositionStatus,
    SimulatedOrder,
)

__all__ = [
    # Paper Simulator
    'PaperSimulator',
    'SimulatedOrder',
    'Fill',
    'Position',
    'PositionStatus',
    'PaperOrderStatus',
    
    # Alpaca Executor
    'AlpacaExecutor',
    'AlpacaBracketOrder',
    'AlpacaOrderSide',
    'AlpacaOrderStatus',
    
    # Adaptive Executor
    'AdaptiveExecutor',
    'ExecutionPlan',
    'ExecutionStrategy',
    'OrderSlice',
    
    # Order Manager
    'OrderManager',
    'MissedTradeProtocol',
    'TradeOrder',
    'TradeState',
    'CancellationReason',
    'MissedTradeLog',
]

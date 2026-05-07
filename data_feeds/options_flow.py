# ============================================================================
# OPTIONS_FLOW.PY - Unusual options activity & flow detection
# ============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import numpy as np


@dataclass
class OptionsFlow:
    """Options flow record."""
    timestamp: datetime
    symbol: str
    strike: float
    call_put: str  # 'CALL' or 'PUT'
    bid: float
    ask: float
    volume: int
    open_interest: int
    iv: float  # Implied volatility
    unusual: bool  # Is this unusual activity


class OptionsFlowDetector:
    """Detects unusual options activity."""
    
    def __init__(self, symbol: str):
        """Initialize detector."""
        self.symbol = symbol
        self.flows: List[OptionsFlow] = []
        self.baseline_volume = {}
    
    def detect_unusual_activity(self, flow: OptionsFlow) -> bool:
        """
        Detect if flow is unusual.
        
        Args:
            flow: OptionsFlow record
        
        Returns:
            True if unusual
        """
        strike_key = f"{flow.strike}_{flow.call_put}"
        
        # Track baseline
        if strike_key not in self.baseline_volume:
            self.baseline_volume[strike_key] = []
        
        self.baseline_volume[strike_key].append(flow.volume)
        
        # Calculate moving average
        recent_volumes = self.baseline_volume[strike_key][-20:]
        avg_volume = np.mean(recent_volumes)
        std_volume = np.std(recent_volumes)
        
        # Unusual if > 2 std devs above mean
        threshold = avg_volume + (2 * std_volume)
        
        return flow.volume > threshold
    
    def get_unusual_flows(self) -> List[OptionsFlow]:
        """Get recent unusual flows."""
        return [f for f in self.flows if f.unusual]
    
    def analyze_put_call_ratio(self) -> dict:
        """Analyze put/call ratio."""
        if not self.flows:
            return {}
        
        recent = self.flows[-100:]
        puts = sum(f.volume for f in recent if f.call_put == 'PUT')
        calls = sum(f.volume for f in recent if f.call_put == 'CALL')
        ratio = puts / calls if calls > 0 else 0
        
        return {
            'put_volume': puts,
            'call_volume': calls,
            'ratio': ratio,
            'sentiment': 'bullish' if ratio < 0.8 else 'bearish' if ratio > 1.2 else 'neutral',
        }


def main():
    """Test options flow."""
    detector = OptionsFlowDetector("AAPL")
    print("Options flow detector initialized")


if __name__ == "__main__":
    main()

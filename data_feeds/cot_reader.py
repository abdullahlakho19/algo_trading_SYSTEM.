# ============================================================================
# COT_READER.PY - COT (Commitments of Traders) report reader & analysis
# ============================================================================

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class COTData:
    """COT report data."""
    report_date: datetime
    contract: str  # e.g., 'EUR/USD', 'GBP/USD'
    commercial_long: int
    commercial_short: int
    non_commercial_long: int
    non_commercial_short: int
    open_interest: int


class COTReader:
    """Reads and analyzes Commitments of Traders reports."""
    
    def __init__(self):
        """Initialize COT reader."""
        self.data: List[COTData] = []
        self.latest_by_contract: Dict[str, COTData] = {}
    
    def add_cot_data(self, data: COTData) -> None:
        """Add COT data point."""
        self.data.append(data)
        self.latest_by_contract[data.contract] = data
    
    def get_net_positioning(self, contract: str) -> dict:
        """Get net positioning for a contract."""
        if contract not in self.latest_by_contract:
            return {}
        
        cot = self.latest_by_contract[contract]
        
        commercial_net = cot.commercial_long - cot.commercial_short
        non_commercial_net = cot.non_commercial_long - cot.non_commercial_short
        
        return {
            'contract': contract,
            'commercial_net': commercial_net,
            'non_commercial_net': non_commercial_net,
            'open_interest': cot.open_interest,
            'commercial_pct': (cot.commercial_long / cot.open_interest * 100) if cot.open_interest > 0 else 0,
        }
    
    def detect_commercial_accumulation(self, contract: str) -> bool:
        """Detect if commercials are accumulating."""
        positioning = self.get_net_positioning(contract)
        if not positioning:
            return False
        
        # Commercials typically short currenciesyet accumulate on weakness
        # If commercial short positions are decreasing, they're accumulating longs (bullish for currency)
        net = positioning['commercial_net']
        return net < 0  # Negative = long positioned commercials
    
    def get_sentiment(self, contract: str) -> str:
        """Get market sentiment from COT."""
        positioning = self.get_net_positioning(contract)
        if not positioning:
            return 'unknown'
        
        net = positioning['non_commercial_net']
        if net > 0:
            return 'bullish'
        elif net < 0:
            return 'bearish'
        else:
            return 'neutral'


def main():
    """Test COT reader."""
    reader = COTReader()
    print("COT reader initialized")


if __name__ == "__main__":
    main()

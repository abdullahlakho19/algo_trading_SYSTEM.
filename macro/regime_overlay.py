# ============================================================================
# REGIME_OVERLAY.PY - Risk-on/off classification for macro layer
# ============================================================================

from datetime import datetime
from enum import Enum
from typing import Dict

import pandas as pd


class RiskRegime(Enum):
    """Risk regime types."""
    RISK_ON = "risk_on"
    RISK_OFF = "risk_off"
    FLIGHT_TO_QUALITY = "flight_to_quality"
    UNCERTAIN = "uncertain"


class RegimeOverlay:
    """Macro risk-on/off regime overlay."""
    
    INDICATORS = {
        'risk_on': {
            'vix': (10, 20),           # Low volatility
            'credit_spreads': 'tight', # Credit spreads narrow
            'commodity_usd': 'weak',   # Dollar weak, commodities up
            'stocks': 'strong',        # Equities outperform
        },
        'risk_off': {
            'vix': (20, 1000),         # High volatility
            'credit_spreads': 'wide',  # Credit spreads widen
            'commodity_usd': 'strong', # Dollar strong
            'stocks': 'weak',          # Equities underperform
        },
        'flight_to_quality': {
            'bonds': 'demand',         # Bond yields crush
            'safe_haven': 'demand',    # Gold, JPY, CHF up
            'vol': 'high',             # Elevated volatility
        }
    }
    
    @staticmethod
    def detect_regime(vix: float = None, spreads: Dict = None) -> RiskRegime:
        """
        Detect current risk regime.
        
        Args:
            vix: VIX level
            spreads: Credit spread data
        
        Returns:
            Current RiskRegime
        """
        if vix is None:
            return RiskRegime.UNCERTAIN
        
        if vix < 15:
            return RiskRegime.RISK_ON
        elif vix > 25:
            if spreads and spreads.get('hg') and spreads['hg'] > 300:
                return RiskRegime.FLIGHT_TO_QUALITY
            return RiskRegime.RISK_OFF
        
        return RiskRegime.UNCERTAIN
    
    @staticmethod
    def get_regime_trading_implications(regime: RiskRegime) -> dict:
        """Get trading implications for regime."""
        implications = {
            RiskRegime.RISK_ON: {
                'best_assets': ['STOCKS', 'COMMODITIES', 'EM', 'HIGH_YIELD'],
                'worst_assets': ['BONDS', 'USD', 'VOLATILITY'],
                'strategy': 'Growth, buy dips, momentum',
                'correlations': 'equities-commodities positive, stocks-bonds negative',
            },
            RiskRegime.RISK_OFF: {
                'best_assets': ['BONDS', 'USD', 'GOLD', 'VOLATILITY'],
                'worst_assets': ['STOCKS', 'COMMODITIES', 'EM'],
                'strategy': 'Defensive, reduce leverage, hedge',
                'correlations': 'negative correlation breakdown',
            },
            RiskRegime.FLIGHT_TO_QUALITY: {
                'best_assets': ['LONG_DATED_BONDS', 'GOLD', 'JPY', 'CHF'],
                'worst_assets': ['EM', 'HIGH_YIELD', 'COMMODITIES'],
                'strategy': 'Extreme defensive',
                'correlations': 'everything down except safe havens',
            },
        }
        return implications.get(regime, {})


def main():
    """Test regime overlay."""
    overlay = RegimeOverlay()
    
    # Example detection
    regime = overlay.detect_regime(vix=15)
    print(f"Detected regime: {regime.value}")


if __name__ == "__main__":
    main()

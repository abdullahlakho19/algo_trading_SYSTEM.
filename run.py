# ============================================================================
# RUN.PY - Interactive CLI Launcher & Setup Wizard
# Prompts user for configuration and saves to agent_config.json
# ============================================================================

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SetupWizard:
    """
    Interactive CLI setup wizard for agent configuration.
    
    Prompts user for 6 configuration questions:
    1. Mode: Live trading, paper trading, or backtest
    2. Markets: Asset class (stocks or forex)
    3. Symbols: Which symbols to trade
    4. Capital: Starting capital
    5. Risk: Risk per trade (%)
    6. Confirm: Final confirmation before saving
    
    STOCKS & FOREX ONLY (Crypto removed)
    """
    
    DEFAULT_CONFIG_PATH = Path("agent_config.json")
    
    MODES = {
        "1": "LIVE",
        "2": "PAPER",
        "3": "BACKTEST",
    }
    
    MARKETS = {
        "1": "STOCKS",
        "2": "FOREX",
    }
    
    RISK_PRESETS = {
        "1": 0.5,
        "2": 1.0,
        "3": 2.0,
        "4": 5.0,
    }
    
    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize setup wizard.
        
        Args:
            config_path: Path to config file (default: agent_config.json)
        """
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.config = {}
        
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        logger.info("✓ Setup Wizard initialized")
    
    def load_existing_config(self) -> bool:
        """
        Check if config file exists and offer to load it.
        
        Returns:
            True if existing config should be used
        """
        if not self.config_path.exists():
            return False
        
        print("\n" + "=" * 70)
        print("📋 EXISTING CONFIGURATION FOUND")
        print("=" * 70)
        
        try:
            with open(self.config_path, "r") as f:
                existing = json.load(f)
            
            print(f"\nConfiguration at: {self.config_path}")
            print(f"Mode: {existing.get('mode', 'N/A')}")
            print(f"Markets: {existing.get('markets', 'N/A')}")
            print(f"Symbols: {', '.join(existing.get('symbols', []))}")
            print(f"Capital: ${existing.get('capital', 0):,.2f}")
            print(f"Risk/Trade: {existing.get('risk_per_trade', 0):.1f}%")
            print(f"Created: {existing.get('created_at', 'N/A')}")
            
            response = input("\nUse this configuration? (y/n): ").strip().lower()
            
            if response == "y":
                self.config = existing
                print("\n✓ Loaded existing configuration")
                return True
        
        except Exception as e:
            logger.warning(f"Failed to load config: {e}")
        
        return False
    
    def prompt_mode(self) -> str:
        """Prompt user for trading mode."""
        print("\n" + "=" * 70)
        print("1️⃣  TRADING MODE")
        print("=" * 70)
        print("\nSelect trading mode:")
        print("  1) LIVE       - Connect to real brokers, execute real trades")
        print("  2) PAPER      - Simulated trading with live market prices")
        print("  3) BACKTEST   - Historical data backtesting and optimization")
        
        while True:
            choice = input("\nEnter choice (1-3): ").strip()
            if choice in self.MODES:
                mode = self.MODES[choice]
                print(f"\n✓ Selected: {mode}")
                return mode
            print("Invalid choice. Please enter 1, 2, or 3.")
    
    def prompt_markets(self) -> str:
        """Prompt user for markets (STOCKS & FOREX ONLY)."""
        print("\n" + "=" * 70)
        print("2️⃣  MARKETS [STOCKS & FOREX ONLY]")
        print("=" * 70)
        print("\nSelect markets to trade:")
        print("  1) STOCKS     - US equities (AAPL, MSFT, SPY, etc.)")
        print("  2) FOREX      - Currency pairs (EUR/USD, GBP/USD, etc.)")
        
        while True:
            choice = input("\nEnter choice (1-2): ").strip()
            if choice in self.MARKETS:
                market = self.MARKETS[choice]
                print(f"\n✓ Selected: {market}")
                return market
            print("Invalid choice. Please enter 1 or 2.")
    
    def prompt_symbols(self, market: str) -> list:
        """Prompt user for symbols to trade (STOCKS & FOREX only)."""
        print("\n" + "=" * 70)
        print("3️⃣  SYMBOLS")
        print("=" * 70)
        
        default_symbols = {
            "STOCKS": ["AAPL", "MSFT", "NVDA", "TSLA", "GOOGL", "JPM", "SPY", "QQQ"],
            "FOREX": ["EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "GBP/JPY", "USD/CAD"],
        }
        
        suggested = default_symbols.get(market, [])
        print(f"\nSuggested symbols for {market}:")
        print(f"  {', '.join(suggested)}")
        
        response = input("\nEnter symbols (comma-separated) or press Enter for suggested: ").strip()
        
        if response:
            symbols = [s.strip().upper() for s in response.split(",")]
        else:
            symbols = suggested
        
        print(f"\n✓ Selected: {', '.join(symbols)}")
        return symbols
    
    def prompt_capital(self) -> float:
        """Prompt user for starting capital."""
        print("\n" + "=" * 70)
        print("4️⃣  STARTING CAPITAL")
        print("=" * 70)
        print("\nEnter initial trading capital (USD)")
        print("Suggested ranges:")
        print("  - Paper/Backtest: $10,000 - $100,000")
        print("  - Live Trading:   $25,000+ (regulatory minimum)")
        
        while True:
            try:
                response = input("\nEnter capital ($): ").strip()
                capital = float(response.replace(",", ""))
                
                if capital < 1000:
                    print("⚠️  Warning: Very small capital may limit trading opportunities")
                    confirm = input("Continue? (y/n): ").strip().lower()
                    if confirm != "y":
                        continue
                
                print(f"\n✓ Starting Capital: ${capital:,.2f}")
                return capital
            
            except ValueError:
                print("Invalid input. Please enter a number.")
    
    def prompt_risk(self) -> float:
        """Prompt user for risk per trade."""
        print("\n" + "=" * 70)
        print("5️⃣  RISK PER TRADE")
        print("=" * 70)
        print("\nRisk management is critical. Select risk per trade:")
        print("  1) 0.5%  - Ultra conservative (low volatility)")
        print("  2) 1.0%  - Conservative (recommended for most)")
        print("  3) 2.0%  - Moderate (experienced traders)")
        print("  4) 5.0%  - Aggressive (very experienced, high risk)")
        print("  5) Custom")
        
        while True:
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice in self.RISK_PRESETS:
                risk = self.RISK_PRESETS[choice]
                print(f"\n✓ Risk per trade: {risk:.1f}%")
                return risk
            
            elif choice == "5":
                try:
                    risk = float(input("Enter custom risk percentage (0.1-10.0): ").strip())
                    if 0.1 <= risk <= 10.0:
                        print(f"\n✓ Risk per trade: {risk:.1f}%")
                        return risk
                    print("⚠️  Must be between 0.1% and 10.0%")
                except ValueError:
                    print("Invalid input.")
            else:
                print("Invalid choice.")
    
    def prompt_confirmation(self, config: Dict) -> bool:
        """Prompt user for final confirmation."""
        print("\n" + "=" * 70)
        print("6️⃣  CONFIRM CONFIGURATION")
        print("=" * 70)
        print("\nReview your settings:")
        print(f"\n  Mode:           {config['mode']}")
        print(f"  Markets:        {config['markets']}")
        print(f"  Symbols:        {', '.join(config['symbols'])}")
        print(f"  Capital:        ${config['capital']:,.2f}")
        print(f"  Risk/Trade:     {config['risk_per_trade']:.1f}%")
        print(f"  Max Loss/Trade: ${config['capital'] * config['risk_per_trade'] / 100:,.2f}")
        
        response = input("\nProceed with this configuration? (y/n): ").strip().lower()
        return response == "y"
    
    def save_config(self) -> Path:
        """Save configuration to JSON file."""
        self.config.update({
            "created_at": datetime.now().isoformat(),
            "created_by": "setup_wizard_v1",
        })
        
        try:
            with open(self.config_path, "w") as f:
                json.dump(self.config, f, indent=2)
            
            logger.info(f"✓ Configuration saved to {self.config_path}")
            print(f"\n✓ Configuration saved: {self.config_path}")
            
            return self.config_path
        
        except Exception as e:
            logger.error(f"✗ Failed to save config: {e}")
            raise
    
    def run(self) -> Dict:
        """
        Run interactive setup wizard.
        
        Returns:
            Configuration dictionary
        """
        print("\n")
        print("╔══════════════════════════════════════════════════════════════════════╗")
        print("║      INSTITUTIONAL TRADING BOT - SETUP WIZARD                       ║")
        print("║      Quantitative Trading Agent Configuration                       ║")
        print("╚══════════════════════════════════════════════════════════════════════╝")
        
        # Check for existing config
        if self.load_existing_config():
            return self.config
        
        # Run questionnaire
        self.config["mode"] = self.prompt_mode()
        self.config["markets"] = self.prompt_markets()
        self.config["symbols"] = self.prompt_symbols(self.config["markets"])
        self.config["capital"] = self.prompt_capital()
        self.config["risk_per_trade"] = self.prompt_risk()
        
        # Confirmation
        if not self.prompt_confirmation(self.config):
            print("\n✗ Configuration cancelled.")
            return None
        
        # Save
        self.save_config()
        
        print("\n" + "=" * 70)
        print("✓ SETUP COMPLETE")
        print("=" * 70)
        print(f"\nYou're ready to start trading! Running in {self.config['mode']} mode.")
        print(f"Run: python main.py --config {self.config_path}")
        
        return self.config


def main():
    """Main entry point for setup wizard."""
    wizard = SetupWizard()
    config = wizard.run()
    
    if config:
        print("\n✓ Ready to launch trading agent")
        return 0
    else:
        print("\n✗ Setup cancelled")
        return 1


if __name__ == "__main__":
    exit(main())

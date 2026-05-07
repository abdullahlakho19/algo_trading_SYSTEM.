# ============================================================================
# Algorithmic TRADING AGENT
# config.py - Centralized Configuration Management
# ============================================================================

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from dotenv import load_dotenv
import logging

# ============================================================================
# SETUP LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION CLASS
# ============================================================================

class Config:
    """
    Centralized configuration manager for the Institutional Trading Agent.
    
    Loads configuration from .env file and validates all critical parameters
    before allowing the system to boot. Implements the fail-fast principle:
    any missing critical keys will raise exceptions immediately.
    """
    
    # ========================================================================
    # INITIALIZATION
    # ========================================================================
    
    def __init__(self, env_path: Optional[str] = None):
        """
        Initialize configuration from .env file.
        
        Args:
            env_path: Optional path to .env file. Defaults to project root.
            
        Raises:
            FileNotFoundError: If .env file not found
            ValueError: If critical API keys are missing
        """
        self.env_path = env_path or self._find_env_file()
        
        if not self.env_path.exists():
            logger.error(f"❌ .env file not found at {self.env_path}")
            logger.info("   → Run: cp .env.example .env")
            logger.info("   → Then fill in your API keys")
            raise FileNotFoundError(f".env file not found: {self.env_path}")
        
        # Load environment variables
        load_dotenv(self.env_path)
        logger.info(f"✓ Configuration loaded from {self.env_path}")
        
        # Validate and set configuration
        self._validate_and_set_config()
    
    @staticmethod
    def _find_env_file() -> Path:
        """Find .env file in project root."""
        current = Path(__file__).parent
        while current != current.parent:
            if (current / ".env").exists():
                return current / ".env"
            current = current.parent
        return Path.cwd() / ".env"
    
    # ========================================================================
    # CONFIGURATION VALIDATION & SETUP
    # ========================================================================
    
    def _validate_and_set_config(self) -> None:
        """Validate all critical configuration parameters."""
        logger.info("=" * 70)
        logger.info("VALIDATING SYSTEM CONFIGURATION [STOCKS & FOREX ONLY]")
        logger.info("=" * 70)
        
        # Alpaca Configuration (Stocks & Forex)
        self._validate_alpaca()
        
        # Trading Mode
        self._validate_trading_mode()
        
        # Risk Management Parameters
        self._validate_risk_parameters()
        
        # ML Model Parameters
        self._validate_ml_parameters()
        
        # Sentiment & News (optional but recommended)
        self._validate_sentiment_apis()
        
        # Data Source Configuration
        self._validate_data_sources()
        
        # Database Configuration
        self._validate_database()
        
        logger.info("=" * 70)
        logger.info("✓ ALL VALIDATION CHECKS PASSED")
        logger.info("=" * 70)
    
    def _validate_alpaca(self) -> None:
        """Validate Alpaca API credentials and settings."""
        logger.info("\n[1/7] Validating Alpaca (Stocks/Forex)...")
        
        self.ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
        self.ALPACA_SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
        self.ALPACA_BASE_URL = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
        paper_mode = os.getenv("ALPACA_PAPER_MODE", "true").lower()
        self.ALPACA_PAPER_MODE = paper_mode in ("true", "1", "yes")
        
        if not self.ALPACA_API_KEY or not self.ALPACA_SECRET_KEY:
            raise ValueError(
                "❌ ALPACA_API_KEY and ALPACA_SECRET_KEY are required in .env\n"
                "   Get them at: https://app.alpaca.markets"
            )
        
        logger.info(f"   ✓ Alpaca API Key: {self.ALPACA_API_KEY[:10]}...")
        logger.info(f"   ✓ Alpaca Base URL: {self.ALPACA_BASE_URL}")
        logger.info(f"   ✓ Paper Mode: {self.ALPACA_PAPER_MODE}")
    
    def _validate_binance(self) -> None:
        """Binance removed — Stocks and Forex only via Alpaca."""
        logger.info("\n[2/7] Crypto removed — Stocks and Forex via Alpaca only")
    
    def _validate_trading_mode(self) -> None:
        """Validate trading mode (paper/backtest/train/live)."""
        logger.info("\n[3/7] Validating Trading Mode...")
        
        valid_modes = ("paper", "backtest", "train", "live")
        self.TRADING_MODE = os.getenv("TRADING_MODE", "paper").lower()
        
        if self.TRADING_MODE not in valid_modes:
            raise ValueError(
                f"❌ TRADING_MODE must be one of: {valid_modes}\n"
                f"   Got: {self.TRADING_MODE}"
            )
        
        logger.info(f"   ✓ Trading Mode: {self.TRADING_MODE.upper()}")
        
        if self.TRADING_MODE == "live" and self.ALPACA_PAPER_MODE:
            logger.warning(
                "   ⚠ WARNING: Live mode selected but ALPACA_PAPER_MODE=true\n"
                "              Update .env to use live API credentials"
            )
    
    def _validate_risk_parameters(self) -> None:
        """Validate risk management thresholds."""
        logger.info("\n[4/7] Validating Risk Management Parameters...")
        
        try:
            self.MAX_DAILY_LOSS_PERCENT = float(os.getenv("MAX_DAILY_LOSS_PERCENT", "2.0"))
            self.MAX_WEEKLY_LOSS_PERCENT = float(os.getenv("MAX_WEEKLY_LOSS_PERCENT", "5.0"))
            self.MAX_DRAWDOWN_PERCENT = float(os.getenv("MAX_DRAWDOWN_PERCENT", "10.0"))
            self.MAX_CORRELATION_THRESHOLD = float(os.getenv("MAX_CORRELATION_THRESHOLD", "0.7"))
            self.POSITION_SIZE_PERCENT = float(os.getenv("POSITION_SIZE_PERCENT", "1.0"))
            self.MAX_RISK_PER_TRADE = float(os.getenv("MAX_RISK_PER_TRADE", "0.01"))
            
            # Validate ranges
            if not (0 < self.MAX_DAILY_LOSS_PERCENT <= 10):
                raise ValueError(f"MAX_DAILY_LOSS_PERCENT must be 0-10%, got {self.MAX_DAILY_LOSS_PERCENT}")
            
            if not (0 < self.MAX_WEEKLY_LOSS_PERCENT <= 20):
                raise ValueError(f"MAX_WEEKLY_LOSS_PERCENT must be 0-20%, got {self.MAX_WEEKLY_LOSS_PERCENT}")
            
            if not (0 < self.MAX_DRAWDOWN_PERCENT <= 30):
                raise ValueError(f"MAX_DRAWDOWN_PERCENT must be 0-30%, got {self.MAX_DRAWDOWN_PERCENT}")
            
            if not (0 < self.MAX_CORRELATION_THRESHOLD <= 1.0):
                raise ValueError(f"MAX_CORRELATION_THRESHOLD must be 0-1.0, got {self.MAX_CORRELATION_THRESHOLD}")
            
            if not (0 < self.POSITION_SIZE_PERCENT <= 5.0):
                raise ValueError(f"POSITION_SIZE_PERCENT must be 0-5%, got {self.POSITION_SIZE_PERCENT}")
            
            if not (0 < self.MAX_RISK_PER_TRADE <= 0.1):
                raise ValueError(f"MAX_RISK_PER_TRADE must be 0-0.1, got {self.MAX_RISK_PER_TRADE}")
            
            logger.info(f"   ✓ Max Daily Loss: {self.MAX_DAILY_LOSS_PERCENT}%")
            logger.info(f"   ✓ Max Weekly Loss: {self.MAX_WEEKLY_LOSS_PERCENT}%")
            logger.info(f"   ✓ Max Drawdown: {self.MAX_DRAWDOWN_PERCENT}%")
            logger.info(f"   ✓ Max Correlation: {self.MAX_CORRELATION_THRESHOLD}")
            logger.info(f"   ✓ Position Size: {self.POSITION_SIZE_PERCENT}%")
            logger.info(f"   ✓ Max Risk Per Trade: {self.MAX_RISK_PER_TRADE * 100:.2f}%")
            
        except ValueError as e:
            raise ValueError(f"❌ Invalid risk parameter: {e}")
    
    def _validate_ml_parameters(self) -> None:
        """Validate ML/AI model parameters [STRICT INSTITUTIONAL STANDARDS]."""
        logger.info("\n[5/7] Validating ML/AI Parameters [STRICT INSTITUTIONAL]...")
        
        try:
            self.MIN_PROBABILITY_THRESHOLD = float(os.getenv("MIN_PROBABILITY_THRESHOLD", "0.78"))
            self.MIN_CONFLUENCE_SIGNALS = int(os.getenv("MIN_CONFLUENCE_SIGNALS", "4"))
            self.MIN_RISK_REWARD_RATIO = float(os.getenv("MIN_RISK_REWARD_RATIO", "2.0"))
            self.MIN_SHARPE_RATIO = float(os.getenv("MIN_SHARPE_RATIO", "1.0"))
            self.RETRAIN_INTERVAL_HOURS = int(os.getenv("RETRAIN_INTERVAL_HOURS", "336"))
            self.MIN_TRAIN_SAMPLES = int(os.getenv("MIN_TRAIN_SAMPLES", "1000"))
            self.MODEL_ACCURACY_THRESHOLD = float(os.getenv("MODEL_ACCURACY_THRESHOLD", "0.65"))
            self.SIGNAL_DECAY_HOURS = int(os.getenv("SIGNAL_DECAY_HOURS", "24"))
            
            # Validate ranges
            if not (0.5 <= self.MIN_PROBABILITY_THRESHOLD <= 0.99):
                raise ValueError(f"MIN_PROBABILITY_THRESHOLD must be 0.5-0.99, got {self.MIN_PROBABILITY_THRESHOLD}")
            
            if not (2 <= self.MIN_CONFLUENCE_SIGNALS <= 5):
                raise ValueError(f"MIN_CONFLUENCE_SIGNALS must be 2-5, got {self.MIN_CONFLUENCE_SIGNALS}")
            
            if not (1.0 <= self.MIN_RISK_REWARD_RATIO <= 3.0):
                raise ValueError(f"MIN_RISK_REWARD_RATIO must be 1.0-3.0, got {self.MIN_RISK_REWARD_RATIO}")
            
            if self.RETRAIN_INTERVAL_HOURS < 24:
                raise ValueError(f"RETRAIN_INTERVAL_HOURS must be >= 24, got {self.RETRAIN_INTERVAL_HOURS}")
            
            if self.MIN_TRAIN_SAMPLES < 500:
                raise ValueError(f"MIN_TRAIN_SAMPLES must be >= 500, got {self.MIN_TRAIN_SAMPLES}")
            
            logger.info(f"   ✓ Min Probability Threshold: {self.MIN_PROBABILITY_THRESHOLD * 100:.1f}%")
            logger.info(f"   ✓ Min Confluence Signals: {self.MIN_CONFLUENCE_SIGNALS}")
            logger.info(f"   ✓ Min Risk/Reward Ratio: {self.MIN_RISK_REWARD_RATIO}:1")
            logger.info(f"   ✓ Retrain Interval: {self.RETRAIN_INTERVAL_HOURS} hours ({self.RETRAIN_INTERVAL_HOURS/24:.0f} days)")
            logger.info(f"   ✓ Min Training Samples: {self.MIN_TRAIN_SAMPLES}")
            
        except ValueError as e:
            raise ValueError(f"❌ Invalid ML parameter: {e}")
    
    def _validate_sentiment_apis(self) -> None:
        """Validate sentiment and news API keys (optional but recommended)."""
        logger.info("\n[6/7] Validating Sentiment & News APIs...")
        
        self.FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
        self.ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY")
        self.NEWS_API_KEY = os.getenv("NEWS_API_KEY")
        self.REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
        self.REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
        self.REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT", "TradingAgent/1.0")
        
        configured = []
        if self.FINNHUB_API_KEY:
            configured.append("Finnhub")
        if self.ALPHA_VANTAGE_API_KEY:
            configured.append("Alpha Vantage")
        if self.NEWS_API_KEY:
            configured.append("NewsAPI")
        if self.REDDIT_CLIENT_ID and self.REDDIT_CLIENT_SECRET:
            configured.append("Reddit (PRAW)")
        
        if configured:
            logger.info(f"   ✓ Sentiment APIs configured: {', '.join(configured)}")
        else:
            logger.warning("   ⚠ No sentiment APIs configured (optional)")
            logger.info("   → Get free keys at: Finnhub, Alpha Vantage, NewsAPI, Reddit")
    
    def _validate_data_sources(self) -> None:
        """Validate data source configuration (yFinance primary, Alpaca for execution)."""
        logger.info("\n[6/7] Validating Data Sources...")
        
        # Data source priority chain: yFinance (historical) | Alpaca (execution only)
        self.DATA_PRIMARY_SOURCE = os.getenv("DATA_PRIMARY_SOURCE", "yfinance").lower()
        self.DATA_EXECUTION_SOURCE = os.getenv("DATA_EXECUTION_SOURCE", "alpaca").lower()
        
        valid_sources = ("yfinance", "alpaca")
        
        if self.DATA_PRIMARY_SOURCE not in valid_sources:
            raise ValueError(f"Invalid DATA_PRIMARY_SOURCE: {self.DATA_PRIMARY_SOURCE}")
        
        if self.DATA_EXECUTION_SOURCE not in valid_sources:
            raise ValueError(f"Invalid DATA_EXECUTION_SOURCE: {self.DATA_EXECUTION_SOURCE}")
        
        if self.DATA_LAST_RESORT not in valid_sources:
            raise ValueError(f"Invalid DATA_LAST_RESORT: {self.DATA_LAST_RESORT}")
        
        logger.info(f"   ✓ Primary Data Source: {self.DATA_PRIMARY_SOURCE.upper()}")
        logger.info(f"   ✓ Fallback Data Source: {self.DATA_FALLBACK_SOURCE.upper()}")
        logger.info(f"   ✓ Last Resort Data Source: {self.DATA_LAST_RESORT.upper()}")
    
    def _validate_database(self) -> None:
        """Validate database configuration."""
        logger.info("\n[7/7] Validating Database Configuration...")
        
        self.DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///data/trading_agent.db")
        self.INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", "100000.00"))
        
        logger.info(f"   ✓ Database: {self.DATABASE_URL}")
        logger.info(f"   ✓ Initial Capital: ${self.INITIAL_CAPITAL:,.2f}")
    
    # ========================================================================
    # UTILITY METHODS
    # ========================================================================
    
    def to_dict(self) -> Dict[str, Any]:
        """Export all configuration as dictionary."""
        return {k: v for k, v in self.__dict__.items() 
                if not k.startswith('_') and k != 'env_path'}
    
    def get_api_status(self) -> Dict[str, bool]:
        """Get status of API key availability."""
        return {
            "alpaca": bool(self.ALPACA_API_KEY),
            "finnhub": bool(self.FINNHUB_API_KEY),
            "alpha_vantage": bool(self.ALPHA_VANTAGE_API_KEY),
            "news_api": bool(self.NEWS_API_KEY),
            "reddit": bool(self.REDDIT_CLIENT_ID and self.REDDIT_CLIENT_SECRET),
        }
    
    def print_summary(self) -> None:
        """Print configuration summary to console."""
        print("\n" + "=" * 70)
        print("SYSTEM CONFIGURATION SUMMARY")
        print("=" * 70)
        print(f"Trading Mode:        {self.TRADING_MODE.upper()}")
        print(f"Database:            {self.DATABASE_URL}")
        print(f"Initial Capital:     ${self.INITIAL_CAPITAL:,.2f}")
        print(f"\nRisk Thresholds:")
        print(f"  Daily Loss Limit:  {self.MAX_DAILY_LOSS_PERCENT}%")
        print(f"  Weekly Loss Limit: {self.MAX_WEEKLY_LOSS_PERCENT}%")
        print(f"  Max Drawdown:      {self.MAX_DRAWDOWN_PERCENT}%")
        print(f"  Risk Per Trade:    {self.MAX_RISK_PER_TRADE * 100:.2f}%")
        print(f"\nML Settings [STRICT]:")
        print(f"  Min Probability:   {self.MIN_PROBABILITY_THRESHOLD * 100:.1f}%")
        print(f"  Min Confluence:    {self.MIN_CONFLUENCE_SIGNALS} signals")
        print(f"  Risk/Reward Ratio: {self.MIN_RISK_REWARD_RATIO}:1")
        print(f"  Retrain Interval:  {self.RETRAIN_INTERVAL_HOURS}h ({self.RETRAIN_INTERVAL_HOURS/24:.0f}d)")
        print(f"  Min Train Samples: {self.MIN_TRAIN_SAMPLES}")
        print(f"\nAPI Status: {self.get_api_status()}")
        print("=" * 70 + "\n")


# ============================================================================
# LAZY SINGLETON INITIALIZATION
# ============================================================================

_config_instance: Optional[Config] = None


def get_config() -> Config:
    """
    Get or create the configuration singleton.
    
    Returns:
        Config: Validated configuration object
        
    Raises:
        FileNotFoundError: If .env file not found
        ValueError: If critical configuration is missing
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = Config()
    
    return _config_instance


# ============================================================================
# STANDALONE TESTING/VALIDATION
# ============================================================================

if __name__ == "__main__":
    try:
        config = get_config()
        config.print_summary()
        print("✓ Configuration validated successfully!")
        sys.exit(0)
    except (FileNotFoundError, ValueError) as e:
        logger.error(f"✗ Configuration validation failed: {e}")
        sys.exit(1)

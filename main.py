# ============================================================================
# MAIN.PY - Master Entry Point with Backtest Mode
# Candle-by-candle backtesting: Layer 2 → Layer 3 → Layer 4 → Paper Simulator
# ============================================================================

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

import pandas as pd
import numpy as np
import joblib

from data_feeds.yfinance_feed import yfinance_feed
from quant.probability_scorer import ProbabilityScorer
from execution.paper_simulator import PaperSimulator
from intelligence.momentum import MomentumAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

MODELS_DIR = Path(__file__).parent / "models"
PROBABILITY_THRESHOLD = 0.78
SYMBOLS = ["AAPL", "GOOGL", "MSFT", "EURUSD=X"]
BACKTEST_DAYS = 365


# ============================================================================
# BACKTEST ENGINE
# ============================================================================

class BacktestEngine:
    """
    Candle-by-candle backtest engine.
    
    Flow for each candle:
    1. Layer 2: Intelligence (momentum, structure)
    2. Layer 3: ML Ensemble (signal model, regime model)
    3. Layer 4: Probability Scorer (final confidence)
    4. Paper Simulator (track entry, SL, TP)
    """
    
    def __init__(self, symbols: List[str], days: int = 365, initial_capital: float = 100000.0):
        """
        Initialize backtest engine.
        
        Args:
            symbols: List of symbols to backtest
            days: Lookback days
            initial_capital: Starting account balance
        """
        self.symbols = symbols
        self.days = days
        self.initial_capital = initial_capital
        
        # Load models
        self.signal_model = None
        self.signal_scaler = None
        self.regime_model = None
        self.regime_scaler = None
        self._load_models()
        
        # Initialize components
        self.probability_scorer = ProbabilityScorer()
        self.simulator = PaperSimulator(initial_capital=initial_capital)
        self.momentum = MomentumAnalyzer()
        
        # Backtest state
        self.candle_count = 0
        self.signal_count = 0
        self.trade_count = 0
        self.backtest_results = []
        
        logger.info(f"✓ BacktestEngine initialized: {len(symbols)} symbols, {days}d lookback")
    
    def _load_models(self) -> None:
        """Load trained ML models from disk."""
        try:
            signal_model_path = MODELS_DIR / "signal_model.pkl"
            signal_scaler_path = MODELS_DIR / "signal_scaler.pkl"
            regime_model_path = MODELS_DIR / "regime_model.pkl"
            regime_scaler_path = MODELS_DIR / "regime_scaler.pkl"
            
            if signal_model_path.exists():
                self.signal_model = joblib.load(signal_model_path)
                self.signal_scaler = joblib.load(signal_scaler_path)
                logger.info("✓ Signal model loaded")
            else:
                logger.warning(f"Signal model not found at {signal_model_path}")
            
            if regime_model_path.exists():
                self.regime_model = joblib.load(regime_model_path)
                self.regime_scaler = joblib.load(regime_scaler_path)
                logger.info("✓ Regime model loaded")
            else:
                logger.warning(f"Regime model not found at {regime_model_path}")
        
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def _extract_features(self, df: pd.DataFrame, lookback: int = 20) -> Optional[np.ndarray]:
        """
        Extract features for ML models.
        
        Returns array with: RSI, MACD, MACD_signal, MACD_hist, ATR, ATR_pct, ADX, vol_ratio, returns
        """
        if len(df) < lookback:
            return None
        
        try:
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # RSI (14-period)
            delta = np.diff(close, prepend=close[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14).mean().values[-1]
            avg_loss = pd.Series(loss).rolling(14).mean().values[-1]
            rs = avg_gain / (avg_loss + 1e-8)
            rsi = 100 - (100 / (1 + rs))
            
            # MACD
            ema_12 = pd.Series(close).ewm(span=12).mean().values[-1]
            ema_26 = pd.Series(close).ewm(span=26).mean().values[-1]
            macd = ema_12 - ema_26
            signal = pd.Series(pd.Series(close).ewm(span=12).mean() - pd.Series(close).ewm(span=26).mean()).ewm(span=9).mean().values[-1]
            macd_hist = macd - signal
            
            # ATR
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            atr = pd.Series(tr).rolling(14).mean().values[-1]
            atr_pct = atr / close[-1]
            
            # ADX (simplified)
            adx = 50.0  # Placeholder
            
            # Volume ratio
            vol_sma = pd.Series(volume).rolling(20).mean().values[-1]
            vol_ratio = volume[-1] / (vol_sma + 1e-8)
            
            # Returns
            returns = (close[-1] - close[-2]) / close[-2] if len(close) > 1 else 0
            
            features = np.array([rsi, macd, signal, macd_hist, atr, atr_pct, adx, vol_ratio, returns])
            return features
        
        except Exception as e:
            logger.warning(f"Feature extraction error: {e}")
            return None
    
    def _predict_signal(self, features: np.ndarray) -> Dict:
        """
        Predict signal using trained models.
        
        Returns dict with 'signal' (-1/0/1) and 'regime' (0/1)
        """
        result = {'signal': 0, 'regime': 0, 'signal_prob': 0.5, 'regime_prob': 0.5}
        
        if self.signal_model is None or self.regime_model is None:
            return result
        
        try:
            # Signal prediction
            features_scaled = self.signal_scaler.transform(features.reshape(1, -1))
            signal_pred = self.signal_model.predict(features_scaled)[0]
            signal_proba = self.signal_model.predict_proba(features_scaled)[0]
            
            result['signal'] = int(signal_pred)
            result['signal_prob'] = float(np.max(signal_proba))
            
            # Regime prediction
            regime_scaled = self.regime_scaler.transform(features.reshape(1, -1))
            regime_pred = self.regime_model.predict(regime_scaled)[0]
            regime_proba = self.regime_model.predict_proba(regime_scaled)[0]
            
            result['regime'] = int(regime_pred)
            result['regime_prob'] = float(np.max(regime_proba))
        
        except Exception as e:
            logger.warning(f"Prediction error: {e}")
        
        return result
    
    async def process_candle(self, symbol: str, candle: pd.Series, candle_idx: int, df_full: pd.DataFrame) -> None:
        """
        Process single candle through 4-gate pipeline.
        
        Args:
            symbol: Trading symbol
            candle: OHLCV row
            candle_idx: Row index
            df_full: Full DataFrame for context
        """
        self.candle_count += 1
        
        # Layer 2: Intelligence - Calculate momentum/structure
        df_lookback = df_full.iloc[max(0, candle_idx-50):candle_idx+1]
        momentum = self.momentum.calculate_momentum(df_lookback, period=12)
        
        # Layer 3: ML Ensemble - Get signal and regime
        features = self._extract_features(df_lookback)
        if features is None:
            return
        
        predictions = self._predict_signal(features)
        signal = predictions['signal']
        regime = predictions['regime']
        
        if signal == 0:  # No signal
            return
        
        self.signal_count += 1
        
        # Layer 4: Probability Scorer - Calculate final confidence
        try:
            probability_score = self.probability_scorer.score_setup(
                ai_signal=signal,
                ai_confidence=predictions['signal_prob'],
                regime=regime,
                momentum=momentum,
                price=candle['close'],
                symbol=symbol
            )
        except Exception as e:
            logger.warning(f"Probability scoring error: {e}")
            return
        
        # Check threshold
        if probability_score < PROBABILITY_THRESHOLD:
            return
        
        # Passed 4 gates! Submit to paper simulator
        self.trade_count += 1
        
        logger.info(
            f"[SIGNAL] {symbol} | candle {self.candle_count} | "
            f"signal={signal} | regime={regime} | prob={probability_score:.2%} | "
            f"price=${candle['close']:.2f}"
        )
        
        # Calculate SL and TP
        atr = (candle['high'] - candle['low'])
        sl_price = candle['close'] - (atr * 1.5) if signal == 1 else candle['close'] + (atr * 1.5)
        tp_price = candle['close'] + (atr * 3.0) if signal == 1 else candle['close'] - (atr * 3.0)
        
        # Submit to simulator
        side = 'BUY' if signal == 1 else 'SELL'
        quantity = 1.0  # Simplified: 1 unit per trade
        
        order = await self.simulator.submit_order(
            order_id=f"{symbol}_{self.trade_count}_{int(candle_idx)}",
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=candle['close'],
            order_type='MARKET'
        )
        
        self.backtest_results.append({
            'timestamp': candle_idx,
            'symbol': symbol,
            'side': side,
            'price': candle['close'],
            'sl': sl_price,
            'tp': tp_price,
            'probability': probability_score,
            'order_id': order.order_id
        })
    
    async def run_backtest(self) -> Dict:
        """
        Run candle-by-candle backtest.
        
        Returns:
            Dictionary with backtest results and statistics
        """
        logger.info("=" * 80)
        logger.info("BACKTEST MODE: LOADING HISTORICAL DATA")
        logger.info("=" * 80)
        
        for symbol in self.symbols:
            logger.info(f"\nProcessing {symbol} ({self.days} days)...")
            
            try:
                # Download data
                df = yfinance_feed.get_bars(
                    symbol=symbol,
                    timeframe="1h",
                    lookback_days=self.days
                )
                
                if df is None or df.empty:
                    logger.warning(f"No data for {symbol}")
                    continue
                
                df.columns = df.columns.str.lower()
                logger.info(f"✓ Loaded {len(df)} candles for {symbol}")
                
                # Process each candle
                for idx in range(len(df)):
                    candle = df.iloc[idx]
                    await self.process_candle(symbol, candle, idx, df)
            
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")
                continue
        
        # Final report
        return self._generate_report()
    
    def _generate_report(self) -> Dict:
        """Generate backtest summary report."""
        stats = self.simulator.portfolio_stats
        
        report = {
            'backtest_period_days': self.days,
            'symbols': self.symbols,
            'total_candles_processed': self.candle_count,
            'signals_generated': self.signal_count,
            'trades_executed': self.trade_count,
            'portfolio_stats': stats,
            'backtest_results': self.backtest_results,
            'timestamp': datetime.now().isoformat()
        }
        
        return report


class TradingAgent:
    """
    Master trading agent controller.
    
    Supports modes:
    - backtest: Historical candle-by-candle simulation
    - paper: Paper trading with live data
    - live: Live trading (requires verified credentials)
    """
    
    def __init__(self, config_path: Path = Path("agent_config.json")):
        """
        Initialize trading agent.
        
        Args:
            config_path: Path to agent_config.json
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        logger.info("[OK] TradingAgent initialized")
    
    def _load_config(self) -> Dict:
        """Load configuration from JSON file."""
        try:
            with open(self.config_path, "r") as f:
                config = json.load(f)
            logger.info(f"✓ Config loaded: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"✗ Config file not found: {self.config_path}")
            logger.info("Run: python run.py")
            sys.exit(1)
        except Exception as e:
            logger.error(f"✗ Failed to load config: {e}")
            sys.exit(1)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

async def main():
    """Main entry point with mode selection."""
    parser = argparse.ArgumentParser(
        description="Autonomous Trading Agent (7-Layer Architecture)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --mode backtest              # Run 365-day backtest
  python main.py --mode backtest --days 180   # Run 180-day backtest
  python main.py --mode backtest --symbols AAPL GOOGL  # Custom symbols
  python main.py --mode paper                 # Paper trading
  python main.py --mode live                  # Live trading (verify credentials!)
        """
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="backtest",
        choices=["backtest", "paper", "live"],
        help="Execution mode (default: backtest)"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data for backtest (default: 365)"
    )
    parser.add_argument(
        "--symbols",
        nargs='+',
        default=None,
        help="Symbols to backtest (default: AAPL GOOGL MSFT EURUSD=X)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=100000.0,
        help="Initial capital for paper/backtest (default: 100000)"
    )
    
    args = parser.parse_args()
    
    # ========================================================================
    # BACKTEST MODE
    # ========================================================================
    
    if args.mode == "backtest":
        logger.info("=" * 80)
        logger.info("BACKTEST MODE")
        logger.info("=" * 80)
        
        symbols = args.symbols or SYMBOLS
        days = args.days
        
        logger.info(f"Symbols: {symbols}")
        logger.info(f"Lookback: {days} days")
        logger.info(f"Initial Capital: ${args.capital:,.2f}")
        
        # Initialize backtest engine
        engine = BacktestEngine(
            symbols=symbols,
            days=days,
            initial_capital=args.capital
        )
        
        # Run backtest
        report = await engine.run_backtest()
        
        # Display results
        logger.info("\n" + "=" * 80)
        logger.info("BACKTEST REPORT")
        logger.info("=" * 80)
        logger.info(f"Candles Processed: {report['total_candles_processed']}")
        logger.info(f"Signals Generated: {report['signals_generated']}")
        logger.info(f"Trades Executed: {report['trades_executed']}")
        logger.info(f"Final Equity: ${report['portfolio_stats']['total_equity']:,.2f}")
        logger.info(f"Total P&L: ${report['portfolio_stats']['total_pnl']:,.2f}")
        logger.info(f"Return: {report['portfolio_stats']['return_percent']:.2f}%")
        logger.info("=" * 80)
        
        # Save report to file
        report_path = Path("data/reports") / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        logger.info(f"✓ Report saved to {report_path}")
        
        return 0
    
    # ========================================================================
    # PAPER TRADING MODE
    # ========================================================================
    
    elif args.mode == "paper":
        logger.info("=" * 80)
        logger.info("PAPER TRADING MODE")
        logger.info("=" * 80)
        logger.info("Not implemented yet")
        return 1
    
    # ========================================================================
    # LIVE TRADING MODE
    # ========================================================================
    
    elif args.mode == "live":
        logger.info("=" * 80)
        logger.info("LIVE TRADING MODE")
        logger.info("=" * 80)
        logger.error("Live mode requires verified credentials and manual approval")
        return 1


if __name__ == '__main__':
    import asyncio
    sys.exit(asyncio.run(main()))

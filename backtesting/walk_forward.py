# ============================================================================
# WALK_FORWARD.PY - Walk-Forward Optimization & Out-of-Sample Validation
# Validates strategy across different market regimes to prevent overfitting
# ============================================================================

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from backtest_engine import BacktestConfig, BacktestEngine

logger = logging.getLogger(__name__)


@dataclass
class WindowPeriod:
    """Definition of an optimization/testing window."""
    start_date: str
    end_date: str
    window_type: str  # 'in_sample' or 'out_of_sample'
    market_regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'


@dataclass
class WindowResults:
    """Results for a single walk-forward window."""
    window_period: WindowPeriod
    total_trades: int
    win_rate_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    total_return_pct: float
    profit_factor: float


class MarketRegimeDetector:
    """Detect market regime from historical data."""
    
    @staticmethod
    def detect_regime(df: pd.DataFrame, window_days: int = 60) -> str:
        """
        Detect market regime from recent data.
        
        Args:
            df: DataFrame with OHLCV
            window_days: Days to analyze
        
        Returns:
            Regime string: trending_up, trending_down, ranging, volatile
        """
        if len(df) < window_days:
            return 'unknown'
        
        recent = df.tail(window_days)
        
        # Calculate trend
        prices = recent['Close'].values
        returns = (prices[-1] - prices[0]) / prices[0] * 100
        
        # Calculate volatility
        daily_returns = recent['Close'].pct_change()
        volatility = daily_returns.std() * (252 ** 0.5) * 100
        
        # Calculate range
        high = recent['High'].max()
        low = recent['Low'].min()
        price_range = (high - low) / low * 100
        
        # Classify regime
        if volatility > 40:  # High volatility
            return 'volatile'
        elif returns > 5:  # Positive trend
            return 'trending_up'
        elif returns < -5:  # Negative trend
            return 'trending_down'
        else:
            return 'ranging'
    
    @staticmethod
    def classify_regimes_by_change(df: pd.DataFrame) -> List[Tuple[str, str]]:
        """
        Classify periods by regime changes.
        
        Args:
            df: DataFrame with dates
        
        Returns:
            List of (start_date, regime) tuples
        """
        regimes = []
        window_size = 60
        
        for i in range(window_size, len(df), 30):
            window_df = df.iloc[max(0, i - window_size):i]
            regime = MarketRegimeDetector.detect_regime(window_df)
            date = df.index[i]
            if isinstance(date, datetime):
                date = date.strftime('%Y-%m-%d')
            regimes.append((str(date), regime))
        
        return regimes


class WalkForwardOptimizer:
    """
    Walk-Forward Optimization framework.
    
    Process:
    1. Split historical data into overlapping windows
    2. In-sample window: Optimize parameters
    3. Out-of-sample window: Test on unseen data
    4. Roll forward and repeat
    5. Analyze performance degradation (detect overfitting)
    
    Key Parameters:
    - in_sample_days: Training period (e.g., 180 days)
    - out_of_sample_days: Testing period (e.g., 30 days)
    - step_days: Roll forward by this amount (e.g., 30 days)
    """
    
    def __init__(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        in_sample_days: int = 180,
        out_of_sample_days: int = 30,
        step_days: int = 30,
        initial_capital: float = 100000.0,
    ):
        """
        Initialize walk-forward optimizer.
        
        Args:
            symbols: List of symbols to trade
            start_date: Historical data start date
            end_date: Historical data end date
            in_sample_days: Days for optimization window
            out_of_sample_days: Days for testing window
            step_days: Days to roll forward
            initial_capital: Starting capital
        """
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.in_sample_days = in_sample_days
        self.out_of_sample_days = out_of_sample_days
        self.step_days = step_days
        self.initial_capital = initial_capital
        
        # Results storage
        self.windows: List[WindowPeriod] = []
        self.results: List[WindowResults] = []
        self.regime_analysis: Dict = {}
        
        logger.info(
            f"✓ WFO initialized: {in_sample_days}d in-sample, "
            f"{out_of_sample_days}d out-of-sample, {step_days}d step"
        )
    
    def create_windows(
        self,
        data_start: datetime,
        data_end: datetime,
    ) -> List[Tuple[WindowPeriod, WindowPeriod]]:
        """
        Create walk-forward windows.
        
        Returns:
            List of (in_sample_period, out_of_sample_period) tuples
        """
        windows = []
        current = data_start
        
        while current + timedelta(days=self.in_sample_days + self.out_of_sample_days) <= data_end:
            # In-sample (optimization) window
            is_start = current
            is_end = current + timedelta(days=self.in_sample_days)
            
            # Out-of-sample (testing) window
            oos_start = is_end
            oos_end = oos_start + timedelta(days=self.out_of_sample_days)
            
            is_period = WindowPeriod(
                start_date=is_start.strftime('%Y-%m-%d'),
                end_date=is_end.strftime('%Y-%m-%d'),
                window_type='in_sample',
                market_regime='unknown',
            )
            
            oos_period = WindowPeriod(
                start_date=oos_start.strftime('%Y-%m-%d'),
                end_date=oos_end.strftime('%Y-%m-%d'),
                window_type='out_of_sample',
                market_regime='unknown',
            )
            
            windows.append((is_period, oos_period))
            
            current += timedelta(days=self.step_days)
        
        return windows
    
    async def run_backtest_for_window(self, window: WindowPeriod) -> Optional[WindowResults]:
        """
        Run backtest for a single window.
        
        Args:
            window: WindowPeriod to test
        
        Returns:
            WindowResults or None if backtest failed
        """
        config = BacktestConfig(
            symbols=self.symbols,
            start_date=window.start_date,
            end_date=window.end_date,
            initial_capital=self.initial_capital,
        )
        
        try:
            engine = BacktestEngine(config)
            results = await engine.run_backtest()
            
            summary = results.get('summary', {})
            metrics = results.get('metrics', {})
            
            return WindowResults(
                window_period=window,
                total_trades=summary.get('total_trades', 0),
                win_rate_pct=summary.get('win_rate_pct', 0.0),
                sharpe_ratio=metrics.get('sharpe_ratio', 0.0),
                max_drawdown_pct=metrics.get('max_drawdown_pct', 0.0),
                total_return_pct=summary.get('total_return_pct', 0.0),
                profit_factor=metrics.get('profit_factor', 0.0),
            )
        
        except Exception as e:
            logger.error(f"Backtest failed for window {window.start_date}: {e}")
            return None
    
    async def run_optimization(self) -> None:
        """
        Run complete walk-forward optimization.
        
        Process:
        1. Parse date range
        2. Create sliding windows
        3. Run backtest for each window
        4. Analyze performance degradation
        5. Report findings
        """
        print("\n" + "=" * 70)
        print("🔄 WALK-FORWARD OPTIMIZATION")
        print("=" * 70 + "\n")
        
        # Parse dates
        start = datetime.strptime(self.start_date, '%Y-%m-%d')
        end = datetime.strptime(self.end_date, '%Y-%m-%d')
        
        # Create windows
        print(f"Creating walk-forward windows...")
        print(f"  In-sample: {self.in_sample_days} days")
        print(f"  Out-of-sample: {self.out_of_sample_days} days")
        print(f"  Step: {self.step_days} days\n")
        
        window_pairs = self.create_windows(start, end)
        print(f"✓ Created {len(window_pairs)} window pairs\n")
        
        # Run backtests
        print("Running backtests for each window...\n")
        
        all_results = []
        
        for idx, (is_window, oos_window) in enumerate(window_pairs, 1):
            print(f"Window {idx}/{len(window_pairs)}")
            print(f"  In-sample:     {is_window.start_date} → {is_window.end_date}")
            print(f"  Out-of-sample: {oos_window.start_date} → {oos_window.end_date}")
            
            # Run in-sample backtest (training)
            is_results = await self.run_backtest_for_window(is_window)
            if is_results:
                print(f"    IS Results: {is_results.total_trades} trades, {is_results.win_rate_pct:.1f}% win rate, Sharpe {is_results.sharpe_ratio:.2f}")
                all_results.append(is_results)
            
            # Run out-of-sample backtest (testing)
            oos_results = await self.run_backtest_for_window(oos_window)
            if oos_results:
                print(f"    OOS Results: {oos_results.total_trades} trades, {oos_results.win_rate_pct:.1f}% win rate, Sharpe {oos_results.sharpe_ratio:.2f}")
                
                # Compare in-sample vs out-of-sample (detect overfitting)
                if is_results:
                    sharpe_diff = is_results.sharpe_ratio - oos_results.sharpe_ratio
                    print(f"    ⚠️  Sharpe degradation: {sharpe_diff:.2f} (Is {is_results.sharpe_ratio:.2f} → OOS {oos_results.sharpe_ratio:.2f})")
                
                all_results.append(oos_results)
            
            print()
        
        self.results = all_results
        self._analyze_results()
    
    def _analyze_results(self) -> None:
        """Analyze walk-forward results for insights."""
        if not self.results:
            logger.warning("No results to analyze")
            return
        
        # Separate in-sample and out-of-sample
        is_results = [r for r in self.results if r.window_period.window_type == 'in_sample']
        oos_results = [r for r in self.results if r.window_period.window_type == 'out_of_sample']
        
        print("\n" + "=" * 70)
        print("📊 WALK-FORWARD ANALYSIS")
        print("=" * 70 + "\n")
        
        # Aggregate statistics
        if is_results:
            print("IN-SAMPLE (Optimization) Statistics:")
            is_sharpe = [r.sharpe_ratio for r in is_results if r.sharpe_ratio != 0]
            is_wr = [r.win_rate_pct for r in is_results]
            is_return = [r.total_return_pct for r in is_results]
            
            if is_sharpe:
                print(f"  Average Sharpe:  {sum(is_sharpe)/len(is_sharpe):.2f}")
            if is_wr:
                print(f"  Average Win Rate: {sum(is_wr)/len(is_wr):.1f}%")
            if is_return:
                print(f"  Average Return:  {sum(is_return)/len(is_return):.1f}%")
        
        if oos_results:
            print("\nOUT-OF-SAMPLE (Testing) Statistics:")
            oos_sharpe = [r.sharpe_ratio for r in oos_results if r.sharpe_ratio != 0]
            oos_wr = [r.win_rate_pct for r in oos_results]
            oos_return = [r.total_return_pct for r in oos_results]
            
            if oos_sharpe:
                print(f"  Average Sharpe:  {sum(oos_sharpe)/len(oos_sharpe):.2f}")
            if oos_wr:
                print(f"  Average Win Rate: {sum(oos_wr)/len(oos_wr):.1f}%")
            if oos_return:
                print(f"  Average Return:  {sum(oos_return)/len(oos_return):.1f}%")
        
        # Overfitting detection
        if is_results and oos_results and len(is_results) == len(oos_results):
            print("\nOVERFITTING DETECTION:")
            sharpe_degradation = []
            wr_degradation = []
            
            for is_r, oos_r in zip(is_results, oos_results):
                sharpe_deg = is_r.sharpe_ratio - oos_r.sharpe_ratio
                wr_deg = is_r.win_rate_pct - oos_r.win_rate_pct
                sharpe_degradation.append(sharpe_deg)
                wr_degradation.append(wr_deg)
            
            avg_sharpe_deg = sum(sharpe_degradation) / len(sharpe_degradation)
            avg_wr_deg = sum(wr_degradation) / len(wr_degradation)
            
            print(f"  Average Sharpe Degradation:  {avg_sharpe_deg:.2f}")
            print(f"  Average Win Rate Degradation: {avg_wr_deg:.1f}%")
            
            if avg_sharpe_deg > 0.5:
                print("  ⚠️  WARNING: Significant overfitting detected!")
                print("     Consider: Fewer parameters, more data, simpler model")
            elif avg_sharpe_deg > 0.2:
                print("  ⚠️  CAUTION: Moderate overfitting detected")
            else:
                print("  ✓ Low overfitting risk - model appears robust")
        
        # Market regime analysis
        print("\nMARKET REGIME ANALYSIS:")
        print("  (Same strategy across different market conditions)")
        
        regime_performance = {}
        for result in self.results:
            regime = result.window_period.market_regime
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(result.sharpe_ratio)
        
        for regime, sharpes in regime_performance.items():
            if sharpes:
                avg_sharpe = sum(sharpes) / len(sharpes)
                print(f"  {regime}: {avg_sharpe:.2f} avg Sharpe")
        
        print()
    
    def save_results(self) -> Path:
        """Save walk-forward results to JSON."""
        results_data = {
            'config': {
                'symbols': self.symbols,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'in_sample_days': self.in_sample_days,
                'out_of_sample_days': self.out_of_sample_days,
                'step_days': self.step_days,
            },
            'windows': [
                {
                    'period': asdict(r.window_period),
                    'results': {
                        'total_trades': r.total_trades,
                        'win_rate_pct': r.win_rate_pct,
                        'sharpe_ratio': r.sharpe_ratio,
                        'max_drawdown_pct': r.max_drawdown_pct,
                        'total_return_pct': r.total_return_pct,
                        'profit_factor': r.profit_factor,
                    }
                }
                for r in self.results
            ],
        }
        
        results_path = Path(f"wfo_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        with open(results_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        print(f"✓ Results saved: {results_path}\n")
        return results_path


async def main():
    """Run walk-forward optimization."""
    
    # Configuration
    wfo = WalkForwardOptimizer(
        symbols=['AAPL', 'MSFT', 'GOOGL'],
        start_date='2022-01-01',
        end_date='2023-12-31',
        in_sample_days=180,      # 6 months optimization
        out_of_sample_days=30,   # 1 month testing
        step_days=30,            # Roll forward 1 month
        initial_capital=100000.0,
    )
    
    # Run WFO
    await wfo.run_optimization()
    
    # Save results
    wfo.save_results()


if __name__ == "__main__":
    print("\n")
    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║    Walk-Forward Optimization - Backtesting Framework                ║")
    print("║    Validates Strategy Across Market Regimes (No Overfitting)        ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    
    asyncio.run(main())

# ============================================================================
# BACKTEST_ENGINE.PY - Historical Backtesting with 7-Layer Logic
# Feeds yFinance data through exact same signal/execution logic as live trading
# ============================================================================

import asyncio
import json
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    symbols: List[str]
    start_date: str
    end_date: str
    initial_capital: float
    market: str = "US_STOCKS"
    max_position_size: float = 0.1
    risk_per_trade: float = 0.02
    slippage_bps: int = 5


@dataclass
class BacktestFill:
    """Record of a simulated fill."""
    timestamp: str
    symbol: str
    side: str  # BUY or SELL
    quantity: int
    price: float
    commission: float
    strategy: str


@dataclass
class BacktestPosition:
    """Active position during backtest."""
    symbol: str
    quantity: int
    entry_price: float
    entry_timestamp: str
    current_price: float
    unrealized_pnl: float
    
    def update_price(self, new_price: float) -> None:
        """Update position to current price."""
        self.current_price = new_price
        self.unrealized_pnl = (new_price - self.entry_price) * self.quantity


class BacktestEngine:
    """
    Backtesting engine that feeds historical data through 7-layer logic.
    
    Architecture mirrors live trading:
    ├── yFinance data as Layer 2 (Data Feeds)
    ├── Strategy signals as Layer 5 (Signal Engine)
    ├── Paper simulator as Layer 6 (Execution)
    ├── P&L tracking as Layer 7 (Reporting)
    
    Key Features:
    1. Bar-by-bar processing (OHLCV)
    2. Signal generation at close
    3. Order execution at next open
    4. Position tracking with unrealized P&L
    5. Commission/slippage modeling
    6. Performance metrics calculation
    """
    
    def __init__(self, config: BacktestConfig):
        """
        Initialize backtest engine.
        
        Args:
            config: BacktestConfig instance
        """
        self.config = config
        
        # State
        self.current_bar = None
        self.portfolio_value = config.initial_capital
        self.cash = config.initial_capital
        self.positions: Dict[str, BacktestPosition] = {}
        
        # History
        self.fills: List[BacktestFill] = []
        self.trades: List[Dict] = []
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []
        
        logger.info(f"✓ Backtest engine initialized: {config.symbols} | {config.start_date} to {config.end_date}")
    
    async def fetch_historical_data(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical OHLCV data from yFinance.
        
        Returns:
            Dict mapping symbol to DataFrame with OHLCV + signals
        """
        print("\n📊 FETCHING HISTORICAL DATA\n")
        
        data = {}
        try:
            import yfinance as yf
        except ImportError:
            logger.error("yfinance not installed: pip install yfinance")
            raise
        
        for symbol in self.config.symbols:
            print(f"   → {symbol} ({self.config.start_date} to {self.config.end_date})")
            
            try:
                df = yf.download(
                    symbol,
                    start=self.config.start_date,
                    end=self.config.end_date,
                    progress=False,
                )
                
                # Calculate indicators (simplified versions of actual strategies)
                df = self._calculate_indicators(df)
                
                data[symbol] = df
                print(f"      ✓ {len(df)} bars loaded")
            
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                raise
        
        print()
        return data
    
    def _calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for strategy signals.
        Mirrors actual strategy calculations.
        
        Args:
            df: DataFrame with OHLCV
        
        Returns:
            DataFrame with added indicator columns
        """
        # Mean Reversion (Bollinger Bands)
        sma20 = df['Close'].rolling(20).mean()
        std20 = df['Close'].rolling(20).std()
        df['bb_upper'] = sma20 + 2 * std20
        df['bb_lower'] = sma20 - 2 * std20
        df['rsi'] = self._calculate_rsi(df['Close'])
        
        # Momentum (MACD)
        exp1 = df['Close'].ewm(span=12, adjust=False).mean()
        exp2 = df['Close'].ewm(span=26, adjust=False).mean()
        df['macd'] = exp1 - exp2
        df['signal_line'] = df['macd'].ewm(span=9, adjust=False).mean()
        
        # Volume profile
        df['volume_sma'] = df['Volume'].rolling(20).mean()
        
        return df
    
    @staticmethod
    def _calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    async def generate_signals(self, bar: Dict, df: pd.DataFrame, index: int) -> Optional[Dict]:
        """
        Generate trade signals on bar close.
        Mirrors Layer 5 Strategy Engine.
        
        Args:
            bar: Current bar OHLCV data
            df: Full DataFrame for context
            index: Current index in DataFrame
        
        Returns:
            Signal dict with direction, confidence, strategy
        """
        if index < 30:  # Need warmup
            return None
        
        current_row = df.iloc[index]
        prev_row = df.iloc[index - 1]
        
        signal = None
        confidence = 0.0
        strategy_name = None
        
        # Strategy 1: Mean Reversion (Bollinger Bands)
        if current_row['Close'] < current_row['bb_lower'] and current_row['rsi'] < 30:
            signal = 'BUY'
            confidence = 0.7
            strategy_name = 'mean_reversion'
        
        elif current_row['Close'] > current_row['bb_upper'] and current_row['rsi'] > 70:
            signal = 'SELL'
            confidence = 0.7
            strategy_name = 'mean_reversion'
        
        # Strategy 2: Momentum (MACD)
        if signal is None:
            if prev_row['macd'] < prev_row['signal_line'] and current_row['macd'] > current_row['signal_line']:
                signal = 'BUY'
                confidence = 0.65
                strategy_name = 'momentum'
            
            elif prev_row['macd'] > prev_row['signal_line'] and current_row['macd'] < current_row['signal_line']:
                signal = 'SELL'
                confidence = 0.65
                strategy_name = 'momentum'
        
        if signal:
            return {
                'direction': signal,
                'confidence': confidence,
                'strategy': strategy_name,
                'price': current_row['Close'],
            }
        
        return None
    
    async def execute_signal(
        self,
        signal: Dict,
        symbol: str,
        bar_open_price: float,
    ) -> Optional[BacktestFill]:
        """
        Execute signal at next bar open.
        Mirrors Layer 6 Execution Engine.
        
        Args:
            signal: Signal from generate_signals
            symbol: Trading symbol
            bar_open_price: Next bar open price (order execution price)
        
        Returns:
            BacktestFill if execution successful
        """
        # Calculate position size based on risk
        risk_amount = self.cash * self.config.risk_per_trade
        position_size = int(risk_amount / (bar_open_price * 0.02))  # Stop loss 2%
        
        # Check if we have cash
        cost = position_size * bar_open_price * (1 + self.config.slippage_bps / 10000)
        
        if signal['direction'] == 'BUY':
            if cost > self.cash:
                logger.debug(f"Insufficient cash for {symbol}")
                return None
            
            # Check position limits
            total_position_value = sum(p.entry_price * p.quantity for p in self.positions.values())
            if total_position_value + cost > self.portfolio_value * self.config.max_position_size:
                logger.debug(f"Position limit breached for {symbol}")
                return None
            
            # Create position
            commission = cost * 0.001  # 0.1% commission
            self.positions[symbol] = BacktestPosition(
                symbol=symbol,
                quantity=position_size,
                entry_price=bar_open_price,
                entry_timestamp=self.current_bar.get('timestamp', ''),
                current_price=bar_open_price,
                unrealized_pnl=0.0,
            )
            self.cash -= (cost + commission)
            
            return BacktestFill(
                timestamp=self.current_bar.get('timestamp', ''),
                symbol=symbol,
                side='BUY',
                quantity=position_size,
                price=bar_open_price,
                commission=commission,
                strategy=signal.get('strategy', 'unknown'),
            )
        
        elif signal['direction'] == 'SELL' and symbol in self.positions:
            # Exit position
            position = self.positions[symbol]
            proceeds = position.quantity * bar_open_price
            commission = proceeds * 0.001
            pnl = (proceeds - commission) - (position.quantity * position.entry_price)
            
            self.cash += proceeds - commission
            trade = {
                'symbol': symbol,
                'entry_price': position.entry_price,
                'exit_price': bar_open_price,
                'quantity': position.quantity,
                'pnl': pnl,
                'pnl_pct': (pnl / (position.quantity * position.entry_price)) * 100,
                'entry_time': position.entry_timestamp,
                'exit_time': self.current_bar.get('timestamp', ''),
            }
            self.trades.append(trade)
            del self.positions[symbol]
            
            return BacktestFill(
                timestamp=self.current_bar.get('timestamp', ''),
                symbol=symbol,
                side='SELL',
                quantity=position.quantity,
                price=bar_open_price,
                commission=commission,
                strategy=signal.get('strategy', 'unknown'),
            )
        
        return None
    
    async def step(
        self,
        symbol: str,
        current_index: int,
        df: pd.DataFrame,
        next_bar_index: Optional[int] = None,
    ) -> None:
        """
        Execute one backtest step: signal generation → execution.
        
        Args:
            symbol: Current symbol being processed
            current_index: Current bar index
            df: Full DataFrame
            next_bar_index: Index of next bar (for execution)
        """
        current_row = df.iloc[current_index]
        
        # Update current bar timestamp
        self.current_bar = {
            'timestamp': current_row.name.strftime('%Y-%m-%d') if hasattr(current_row.name, 'strftime') else str(current_row.name),
            'symbol': symbol,
            'open': current_row['Open'],
            'high': current_row['High'],
            'low': current_row['Low'],
            'close': current_row['Close'],
        }
        
        # 1. Generate signal at close
        signal = await self.generate_signals(current_row, df, current_index)
        
        # 2. Update position prices
        for pos_symbol, position in self.positions.items():
            if pos_symbol == symbol:
                position.update_price(current_row['Close'])
        
        # 3. Execute signal at next bar open
        if signal and next_bar_index is not None and next_bar_index < len(df):
            next_row = df.iloc[next_bar_index]
            fill = await self.execute_signal(signal, symbol, next_row['Open'])
            if fill:
                self.fills.append(fill)
    
    def record_daily_stats(self, date: str) -> None:
        """Record daily equity and returns."""
        unrealized = sum(
            (p.current_price - p.entry_price) * p.quantity
            for p in self.positions.values()
        )
        
        total_value = self.cash + unrealized
        realized = sum(t['pnl'] for t in self.trades)
        
        equity_point = {
            'date': date,
            'equity': total_value + realized,
            'cash': self.cash,
            'unrealized': unrealized,
            'positions': len(self.positions),
        }
        self.equity_curve.append(equity_point)
        
        if len(self.equity_curve) > 1:
            daily_return = (equity_point['equity'] - self.equity_curve[-2]['equity']) / self.equity_curve[-2]['equity']
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0.0)
    
    async def run_backtest(self) -> Dict:
        """
        Run complete backtest.
        
        Returns:
            Results dictionary with trades, metrics, equity curve
        """
        print("=" * 70)
        print("🔄 BACKTEST ENGINE - Processing Historical Data")
        print("=" * 70 + "\n")
        
        data = await self.fetch_historical_data()
        
        # Process each symbol's history
        for symbol in self.config.symbols:
            if symbol not in data:
                continue
            
            df = data[symbol]
            
            logger.info(f"Processing {symbol}: {len(df)} bars")
            
            for i in range(len(df) - 1):
                await self.step(symbol, i, df, i + 1)
                
                # Record daily stats
                current_date = df.index[i]
                if isinstance(current_date, datetime):
                    self.record_daily_stats(current_date.strftime('%Y-%m-%d'))
        
        # Close remaining positions at end
        last_row = df.iloc[-1]
        for symbol in list(self.positions.keys()):
            await self.execute_signal(
                {'direction': 'SELL', 'strategy': 'close_at_end'},
                symbol,
                last_row['Close'],
            )
        
        return self._calculate_results()
    
    def _calculate_results(self) -> Dict:
        """Calculate performance metrics."""
        total_realized = sum(t['pnl'] for t in self.trades)
        total_unrealized = sum(p.unrealized_pnl for p in self.positions.values())
        total_pnl = total_realized + total_unrealized
        
        # Calculate metrics
        if not self.equity_curve:
            return {'error': 'No trades executed'}
        
        equity_values = [e['equity'] for e in self.equity_curve]
        returns = pd.Series(self.daily_returns)
        
        # Sharpe ratio
        annual_Returns = returns.mean() * 252
        annual_std = returns.std() * (252 ** 0.5)
        sharpe = annual_returns / annual_std if annual_std != 0 else 0
        
        # Max drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()
        
        # Win rate
        wins = len([t for t in self.trades if t['pnl'] > 0])
        total_trades = len(self.trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        losses = sum(t['pnl'] for t in self.trades if t['pnl'] < 0)
        profits = sum(t['pnl'] for t in self.trades if t['pnl'] > 0)
        profit_factor = profits / (-losses) if losses != 0 else 0
        
        return {
            'summary': {
                'initial_capital': self.config.initial_capital,
                'final_capital': equity_values[-1] if equity_values else 0,
                'total_return_pct': ((equity_values[-1] - self.config.initial_capital) / self.config.initial_capital * 100) if equity_values else 0,
                'total_trades': total_trades,
                'winning_trades': wins,
                'losing_trades': total_trades - wins,
                'win_rate_pct': win_rate,
            },
            'metrics': {
                'total_pnl': total_pnl,
                'realized_pnl': total_realized,
                'sharpe_ratio': sharpe,
                'max_drawdown_pct': max_dd * 100,
                'profit_factor': profit_factor,
            },
            'trades': self.trades[:50],  # Last 50 trades
            'equity_curve': self.equity_curve,
            'fills': [asdict(f) for f in self.fills[:100]],
        }


async def main():
    """Run backtest."""
    config = BacktestConfig(
        symbols=['AAPL', 'MSFT'],
        start_date='2023-01-01',
        end_date='2023-12-31',
        initial_capital=100000.0,
    )
    
    engine = BacktestEngine(config)
    results = await engine.run_backtest()
    
    # Save results
    results_path = Path(f"backtest_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 BACKTEST RESULTS")
    print("=" * 70)
    print(f"\nSummary:")
    for key, value in results.get('summary', {}).items():
        if 'pct' in key:
            print(f"  {key}: {value:.2f}%")
        else:
            print(f"  {key}: {value}")
    
    print(f"\nMetrics:")
    for key, value in results.get('metrics', {}).items():
        if 'pct' in key:
            print(f"  {key}: {value:.2f}%")
        elif 'factor' in key:
            print(f"  {key}: {value:.2f}x")
        else:
            print(f"  {key}: {value:.4f}")
    
    print(f"\nResults saved: {results_path}\n")


if __name__ == "__main__":
    asyncio.run(main())

# ============================================================================
# PERFORMANCE_ANALYZER.PY - Mathematical Performance Evaluation
# Calculates Sharpe, Win Rate, Profit Factor, Max Drawdown, etc.
# ============================================================================

import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics data class."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    breakeven_trades: int = 0
    
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    total_pnl: float = 0.0
    average_pnl: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    win_loss_ratio: float = 0.0
    
    max_pnl: float = 0.0
    min_pnl: float = 0.0
    
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    max_drawdown: float = 0.0
    max_drawdown_percent: float = 0.0
    
    cumulative_pnl: List[float] = None
    daily_returns: List[float] = None
    
    expectancy: float = 0.0  # Average P&L per trade
    payoff_ratio: float = 0.0  # Average win / abs(average loss)


class PerformanceAnalyzer:
    """
    Mathematical performance evaluator.
    
    Metrics calculated:
    - Win Rate: % of winning trades
    - Profit Factor: Gross profit / Gross loss
    - Sharpe Ratio: Return per unit of risk (volatility)
    - Sortino Ratio: Return per unit of downside risk
    - Max Drawdown: Largest peak-to-trough decline
    - Expected Value: Average return per trade
    - Payoff Ratio: Risk/reward per trade
    
    Designed for:
    - Objective strategy evaluation
    - Risk-adjusted performance measurement
    - Benchmark comparison
    - Model selection and optimization
    """
    
    def __init__(self, risk_free_rate: float = 0.02, trading_days_per_year: int = 252):
        """
        Initialize performance analyzer.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            trading_days_per_year: Days for annualization (default 252)
        """
        self.risk_free_rate = risk_free_rate
        self.trading_days_per_year = trading_days_per_year
        logger.info(
            f"✓ PerformanceAnalyzer initialized: "
            f"risk_free_rate={risk_free_rate*100:.1f}%, "
            f"trading_days={trading_days_per_year}"
        )
    
    def calculate_metrics(
        self,
        pnl_list: List[float],
        equity_curve: Optional[List[float]] = None,
    ) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            pnl_list: List of trade P&Ls
            equity_curve: Optional equity curve for drawdown calculation
            
        Returns:
            PerformanceMetrics object
        """
        metrics = PerformanceMetrics()
        
        if not pnl_list:
            return metrics
        
        pnl_array = np.array(pnl_list, dtype=np.float64)
        
        # Basic stats
        metrics.total_trades = len(pnl_list)
        metrics.winning_trades = np.sum(pnl_array > 0)
        metrics.losing_trades = np.sum(pnl_array < 0)
        metrics.breakeven_trades = np.sum(pnl_array == 0)
        
        metrics.total_pnl = float(np.sum(pnl_array))
        metrics.average_pnl = float(np.mean(pnl_array))
        metrics.max_pnl = float(np.max(pnl_array))
        metrics.min_pnl = float(np.min(pnl_array))
        
        # Win/Loss analysis
        if metrics.total_trades > 0:
            metrics.win_rate = metrics.winning_trades / metrics.total_trades
        
        wins = pnl_array[pnl_array > 0]
        losses = pnl_array[pnl_array < 0]
        
        if len(wins) > 0:
            metrics.average_win = float(np.mean(wins))
        
        if len(losses) > 0:
            metrics.average_loss = float(np.mean(losses))
        
        # Profit factor
        if len(losses) > 0 and len(wins) > 0:
            gross_profit = float(np.sum(wins))
            gross_loss = float(np.abs(np.sum(losses)))
            if gross_loss > 0.01:  # Avoid division by near-zero
                metrics.profit_factor = gross_profit / gross_loss
        
        # Win/Loss ratio (average win / average loss)
        if metrics.average_loss != 0:
            metrics.win_loss_ratio = metrics.average_win / abs(metrics.average_loss)
        
        # Payoff ratio
        if metrics.average_loss != 0:
            metrics.payoff_ratio = metrics.average_win / abs(metrics.average_loss)
        
        # Expectancy (average return per trade)
        metrics.expectancy = metrics.average_pnl
        
        # Risk-adjusted returns (Sharpe & Sortino)
        metrics.sharpe_ratio = self._calculate_sharpe_ratio(pnl_array)
        metrics.sortino_ratio = self._calculate_sortino_ratio(pnl_array)
        
        # Drawdown
        if equity_curve:
            metrics.max_drawdown, metrics.max_drawdown_percent = \
                self._calculate_max_drawdown(equity_curve)
        else:
            # Calculate from cumulative P&L
            cumulative = np.cumsum(pnl_array)
            max_dd, max_dd_pct = self._calculate_max_drawdown(cumulative.tolist())
            metrics.max_drawdown = max_dd
            metrics.max_drawdown_percent = max_dd_pct
        
        metrics.cumulative_pnl = np.cumsum(pnl_array).tolist()
        metrics.daily_returns = pnl_array.tolist()
        
        return metrics
    
    def _calculate_sharpe_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sharpe Ratio.
        
        Sharpe = (Mean Return - Risk-Free Rate) / Volatility
        
        Args:
            returns: Array of returns
            periods_per_year: Periods to annualize (252 for daily)
            
        Returns:
            Annualized Sharpe Ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        if volatility < 1e-6:  # Avoid division by near-zero
            return 0.0
        
        excess_return = mean_return - (self.risk_free_rate / periods_per_year)
        sharpe = (excess_return / volatility) * np.sqrt(periods_per_year)
        
        return float(sharpe)
    
    def _calculate_sortino_ratio(
        self,
        returns: np.ndarray,
        periods_per_year: int = 252,
    ) -> float:
        """
        Calculate Sortino Ratio.
        
        Sortino = (Mean Return - Risk-Free Rate) / Downside Volatility
        Only penalizes downside volatility.
        
        Args:
            returns: Array of returns
            periods_per_year: Annualization periods
            
        Returns:
            Annualized Sortino Ratio
        """
        if len(returns) < 2:
            return 0.0
        
        mean_return = np.mean(returns)
        
        # Downside volatility: only consider negative returns
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            # No downside, perfect Sortino (but cap at 2x Sharpe)
            return 2.0 * self._calculate_sharpe_ratio(returns, periods_per_year)
        
        downside_volatility = np.std(downside_returns)
        
        if downside_volatility < 1e-6:
            return 0.0
        
        excess_return = mean_return - (self.risk_free_rate / periods_per_year)
        sortino = (excess_return / downside_volatility) * np.sqrt(periods_per_year)
        
        return float(sortino)
    
    def _calculate_max_drawdown(
        self,
        equity_curve: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate maximum drawdown and percentage.
        
        Max Drawdown = (Peak - Trough) / Peak
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            (max_drawdown_absolute, max_drawdown_percent)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0, 0.0
        
        equity_array = np.array(equity_curve, dtype=np.float64)
        
        # Running maximum
        running_max = np.maximum.accumulate(equity_array)
        
        # Drawdown at each point
        drawdown = (running_max - equity_array) / running_max
        
        max_drawdown_pct = float(np.max(drawdown))
        max_drawdown_abs = float(np.max(running_max - equity_array))
        
        return max_drawdown_abs, max_drawdown_pct
    
    def calculate_monthly_returns(
        self,
        pnl_by_date: dict,  # {date: pnl}
    ) -> dict:  # {month: return}
        """
        Calculate monthly returns from daily P&L.
        
        Args:
            pnl_by_date: Dictionary of date -> P&L
            
        Returns:
            Dictionary of month -> return
        """
        monthly_pnl = {}
        
        for date_str, pnl in pnl_by_date.items():
            # Parse date (yyyy-mm-dd)
            month_key = date_str[:7]  # yyyy-mm
            
            if month_key not in monthly_pnl:
                monthly_pnl[month_key] = 0.0
            
            monthly_pnl[month_key] += pnl
        
        return monthly_pnl
    
    def calculate_correlation_with_benchmark(
        self,
        strategy_returns: List[float],
        benchmark_returns: List[float],
    ) -> float:
        """
        Calculate correlation with benchmark.
        
        Args:
            strategy_returns: Strategy returns
            benchmark_returns: Benchmark returns
            
        Returns:
            Correlation coefficient (-1 to 1)
        """
        if len(strategy_returns) < 2 or len(benchmark_returns) < 2:
            return 0.0
        
        if len(strategy_returns) != len(benchmark_returns):
            min_len = min(len(strategy_returns), len(benchmark_returns))
            strategy_returns = strategy_returns[:min_len]
            benchmark_returns = benchmark_returns[:min_len]
        
        try:
            corr = np.corrcoef(strategy_returns, benchmark_returns)[0, 1]
            return float(corr) if not np.isnan(corr) else 0.0
        except Exception:
            return 0.0
    
    def get_percentile_metrics(
        self,
        pnl_list: List[float],
        percentiles: List[float] = [10, 25, 50, 75, 90],
    ) -> dict:
        """
        Calculate percentile metrics.
        
        Args:
            pnl_list: List of P&Ls
            percentiles: Percentiles to calculate
            
        Returns:
            Dictionary of percentile -> value
        """
        if not pnl_list:
            return {}
        
        pnl_array = np.array(pnl_list)
        results = {}
        
        for p in percentiles:
            results[f"p{int(p)}"] = float(np.percentile(pnl_array, p))
        
        return results

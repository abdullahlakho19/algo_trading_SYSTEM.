# ============================================================================
# MONTE CARLO SIMULATOR - L4 QUANTITATIVE LAYER
# Stress-tests trade setups with 1,000+ random outcome scenarios
# ============================================================================

import logging
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class MonteCarloResult:
    """
    Monte Carlo simulation results for a trade setup.
    
    Attributes:
        num_simulations: Number of scenarios run
        win_rate: % of simulations that hit take profit
        loss_rate: % of simulations that hit stop loss
        avg_return_pct: Average return across all simulations
        best_case_return: Best outcome (95th percentile)
        worst_case_return: Worst outcome (5th percentile)
        profit_factor: Ratio of total wins to total losses
        sharpe_ratio: Risk-adjusted return metric
        max_drawdown: Worst drawdown across simulations
        confidence_level: Confidence in the setup 0-1
        returns_distribution: Array of all returns from simulations
        max_profit: Maximum possible profit
        max_loss: Maximum possible loss
        probability_adjusted_return: Expected value considering win/loss probability
    """
    num_simulations: int
    win_rate: float
    loss_rate: float
    avg_return_pct: float
    best_case_return: float
    worst_case_return: float
    profit_factor: float
    sharpe_ratio: float
    max_drawdown: float
    confidence_level: float
    returns_distribution: np.ndarray
    max_profit: float
    max_loss: float
    probability_adjusted_return: float


class MonteCarlo:
    """
    Monte Carlo Simulation Engine for Pre-Trade Stress Testing.
    
    Runs 1,000+ random outcome scenarios to answer:
    - What's the probability of hitting take profit?
    - What's the worst-case drawdown?
    - What's the profit factor (reward/risk)?
    - Is the risk/reward ratio favorable?
    
    Mathematical approach:
    1. Define entry, stop loss, and take profit levels
    2. Simulate random price walks using geometric Brownian motion
    3. Track outcome for each path (TP hit, SL hit, or timeout)
    4. Aggregate statistics across all paths
    5. Calculate risk-adjusted metrics (Sharpe, profit factor, etc.)
    """
    
    def __init__(self, num_simulations: int = 1000, risk_free_rate: float = 0.02):
        """
        Initialize Monte Carlo Engine.
        
        Args:
            num_simulations: Number of scenarios to simulate per setup
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.num_simulations = num_simulations
        self.risk_free_rate = risk_free_rate
        logger.info(f"✓ MonteCarlo initialized ({num_simulations} simulations)")
    
    # ========================================================================
    # GEOMETRIC BROWNIAN MOTION
    # ========================================================================
    
    def generate_price_paths(
        self,
        current_price: float,
        expected_return: float,
        volatility: float,
        time_steps: int,
        dt: float = 1.0 / 252.0
    ) -> np.ndarray:
        """
        Generate random price paths using Geometric Brownian Motion (GBM).
        
        GBM formula: dS = μ*S*dt + σ*S*dW
        
        Where:
        - S = current price
        - μ = expected return (drift)
        - σ = volatility (annualized)
        - dW = random Wiener process increment
        
        Args:
            current_price: Starting price level
            expected_return: Expected drift (annualized return)
            volatility: Price volatility (annualized, 0.2 = 20%)
            time_steps: Number of periods to simulate
            dt: Time delta (1/252 = 1 trading day)
            
        Returns:
            Array of shape (num_simulations, time_steps) with price paths
        """
        # Initialize price matrix
        paths = np.zeros((self.num_simulations, time_steps))
        paths[:, 0] = current_price
        
        # Generate random increments
        random_increments = np.random.standard_normal((self.num_simulations, time_steps - 1))
        
        # Calculate drift and diffusion components
        drift = expected_return * dt
        diffusion = volatility * np.sqrt(dt)
        
        # Simulate price paths
        for t in range(1, time_steps):
            # dS/S = (μ - σ²/2)*dt + σ*dW
            paths[:, t] = paths[:, t - 1] * np.exp(
                (drift - 0.5 * volatility ** 2 * dt) + diffusion * random_increments[:, t - 1]
            )
        
        return paths
    
    # ========================================================================
    # OUTCOME CLASSIFICATION
    # ========================================================================
    
    def classify_outcomes(
        self,
        price_paths: np.ndarray,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        time_limit: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Classify outcomes of each simulation path.
        
        For each path, determine:
        - Did price hit take profit first? (WIN)
        - Did price hit stop loss first? (LOSS)
        - Did price timeout without hitting? (NEUTRAL)
        
        Args:
            price_paths: Array of shape (num_simulations, time_steps)
            entry_price: Trade entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            time_limit: Maximum bars to hold (None = entire path)
            
        Returns:
            Tuple of (returns_pct, outcomes) arrays
        """
        num_sims, num_steps = price_paths.shape
        
        if time_limit is None:
            time_limit = num_steps
        else:
            time_limit = min(time_limit, num_steps)
        
        returns_pct = np.zeros(num_sims)
        outcomes = np.zeros(num_sims)  # 1=win, -1=loss, 0=neutral
        
        # Ensure SL < Entry < TP or TP < Entry < SL (based on direction)
        is_long = take_profit > entry_price
        
        if is_long:
            sl_level = min(stop_loss, entry_price)
            tp_level = max(take_profit, entry_price)
        else:
            sl_level = max(stop_loss, entry_price)
            tp_level = min(take_profit, entry_price)
        
        for i in range(num_sims):
            path = price_paths[i, :time_limit]
            
            # Check if TP or SL hit
            tp_hit = False
            sl_hit = False
            
            if is_long:
                # For long: TP is above entry, SL is below
                if np.any(path >= tp_level):
                    tp_hit = True
                    exit_price = tp_level
                if np.any(path <= sl_level):
                    sl_hit = True
                    exit_price = sl_level
            else:
                # For short: TP is below entry, SL is above
                if np.any(path <= tp_level):
                    tp_hit = True
                    exit_price = tp_level
                if np.any(path >= sl_level):
                    sl_hit = True
                    exit_price = sl_level
            
            # Determine which hit first
            if tp_hit and not sl_hit:
                returns_pct[i] = (exit_price - entry_price) / entry_price * 100
                outcomes[i] = 1  # WIN
            elif sl_hit and not tp_hit:
                returns_pct[i] = (exit_price - entry_price) / entry_price * 100
                outcomes[i] = -1  # LOSS
            elif sl_hit and tp_hit:
                # Both hit - find which one hit first
                sl_idx = np.where(path <= sl_level)[0][0] if np.any(path <= sl_level) else np.inf
                tp_idx = np.where(path >= tp_level)[0][0] if np.any(path >= tp_level) else np.inf
                
                if sl_idx < tp_idx:
                    returns_pct[i] = (sl_level - entry_price) / entry_price * 100
                    outcomes[i] = -1
                else:
                    returns_pct[i] = (tp_level - entry_price) / entry_price * 100
                    outcomes[i] = 1
            else:
                # Neither hit - use final price
                exit_price = path[-1]
                returns_pct[i] = (exit_price - entry_price) / entry_price * 100
                outcomes[i] = 0  # NEUTRAL
        
        return returns_pct, outcomes
    
    # ========================================================================
    # STATISTICS CALCULATION
    # ========================================================================
    
    def calculate_statistics(
        self,
        returns_pct: np.ndarray,
        outcomes: np.ndarray,
        risk_per_trade: float = 1.0
    ) -> MonteCarloResult:
        """
        Calculate comprehensive statistics from simulation results.
        
        Args:
            returns_pct: Array of returns from each simulation
            outcomes: Array of outcomes (-1, 0, 1)
            risk_per_trade: Risk per trade as % of account (for Sharpe calc)
            
        Returns:
            MonteCarloResult with all metrics
        """
        num_sims = len(returns_pct)
        
        # Outcome rates
        wins = np.sum(outcomes == 1)
        losses = np.sum(outcomes == -1)
        neutrals = np.sum(outcomes == 0)
        
        win_rate = (wins / num_sims) * 100 if num_sims > 0 else 0
        loss_rate = (losses / num_sims) * 100 if num_sims > 0 else 0
        
        # Return metrics
        avg_return = np.mean(returns_pct)
        best_case = np.percentile(returns_pct, 95)  # 95th percentile (best)
        worst_case = np.percentile(returns_pct, 5)   # 5th percentile (worst)
        
        # Profit factor: Total profit / Total loss
        winning_returns = returns_pct[outcomes == 1]
        losing_returns = returns_pct[outcomes == -1]
        
        total_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0
        total_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 1e-9
        
        profit_factor = total_profit / total_loss if total_loss > 0 else 0
        
        # Sharpe Ratio: (average return - risk-free rate) / std deviation
        # Calculate daily risk-free rate
        daily_rf_rate = self.risk_free_rate / 252
        excess_returns = returns_pct / 100 - daily_rf_rate
        sharpe_ratio = (np.mean(excess_returns) / (np.std(excess_returns) + 1e-9)) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative_returns = np.cumprod(1 + (returns_pct / 100))
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100 if len(drawdown) > 0 else 0
        
        # Confidence level: Based on win rate and Sharpe
        confidence = (win_rate / 100 * 0.6 + (sharpe_ratio / 5) * 0.4)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        # Max profit/loss
        max_profit = np.max(returns_pct) if len(returns_pct) > 0 else 0
        max_loss = np.min(returns_pct) if len(returns_pct) > 0 else 0
        
        # Probability-adjusted expected value
        # E[return] = (% win * avg win) + (% loss * avg loss)
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0
        avg_loss = np.mean(losing_returns) if len(losing_returns) > 0 else 0
        
        prob_adjusted_return = (win_rate / 100 * avg_win) + (loss_rate / 100 * avg_loss)
        
        return MonteCarloResult(
            num_simulations=num_sims,
            win_rate=win_rate,
            loss_rate=loss_rate,
            avg_return_pct=avg_return,
            best_case_return=best_case,
            worst_case_return=worst_case,
            profit_factor=profit_factor,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            confidence_level=confidence,
            returns_distribution=returns_pct,
            max_profit=max_profit,
            max_loss=max_loss,
            probability_adjusted_return=prob_adjusted_return
        )
    
    # ========================================================================
    # MAIN SIMULATION INTERFACE
    # ========================================================================
    
    def simulate(
        self,
        current_price: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        volatility: float,
        expected_return: float = 0.0,
        num_days: int = 5,
        risk_per_trade: float = 1.0
    ) -> MonteCarloResult:
        """
        Run complete Monte Carlo simulation for a trade setup.
        
        Args:
            current_price: Current market price
            entry_price: Trade entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            volatility: Annualized volatility (0.2 = 20%)
            expected_return: Expected daily return bias
            num_days: Number of days to hold the trade
            risk_per_trade: Risk as % of account
            
        Returns:
            MonteCarloResult with all statistics
        """
        # Convert days to trading periods (252 trading days per year)
        time_steps = max(num_days, 1)
        
        # Generate price paths
        logger.info(f"Generating {self.num_simulations} price paths...")
        paths = self.generate_price_paths(
            current_price=current_price,
            expected_return=expected_return,
            volatility=volatility,
            time_steps=time_steps
        )
        
        # Classify outcomes
        logger.info("Classifying outcomes...")
        returns_pct, outcomes = self.classify_outcomes(
            price_paths=paths,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            time_limit=time_steps
        )
        
        # Calculate statistics
        logger.info("Calculating statistics...")
        result = self.calculate_statistics(returns_pct, outcomes, risk_per_trade)
        
        logger.info(f"✓ Monte Carlo Results:")
        logger.info(f"  - Win Rate: {result.win_rate:.1f}%")
        logger.info(f"  - Profit Factor: {result.profit_factor:.2f}")
        logger.info(f"  - Sharpe Ratio: {result.sharpe_ratio:.2f}")
        logger.info(f"  - Expected Return: {result.probability_adjusted_return:.2f}%")
        logger.info(f"  - Confidence: {result.confidence_level:.1%}")
        
        return result

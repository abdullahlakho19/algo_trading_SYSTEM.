# ============================================================================
# EDGE CALCULATOR - L4 QUANTITATIVE LAYER
# Calculates if strategy has true mathematical edge vs random chance
# ============================================================================

import logging
import numpy as np
from scipy import stats
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class EdgeAnalysis:
    """
    Statistical edge analysis result.
    
    Attributes:
        has_edge: Boolean - does strategy have true edge?
        edge_percentage: % edge over random chance
        statistical_significance: P-value from hypothesis test
        confidence_level: Confidence in edge claim (95%/99%)
        win_rate: Strategy win rate %
        expectancy: Expected $ per trade
        kelly_criterion: Kelly leverage factor (optimal bet size)
        sample_size: Number of trades analyzed
        min_trades_for_confidence: Trades needed at current win rate
        rationale: Explanation of edge (or lack thereof)
    """
    has_edge: bool
    edge_percentage: float
    statistical_significance: float
    confidence_level: float
    win_rate: float
    expectancy: float
    kelly_criterion: float
    sample_size: int
    min_trades_for_confidence: int
    rationale: str


class EdgeCalculator:
    """
    Statistical Edge Calculator - Validates True Trading Edge.
    
    Prevents the bot from trading on "lucky noise" by testing:
    1. Win rate significance (vs 50% random coin flip)
    2. Expectancy: (Win% * Avg Win) - (Loss% * Avg Loss)
    3. Kelly Criterion: Optimal leverage based on edge
    4. P-value hypothesis testing for statistical significance
    
    Mathematical foundation:
    - Binomial test: Is win rate significantly > 50%?
    - Expectancy formula: E = (Win% * Avg Win) - (Loss% * Avg Loss)
    - Kelly Criterion: f = (Win% * Avg Win / Avg Loss) - (Loss%)
    - Sample size: How many trades to validate?
    """
    
    def __init__(self, significance_level: float = 0.05, min_sample_size: int = 30):
        """
        Initialize Edge Calculator.
        
        Args:
            significance_level: P-value threshold (0.05 = 95% confidence)
            min_sample_size: Minimum trades to claim edge (30 is conservative)
        """
        self.significance_level = significance_level
        self.min_sample_size = min_sample_size
        logger.info(f"✓ EdgeCalculator initialized (α={significance_level}, min_n={min_sample_size})")
    
    # ========================================================================
    # EDGE DETECTION
    # ========================================================================
    
    def test_win_rate_significance(
        self,
        num_wins: int,
        num_total_trades: int,
        null_hypothesis_rate: float = 0.50
    ) -> float:
        """
        Test if win rate is statistically significant via binomial test.
        
        Hypothesis:
        - H0 (Null): Win rate = 50% (random chance)
        - H1 (Alt): Win rate > 50% (has edge)
        
        Args:
            num_wins: Number of winning trades
            num_total_trades: Total number of trades
            null_hypothesis_rate: Assumed chance of random outcome (0.50)
            
        Returns:
            P-value: Probability of observed outcome under null hypothesis
            - p < 0.05 = 95% confident we have an edge
            - p < 0.01 = 99% confident
            - p > 0.05 = No significant edge detected
        """
        # Binomial test: scipy.stats.binom_test
        # Calculates probability of observing num_wins or more wins
        # if true probability is null_hypothesis_rate
        
        p_value = stats.binom_test(
            num_wins,
            num_total_trades,
            null_hypothesis_rate,
            alternative='greater'  # Testing if win_rate > 50%
        )
        
        return p_value
    
    def calculate_expectancy(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> float:
        """
        Calculate expected value per trade.
        
        E = (WinRate% * AvgWin) - (LossRate% * AbsAvgLoss)
        
        Interpretation:
        - E > 0: Positive expectancy (edge exists)
        - E < 0: Negative expectancy (lossy strategy)
        - E = 0: Breakeven
        
        Args:
            win_rate: Win rate as decimal (0.5 = 50%)
            avg_win: Average profit per winning trade (in $)
            avg_loss: Average loss per losing trade (in $ - enter as positive)
            
        Returns:
            Expected value per trade in $
        """
        loss_rate = 1.0 - win_rate
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        return expectancy
    
    def calculate_kelly_criterion(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        safety_factor: float = 0.25
    ) -> float:
        """
        Calculate Kelly Criterion for optimal bet sizing.
        
        Kelly Criterion: f = (WR * W - LR * L) / W
        Where:
        - WR = win rate
        - W = average win
        - LR = loss rate
        - L = average loss
        
        Result: Fraction of capital to risk per trade for max growth
        
        Note: Full Kelly is aggressive and can lead to ruin.
        Recommendations:
        - Kelly/2 = conservative
        - Kelly/4 = very conservative
        
        Args:
            win_rate: Win rate as decimal
            avg_win: Average profit per win (in %)
            avg_loss: Average loss per loss (in % - enter positive)
            safety_factor: Fraction of Kelly to use (0.25 = 1/4 Kelly)
            
        Returns:
            Kelly fraction (0.05 = 5% per trade)
        """
        loss_rate = 1.0 - win_rate
        
        # Avoid division by zero
        if avg_win <= 0:
            return 0.0
        
        # Kelly formula
        kelly = (win_rate * avg_win - loss_rate * avg_loss) / avg_win
        
        # Apply safety factor (e.g., Kelly/4)
        kelly_safe = kelly * safety_factor
        
        # Clip to reasonable bounds (never risk more than 5% per trade without strong edge)
        kelly_safe = np.clip(kelly_safe, 0.01, 0.05)
        
        return kelly_safe
    
    # ========================================================================
    # SAMPLE SIZE ANALYSIS
    # ========================================================================
    
    def calculate_min_sample_size(
        self,
        desired_confidence: float = 0.95,
        win_rate: Optional[float] = None,
        effect_size: float = 0.10
    ) -> int:
        """
        Calculate minimum trades needed to validate edge with confidence.
        
        Uses normal approximation to binomial:
        n = ((z^2 * p * (1-p)) / effect_size^2)
        
        Args:
            desired_confidence: Confidence level (0.95 = 95%)
            win_rate: Actual win rate (None = assume 55%)
            effect_size: How far from 50% to detect (0.10 = 5% away)
            
        Returns:
            Minimum sample size
        """
        if win_rate is None:
            win_rate = 0.55
        
        # Z-score for desired confidence
        alpha = 1.0 - desired_confidence
        z_score = stats.norm.ppf(1 - alpha / 2)
        
        # Null hypothesis probability (50% = no edge)
        p0 = 0.50
        
        # Required sample size
        # This uses normal approximation to binomial
        variance = p0 * (1 - p0)
        
        n = (z_score ** 2 * variance) / (effect_size ** 2)
        
        return int(np.ceil(n))
    
    # ========================================================================
    # COMPLETE EDGE ANALYSIS
    # ========================================================================
    
    def analyze_edge(
        self,
        trades_data: List[Dict],
        account_size: float = 10000.0
    ) -> EdgeAnalysis:
        """
        Complete statistical edge analysis.
        
        Parameters are expected in trades_data:
        - 'outcome': 'win' or 'loss'
        - 'return_pct': Return as percentage
        - 'pnl': Profit/loss in dollars
        
        Args:
            trades_data: List of trade dicts with outcome and P&L
            account_size: Account size for leverage calculations
            
        Returns:
            EdgeAnalysis with full edge breakdown
        """
        
        if not trades_data:
            return EdgeAnalysis(
                has_edge=False,
                edge_percentage=0.0,
                statistical_significance=1.0,
                confidence_level=0.0,
                win_rate=0.0,
                expectancy=0.0,
                kelly_criterion=0.0,
                sample_size=0,
                min_trades_for_confidence=self.min_sample_size,
                rationale="No trade history provided"
            )
        
        n_trades = len(trades_data)
        
        # ====================================================================
        # EXTRACT TRADE STATISTICS
        # ====================================================================
        
        wins = [t for t in trades_data if t.get('outcome') == 'win']
        losses = [t for t in trades_data if t.get('outcome') == 'loss']
        
        n_wins = len(wins)
        n_losses = len(losses)
        win_rate = n_wins / n_trades if n_trades > 0 else 0
        
        # Calculate average win/loss
        if wins:
            avg_win = np.mean([t.get('pnl', t.get('return_pct', 0)) for t in wins])
        else:
            avg_win = 0
        
        if losses:
            avg_loss = abs(np.mean([t.get('pnl', t.get('return_pct', 0)) for t in losses]))
        else:
            avg_loss = 0
        
        # ====================================================================
        # STATISTICAL SIGNIFICANCE TEST
        # ====================================================================
        
        p_value = self.test_win_rate_significance(n_wins, n_trades)
        
        has_edge = (
            p_value < self.significance_level and
            n_trades >= self.min_sample_size
        )
        
        # Edge percentage: how much better than 50%?
        edge_pct = (win_rate - 0.50) * 100
        
        # ====================================================================
        # EXPECTANCY CALCULATION
        # ====================================================================
        
        expectancy = self.calculate_expectancy(win_rate, avg_win, avg_loss)
        
        # ====================================================================
        # KELLY CRITERION
        # ====================================================================
        
        kelly = self.calculate_kelly_criterion(win_rate, avg_win, avg_loss)
        
        # ====================================================================
        # SAMPLE SIZE ANALYSIS
        # ====================================================================
        
        min_sample = self.calculate_min_sample_size(
            desired_confidence=0.95,
            win_rate=win_rate
        )
        
        # ====================================================================
        # GENERATE RATIONALE
        # ====================================================================
        
        rationale_lines = []
        
        rationale_lines.append(f"Sample size: {n_trades} trades")
        rationale_lines.append(f"Win rate: {win_rate:.1%} ({n_wins}W / {n_losses}L)")
        rationale_lines.append(f"Average win: ${avg_win:.2f}, Average loss: ${avg_loss:.2f}")
        
        if n_trades < self.min_sample_size:
            rationale_lines.append(
                f"⚠️  Insufficient sample size ({n_trades} < {self.min_sample_size} required)"
            )
        
        if p_value < 0.01:
            rationale_lines.append(f"✓ HIGHLY SIGNIFICANT (p={p_value:.4f}) - 99%+ confidence")
        elif p_value < 0.05:
            rationale_lines.append(f"✓ Significant (p={p_value:.4f}) - 95% confidence")
        else:
            rationale_lines.append(f"✗ NOT SIGNIFICANT (p={p_value:.4f}) - Could be luck")
        
        if expectancy > 0:
            rationale_lines.append(f"✓ Positive expectancy: ${expectancy:.2f}/trade")
        else:
            rationale_lines.append(f"✗ Negative expectancy: ${expectancy:.2f}/trade")
        
        if has_edge:
            rationale_lines.append(f"\n✓✓✓ STRATEGY HAS EDGE - {edge_pct:.1f}% above random chance")
            rationale_lines.append(f"Kelly Criterion: Risk {kelly:.1%} per trade (1/{1/kelly:.0f})")
        else:
            if win_rate > 0.50:
                reason = "Insufficient sample size or weak statistical significance"
            else:
                reason = "Win rate at or below chance"
            rationale_lines.append(f"\n✗✗✗ NO EDGE DETECTED - {reason}")
        
        rationale = "\n".join(rationale_lines)
        
        return EdgeAnalysis(
            has_edge=has_edge,
            edge_percentage=edge_pct,
            statistical_significance=p_value,
            confidence_level=1.0 - self.significance_level,
            win_rate=win_rate * 100,
            expectancy=expectancy,
            kelly_criterion=kelly,
            sample_size=n_trades,
            min_trades_for_confidence=min_sample,
            rationale=rationale
        )
    
    # ========================================================================
    # RISK OF RUIN CALCULATION
    # ========================================================================
    
    def calculate_risk_of_ruin(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        kelly_fraction: float,
        max_drawdown_target: float = 0.20
    ) -> float:
        """
        Calculate probability of account ruin using Kelly Criterion.
        
        Risk of Ruin = Risk of losing everything or max_drawdown_target.
        
        Lower RoR is better:
        - RoR < 1%: Safe
        - RoR 1-5%: Acceptable with caution
        - RoR > 5%: Dangerous
        
        Args:
            win_rate: Win rate as decimal
            avg_win: Average win %
            avg_loss: Average loss % (positive number)
            kelly_fraction: Kelly factor being used (0.25 = 1/4 Kelly)
            max_drawdown_target: Max allowed drawdown (0.2 = 20%)
            
        Returns:
            Risk of ruin 0-1 (lower is better)
        """
        loss_rate = 1.0 - win_rate
        
        # Calculate "win" and "loss" from Kelly perspective
        # Assuming symmetric size bets
        win_amount = kelly_fraction * avg_win
        loss_amount = kelly_fraction * avg_loss
        
        # Probability of eventual ruin
        # U = ((1-WR) / WR) ^ (Account / Loss_per_trade)
        # Simplified risk of ruin approximation
        
        if win_amount <= 0 or loss_amount <= 0:
            return 0.5
        
        ratio = loss_rate / win_rate
        
        # Risk of ruin (approximation)
        if ratio < 1.0:
            ror = (ratio ** 10)  # Small ratio = low RoR
        else:
            ror = 1.0 - (1.0 / (ratio ** 10))
        
        return np.clip(ror, 0.0, 1.0)

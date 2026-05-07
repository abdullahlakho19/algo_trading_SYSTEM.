# ============================================================================
# MODEL_EVALUATOR.PY - Model Health Score (A-F Rating System)
# Comprehensive model evaluation and grading
# ============================================================================

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from reporting.performance_analyzer import PerformanceAnalyzer, PerformanceMetrics

logger = logging.getLogger(__name__)


class HealthGrade(str, Enum):
    """Model health grades."""
    A_PLUS = "A+"
    A = "A"
    A_MINUS = "A-"
    B_PLUS = "B+"
    B = "B"
    B_MINUS = "B-"
    C_PLUS = "C+"
    C = "C"
    C_MINUS = "C-"
    D = "D"
    F = "F"


@dataclass
class HealthScore:
    """Model health score details."""
    overall_grade: HealthGrade
    overall_score: float  # 0-100
    
    # Component scores
    profitability_score: float  # 0-100
    consistency_score: float  # 0-100
    risk_management_score: float  # 0-100
    efficiency_score: float  # 0-100
    
    # Ratings
    profitability_grade: HealthGrade
    consistency_grade: HealthGrade
    risk_management_grade: HealthGrade
    efficiency_grade: HealthGrade
    
    # Reasoning
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    
    # Raw metrics
    metrics: PerformanceMetrics


class ModelEvaluator:
    """
    Comprehensive model health evaluator.
    
    Scoring Components:
    1. Profitability (40%): Win rate, profit factor, R:R ratio
    2. Consistency (25%): Sharpe ratio, Sortino ratio, variance
    3. Risk Management (20%): Max drawdown, drawdown recovery
    4. Efficiency (15%): Average P&L, expectancy, trades per period
    
    Grading Scale:
    A+ (95-100): Exceptional, institutional-grade performance
    A (90-95): Excellent performance
    B+ (80-90): Good, ready for live trading
    B (70-80): Adequate, needs optimization
    C+ (60-70): Marginal, risky
    C (50-60): Poor, not recommended
    D (40-50): Very poor
    F (<40): Fail
    
    Designed for:
    - Model selection and ranking
    - Performance validation
    - Go/no-go trading decisions
    - Portfolio optimization
    """
    
    def __init__(self, analyzer: Optional[PerformanceAnalyzer] = None):
        """
        Initialize model evaluator.
        
        Args:
            analyzer: PerformanceAnalyzer instance (creates default if None)
        """
        self.analyzer = analyzer or PerformanceAnalyzer()
        logger.info("✓ ModelEvaluator initialized")
    
    def _score_to_grade(self, score: float) -> HealthGrade:
        """Convert numerical score to letter grade."""
        if score >= 95:
            return HealthGrade.A_PLUS
        elif score >= 90:
            return HealthGrade.A
        elif score >= 85:
            return HealthGrade.A_MINUS
        elif score >= 80:
            return HealthGrade.B_PLUS
        elif score >= 75:
            return HealthGrade.B
        elif score >= 70:
            return HealthGrade.B_MINUS
        elif score >= 65:
            return HealthGrade.C_PLUS
        elif score >= 60:
            return HealthGrade.C
        elif score >= 55:
            return HealthGrade.C_MINUS
        elif score >= 40:
            return HealthGrade.D
        else:
            return HealthGrade.F
    
    def _calculate_profitability_score(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate profitability component score.
        
        Factors:
        - Win rate: target 55%+
        - Profit factor: target 1.5+
        - Average win / avg loss ratio: target 1.2+
        - Total P&L: absolute return
        
        Returns:
            Score 0-100
        """
        score = 0.0
        
        # Win rate (target 55%+)
        win_rate_score = min(100, (metrics.win_rate / 0.55) * 100)
        score += win_rate_score * 0.30
        
        # Profit factor (target 1.5+)
        pf_score = min(100, (metrics.profit_factor / 1.5) * 100)
        score += pf_score * 0.40
        
        # Win/Loss ratio (target 1.2+)
        if metrics.win_loss_ratio > 0:
            wr_score = min(100, (metrics.win_loss_ratio / 1.2) * 100)
        else:
            wr_score = 0.0
        score += wr_score * 0.30
        
        return min(100, score)
    
    def _calculate_consistency_score(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate consistency component score.
        
        Factors:
        - Sharpe ratio: target 1.5+
        - Sortino ratio: target 2.0+
        - Return variance: lower is better
        - Equity curve smoothness
        
        Returns:
            Score 0-100
        """
        score = 0.0
        
        # Sharpe ratio (target 1.5+)
        sharpe_score = min(100, (metrics.sharpe_ratio / 1.5) * 100)
        score += sharpe_score * 0.40
        
        # Sortino ratio (target 2.0+)
        sortino_score = min(100, (metrics.sortino_ratio / 2.0) * 100)
        score += sortino_score * 0.40
        
        # Return variance (lower is better)
        if metrics.daily_returns:
            import numpy as np
            std_dev = np.std(metrics.daily_returns)
            mean_return = np.mean(metrics.daily_returns)
            
            # Coefficient of variation (std/mean)
            if mean_return > 0.01:
                cv = std_dev / mean_return
                cv_score = max(0, 100 - (cv * 20))  # Lower CV is better
            else:
                cv_score = 0.0
        else:
            cv_score = 0.0
        
        score += cv_score * 0.20
        
        return min(100, score)
    
    def _calculate_risk_management_score(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate risk management component score.
        
        Factors:
        - Max drawdown: target < 20%
        - Drawdown recovery: faster is better
        - Drawdown frequency: lower is better
        
        Returns:
            Score 0-100
        """
        score = 0.0
        
        # Max drawdown (target < 20%, penalize heavily for > 40%)
        max_dd = metrics.max_drawdown_percent
        if max_dd < 0.10:
            dd_score = 100.0
        elif max_dd < 0.20:
            dd_score = 100 - ((max_dd - 0.10) / 0.10) * 20
        elif max_dd < 0.40:
            dd_score = 80 - ((max_dd - 0.20) / 0.20) * 60
        else:
            dd_score = max(0, 20 - ((max_dd - 0.40) / 0.40) * 20)
        
        score += dd_score * 1.0  # Max drawdown is most important
        
        return min(100, score)
    
    def _calculate_efficiency_score(self, metrics: PerformanceMetrics) -> float:
        """
        Calculate efficiency component score.
        
        Factors:
        - Average P&L per trade
        - Expectancy (edge)
        - Profit per trade vs risk
        
        Returns:
            Score 0-100
        """
        score = 0.0
        
        # Expectancy (average return per trade)
        # Target: positive expectancy
        if metrics.expectancy > 0:
            expectancy_score = min(100, (abs(metrics.expectancy) / 100) * 100)
        else:
            expectancy_score = max(0, 50 + (metrics.expectancy / 100) * 50)
        
        score += expectancy_score * 0.50
        
        # Payoff ratio (average win / average loss)
        # Target: 1.5+
        if metrics.payoff_ratio > 0:
            payoff_score = min(100, (metrics.payoff_ratio / 1.5) * 100)
        else:
            payoff_score = 0.0
        
        score += payoff_score * 0.50
        
        return min(100, score)
    
    async def evaluate_model(
        self,
        pnl_list: List[float],
        equity_curve: Optional[List[float]] = None,
    ) -> HealthScore:
        """
        Evaluate model and generate comprehensive health score.
        
        Args:
            pnl_list: List of trade P&Ls
            equity_curve: Optional equity curve
            
        Returns:
            HealthScore object
        """
        # Calculate raw metrics
        metrics = self.analyzer.calculate_metrics(pnl_list, equity_curve)
        
        # Calculate component scores
        profitability_score = self._calculate_profitability_score(metrics)
        consistency_score = self._calculate_consistency_score(metrics)
        risk_management_score = self._calculate_risk_management_score(metrics)
        efficiency_score = self._calculate_efficiency_score(metrics)
        
        # Calculate overall score (weighted average)
        overall_score = (
            profitability_score * 0.40 +
            consistency_score * 0.25 +
            risk_management_score * 0.20 +
            efficiency_score * 0.15
        )
        
        # Convert to grades
        profitability_grade = self._score_to_grade(profitability_score)
        consistency_grade = self._score_to_grade(consistency_score)
        risk_management_grade = self._score_to_grade(risk_management_score)
        efficiency_grade = self._score_to_grade(efficiency_score)
        overall_grade = self._score_to_grade(overall_score)
        
        # Generate insights
        strengths, weaknesses, recommendations = self._generate_insights(
            metrics, overall_score
        )
        
        health_score = HealthScore(
            overall_grade=overall_grade,
            overall_score=overall_score,
            profitability_score=profitability_score,
            consistency_score=consistency_score,
            risk_management_score=risk_management_score,
            efficiency_score=efficiency_score,
            profitability_grade=profitability_grade,
            consistency_grade=consistency_grade,
            risk_management_grade=risk_management_grade,
            efficiency_grade=efficiency_grade,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            metrics=metrics,
        )
        
        logger.info(
            f"✓ Model evaluated: grade={overall_grade.value} "
            f"score={overall_score:.1f} | "
            f"Profitability={profitability_grade.value} "
            f"Consistency={consistency_grade.value} "
            f"RiskMgmt={risk_management_grade.value} "
            f"Efficiency={efficiency_grade.value}"
        )
        
        return health_score
    
    def _generate_insights(
        self,
        metrics: PerformanceMetrics,
        overall_score: float,
    ) -> tuple:
        """
        Generate strengths, weaknesses, and recommendations.
        
        Returns:
            (strengths_list, weaknesses_list, recommendations_list)
        """
        strengths = []
        weaknesses = []
        recommendations = []
        
        # Profitability insights
        if metrics.win_rate > 0.60:
            strengths.append(f"Excellent win rate: {metrics.win_rate*100:.1f}%")
        elif metrics.win_rate < 0.50:
            weaknesses.append(f"Low win rate: {metrics.win_rate*100:.1f}% (target >55%)")
            recommendations.append("Review entry logic - increase entry confirmation filters")
        
        if metrics.profit_factor > 2.0:
            strengths.append(f"Outstanding profit factor: {metrics.profit_factor:.2f}")
        elif metrics.profit_factor < 1.0:
            weaknesses.append(f"Negative profit factor: {metrics.profit_factor:.2f} (losing money)")
            recommendations.append("Increase stop loss discipline or review strategy logic")
        
        # Risk management insights
        if metrics.max_drawdown_percent < 0.15:
            strengths.append(f"Well-controlled drawdown: {metrics.max_drawdown_percent*100:.1f}%")
        elif metrics.max_drawdown_percent > 0.40:
            weaknesses.append(f"Excessive drawdown: {metrics.max_drawdown_percent*100:.1f}% (limit to 20%)")
            recommendations.append("Reduce position size or improve stop loss timing")
        
        # Consistency insights
        if metrics.sharpe_ratio > 1.5:
            strengths.append(f"Strong risk-adjusted return: Sharpe={metrics.sharpe_ratio:.2f}")
        elif metrics.sharpe_ratio < 0.5:
            weaknesses.append(f"Poor risk-adjusted return: Sharpe={metrics.sharpe_ratio:.2f}")
            recommendations.append("Increase return consistency or reduce volatility")
        
        # Expectancy insights
        if metrics.expectancy > 0:
            strengths.append(f"Positive expectancy: ${metrics.expectancy:.2f} per trade")
        else:
            weaknesses.append(f"Negative expectancy: ${metrics.expectancy:.2f} per trade")
            recommendations.append("Fundamental strategy issue - expect losses per trade")
        
        # Overall readiness
        if overall_score >= 80:
            recommendations.append("✓ Ready for live trading with proper position sizing")
        elif overall_score >= 60:
            recommendations.append("⚠ Suitable for further optimization before live trading")
        else:
            recommendations.append("✗ Not recommended for live trading - needs major improvements")
        
        return strengths, weaknesses, recommendations
    
    def generate_report(self, health_score: HealthScore) -> str:
        """
        Generate human-readable health report.
        
        Args:
            health_score: HealthScore object
            
        Returns:
            Formatted report string
        """
        report = f"""
╔════════════════════════════════════════════════════════════════╗
║           MODEL HEALTH REPORT - Grade: {health_score.overall_grade.value:^2}            ║
║                   Score: {health_score.overall_score:>5.1f}/100                    ║
╚════════════════════════════════════════════════════════════════╝

COMPONENT SCORES:
├─ Profitability:      {health_score.profitability_score:>5.1f}/100 [{health_score.profitability_grade.value:^3}]
├─ Consistency:        {health_score.consistency_score:>5.1f}/100 [{health_score.consistency_grade.value:^3}]
├─ Risk Management:    {health_score.risk_management_score:>5.1f}/100 [{health_score.risk_management_grade.value:^3}]
└─ Efficiency:         {health_score.efficiency_score:>5.1f}/100 [{health_score.efficiency_grade.value:^3}]

KEY METRICS:
├─ Total Trades:       {health_score.metrics.total_trades}
├─ Win Rate:           {health_score.metrics.win_rate*100:.1f}%
├─ Profit Factor:      {health_score.metrics.profit_factor:.2f}
├─ Sharpe Ratio:       {health_score.metrics.sharpe_ratio:.2f}
├─ Max Drawdown:       {health_score.metrics.max_drawdown_percent*100:.1f}%
└─ Expectancy:         ${health_score.metrics.expectancy:.2f}

STRENGTHS:
"""
        for s in health_score.strengths:
            report += f"  ✓ {s}\n"
        
        report += "\nWEAKNESSES:\n"
        for w in health_score.weaknesses:
            report += f"  ✗ {w}\n"
        
        report += "\nRECOMMENDATIONS:\n"
        for r in health_score.recommendations:
            report += f"  → {r}\n"
        
        report += "\n" + "="*64 + "\n"
        
        return report

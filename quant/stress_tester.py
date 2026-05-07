# ============================================================================
# STRESS TESTER - L4 QUANTITATIVE LAYER ORCHESTRATOR
# Master integrator: Runs all checks before trade execution
# ============================================================================

import logging
import numpy as np
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
from datetime import datetime

from .monte_carlo import MonteCarlo, MonteCarloResult
from .black_scholes import BlackScholes, BlackScholesResult
from .probability_scorer import ProbabilityScorer, ProbabilityScore
from .edge_calculator import EdgeCalculator, EdgeAnalysis

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class StressTestResult:
    """
    Complete pre-trade stress test result.
    
    Attributes:
        is_approved: Boolean - can the trade execute?
        approval_reason: Explanation of approval/rejection
        monte_carlo_result: Monte Carlo analysis
        black_scholes_result: Option pricing analysis
        probability_score: Final confidence score
        edge_analysis: Statistical edge validation
        risk_metrics: Dict with all risk metrics
        timestamp: When test was performed
    """
    is_approved: bool
    approval_reason: str
    monte_carlo_result: Optional[MonteCarloResult] = None
    black_scholes_result: Optional[BlackScholesResult] = None
    probability_score: Optional[ProbabilityScore] = None
    edge_analysis: Optional[EdgeAnalysis] = None
    risk_metrics: Dict = None
    timestamp: datetime = None


class StressTester:
    """
    Master Pre-Trade Stress Tester - All Approvals Integrated.
    
    Orchestrates the complete quantitative validation pipeline:
    1. Monte Carlo: Does setup have positive probability?
    2. Black-Scholes: What's the IV environment telling us?
    3. Probability Scorer: Combined confidence score
    4. Edge Calculator: Does signal have statistical edge?
    5. Final Approval: All gates must pass
    
    Trade is REJECTED if ANY gate fails:
    - Probability score < 65%
    - Monte Carlo win rate < 52%
    - Risk/reward < 1.5:1
    - Historical edge not validated
    - Conviction level < 60%
    
    Philosophy: "Don't risk capital on uncertain setups"
    """
    
    def __init__(
        self,
        prob_score_threshold: float = 65.0,
        mc_winrate_threshold: float = 0.52,
        min_rr_ratio: float = 1.5,
        min_conviction: float = 0.60,
        num_simulations: int = 1000
    ):
        """
        Initialize the Stress Tester with validation thresholds.
        
        Args:
            prob_score_threshold: Minimum probability score to trade (65 = 65%)
            mc_winrate_threshold: Minimum MC win rate (0.52 = 52%)
            min_rr_ratio: Minimum risk/reward ratio (1.5:1)
            min_conviction: Minimum signal conviction (0.60 = 60%)
            num_simulations: Monte Carlo simulations to run
        """
        self.prob_score_threshold = prob_score_threshold
        self.mc_winrate_threshold = mc_winrate_threshold
        self.min_rr_ratio = min_rr_ratio
        self.min_conviction = min_conviction
        
        # Initialize sub-engines
        self.monte_carlo = MonteCarlo(num_simulations=num_simulations)
        self.black_scholes = BlackScholes()
        self.probability_scorer = ProbabilityScorer(
            min_score_threshold=prob_score_threshold
        )
        self.edge_calculator = EdgeCalculator()
        
        logger.info("✓ StressTester initialized")
        logger.info(f"  - Probability Score Threshold: {prob_score_threshold}%")
        logger.info(f"  - MC Win Rate Threshold: {mc_winrate_threshold:.0%}")
        logger.info(f"  - Min R/R Ratio: {min_rr_ratio}:1")
        logger.info(f"  - Min Conviction: {min_conviction:.0%}")
    
    # ========================================================================
    # VALIDATION GATES
    # ========================================================================
    
    def _check_probability_score(self, prob_score: ProbabilityScore) -> Tuple[bool, str]:
        """
        Gate 1: Probability Score Validation.
        
        Rejects if score < threshold or conviction too low.
        """
        if prob_score.score < self.prob_score_threshold:
            return False, (
                f"Probability score {prob_score.score:.1f}% < "
                f"threshold {self.prob_score_threshold:.0f}%"
            )
        
        if prob_score.conviction_level < self.min_conviction:
            return False, (
                f"Signal conviction {prob_score.conviction_level:.0%} < "
                f"minimum {self.min_conviction:.0%}"
            )
        
        return True, f"✓ Probability score {prob_score.score:.1f}% passes"
    
    def _check_monte_carlo(self, mc_result: MonteCarloResult) -> Tuple[bool, str]:
        """
        Gate 2: Monte Carlo Validation.
        
        Rejects if win rate too low or expected return negative.
        """
        if mc_result.win_rate < (self.mc_winrate_threshold * 100):
            return False, (
                f"MC win rate {mc_result.win_rate:.1f}% < "
                f"threshold {self.mc_winrate_threshold*100:.0f}%"
            )
        
        if mc_result.probability_adjusted_return < 0:
            return False, (
                f"MC expected return {mc_result.probability_adjusted_return:.2f}% "
                f"is negative - losing setup"
            )
        
        if mc_result.max_drawdown < -25.0:
            return False, (
                f"Maximum drawdown {mc_result.max_drawdown:.1f}% exceeds -25% limit"
            )
        
        return True, (
            f"✓ MC validates: WinRate {mc_result.win_rate:.1f}%, "
            f"PF {mc_result.profit_factor:.2f}, "
            f"Sharpe {mc_result.sharpe_ratio:.2f}"
        )
    
    def _check_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float
    ) -> Tuple[bool, str]:
        """
        Gate 3: Risk/Reward Geometry Validation.
        
        Rejects if R/R ratio below minimum.
        """
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk <= 0:
            return False, "Risk is zero (SL = Entry) - invalid setup"
        
        rr_ratio = reward / risk
        
        if rr_ratio < self.min_rr_ratio:
            return False, (
                f"R/R ratio {rr_ratio:.2f}:1 < "
                f"minimum {self.min_rr_ratio:.2f}:1"
            )
        
        return True, f"✓ R/R geometry valid: {rr_ratio:.2f}:1"
    
    def _check_edge(
        self,
        historical_trades: Optional[List[Dict]] = None
    ) -> Tuple[bool, str]:
        """
        Gate 4: Historical Edge Validation.
        
        Optional gate - if historical data available, verify edge exists.
        If no historical data, skip this gate.
        """
        if not historical_trades or len(historical_trades) == 0:
            return True, "⊘ No historical trades provided - skipping edge validation"
        
        edge_analysis = self.edge_calculator.analyze_edge(historical_trades)
        
        if not edge_analysis.has_edge:
            return False, (
                f"No statistical edge detected: "
                f"{edge_analysis.rationale.split(chr(10))[0]}"
            )
        
        return True, (
            f"✓ Strategy has edge: {edge_analysis.edge_percentage:.1f}% "
            f"above random"
        )
    
    # ========================================================================
    # MAIN STRESS TEST ORCHESTRATION
    # ========================================================================
    
    def run_stress_test(
        self,
        # Trade Setup
        current_price: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        
        # Signals
        ai_ensemble_probability: float,
        volatility: float,
        
        # Market Data
        iv_percentile: Optional[float] = None,
        historical_trades: Optional[List[Dict]] = None,
        
        # Options
        num_days: int = 5,
        expected_return: float = 0.0
    ) -> StressTestResult:
        """
        Execute complete pre-trade stress test.
        
        Runs all 4 validation gates. Trade is approved ONLY if ALL gates pass.
        
        Args:
            current_price: Current market price
            entry_price: Proposed entry price
            stop_loss: Stop loss level
            take_profit: Take profit level
            ai_ensemble_probability: AI model confidence 0-1
            volatility: Current market volatility (0.2 = 20%)
            iv_percentile: IV percentile 0-100 (optional)
            historical_trades: List of past trades for edge validation (optional)
            num_days: Days to hold trade
            expected_return: Expected daily return bias
            
        Returns:
            StressTestResult with approval/rejection and full breakdown
        """
        logger.info("=" * 80)
        logger.info("🚀 STARTING PRE-TRADE STRESS TEST")
        logger.info("=" * 80)
        
        results = {}
        gates_passed = []
        gates_failed = []
        
        # ====================================================================
        # GATE 1: MONTE CARLO SIMULATION
        # ====================================================================
        
        logger.info("\n[GATE 1/4] Running Monte Carlo Simulation...")
        try:
            mc_result = self.monte_carlo.simulate(
                current_price=current_price,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                volatility=volatility,
                expected_return=expected_return,
                num_days=num_days
            )
            results['monte_carlo'] = mc_result
            
            mc_pass, mc_reason = self._check_monte_carlo(mc_result)
            if mc_pass:
                gates_passed.append(mc_reason)
                logger.info(f"✓ {mc_reason}")
            else:
                gates_failed.append(mc_reason)
                logger.warning(f"✗ {mc_reason}")
        
        except Exception as e:
            logger.error(f"✗ Monte Carlo failed: {e}")
            gates_failed.append(f"Monte Carlo error: {str(e)}")
            mc_result = None
            mc_pass = False
        
        # ====================================================================
        # GATE 2: BLACK-SCHOLES ANALYSIS
        # ====================================================================
        
        logger.info("[GATE 2/4] Analyzing IV with Black-Scholes...")
        try:
            # Use at-the-money option pricing
            days_to_expiry = num_days / 252
            bs_result = self.black_scholes.price_option(
                spot_price=current_price,
                strike_price=current_price,  # ATM for analysis
                time_to_expiry=days_to_expiry,
                risk_free_rate=0.05,
                volatility=volatility
            )
            results['black_scholes'] = bs_result
            
            if iv_percentile is not None:
                sentiment = self.black_scholes.get_market_sentiment(iv_percentile)
                logger.info(f"✓ IV Sentiment: {sentiment['sentiment']} ({iv_percentile:.0f}th percentile)")
            else:
                logger.info("⊘ No IV data provided - skipping sentiment analysis")
        
        except Exception as e:
            logger.error(f"✗ Black-Scholes failed: {e}")
            bs_result = None
        
        # ====================================================================
        # GATE 3: PROBABILITY SCORING
        # ====================================================================
        
        logger.info("[GATE 3/4] Calculating Probability Score...")
        try:
            prob_score = self.probability_scorer.calculate_probability_score(
                ai_ensemble_probability=ai_ensemble_probability,
                monte_carlo_win_rate=mc_result.win_rate if mc_result else 50,
                monte_carlo_profit_factor=mc_result.profit_factor if mc_result else 1.0,
                monte_carlo_sharpe=mc_result.sharpe_ratio if mc_result else 0.5,
                iv_percentile=iv_percentile or 50,
                entry_price=entry_price,
                stop_loss=stop_loss,
                take_profit=take_profit
            )
            results['probability_score'] = prob_score
            
            for line in prob_score.rationale:
                logger.info(f"  {line}")
            
            prob_pass, prob_reason = self._check_probability_score(prob_score)
            if prob_pass:
                gates_passed.append(prob_reason)
                logger.info(f"✓ {prob_reason}")
            else:
                gates_failed.append(prob_reason)
                logger.warning(f"✗ {prob_reason}")
        
        except Exception as e:
            logger.error(f"✗ Probability scoring failed: {e}")
            gates_failed.append(f"Probability scoring error: {str(e)}")
            prob_score = None
            prob_pass = False
        
        # ====================================================================
        # GATE 4: RISK/REWARD VALIDATION
        # ====================================================================
        
        logger.info("[GATE 4/4] Validating Risk/Reward Geometry...")
        try:
            rr_pass, rr_reason = self._check_risk_reward(entry_price, stop_loss, take_profit)
            if rr_pass:
                gates_passed.append(rr_reason)
                logger.info(f"✓ {rr_reason}")
            else:
                gates_failed.append(rr_reason)
                logger.warning(f"✗ {rr_reason}")
        except Exception as e:
            logger.error(f"✗ Risk/Reward check failed: {e}")
            gates_failed.append(f"R/R validation error: {str(e)}")
            rr_pass = False
        
        # ====================================================================
        # OPTIONAL GATE: HISTORICAL EDGE VALIDATION
        # ====================================================================
        
        logger.info("[OPTIONAL] Checking Historical Edge...")
        try:
            edge_pass, edge_reason = self._check_edge(historical_trades)
            logger.info(f"  {edge_reason}")
            
            if historical_trades and len(historical_trades) > 0:
                edge_analysis = self.edge_calculator.analyze_edge(historical_trades)
                results['edge_analysis'] = edge_analysis
                if not edge_pass:
                    gates_failed.append(edge_reason)
        except Exception as e:
            logger.warning(f"⚠️  Edge analysis error: {e}")
        
        # ====================================================================
        # FINAL APPROVAL DECISION
        # ====================================================================
        
        is_approved = (len(gates_failed) == 0)
        
        # Compile approval reason
        if is_approved:
            approval_reason = "✓✓✓ ALL GATES PASSED - TRADE APPROVED ✓✓✓\n"
            approval_reason += "\n".join(gates_passed)
        else:
            approval_reason = "✗✗✗ TRADE REJECTED - GATES FAILED ✗✗✗\n"
            approval_reason += "Failed checks:\n"
            approval_reason += "\n".join([f"  ✗ {reason}" for reason in gates_failed])
        
        # ====================================================================
        # COMPILE RISK METRICS
        # ====================================================================
        
        risk_metrics = {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'risk': abs(entry_price - stop_loss),
            'reward': abs(take_profit - entry_price),
            'rr_ratio': abs(take_profit - entry_price) / abs(entry_price - stop_loss),
            'volatility': volatility,
            'holding_days': num_days,
            'ai_confidence': ai_ensemble_probability * 100
        }
        
        # Add optional metrics
        if mc_result:
            risk_metrics['mc_win_rate'] = mc_result.win_rate
            risk_metrics['mc_sharpe'] = mc_result.sharpe_ratio
            risk_metrics['expected_return'] = mc_result.probability_adjusted_return
            risk_metrics['max_drawdown'] = mc_result.max_drawdown
        
        if prob_score:
            risk_metrics['final_confidence'] = prob_score.score
            risk_metrics['conviction'] = prob_score.conviction_level * 100
        
        # ====================================================================
        # LOG FINAL DECISION
        # ====================================================================
        
        logger.info("\n" + "=" * 80)
        logger.info(approval_reason)
        logger.info("=" * 80)
        
        return StressTestResult(
            is_approved=is_approved,
            approval_reason=approval_reason,
            monte_carlo_result=mc_result,
            black_scholes_result=bs_result,
            probability_score=prob_score,
            edge_analysis=results.get('edge_analysis'),
            risk_metrics=risk_metrics,
            timestamp=datetime.now()
        )

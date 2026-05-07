# ============================================================================
# PROBABILITY SCORER - L4 QUANTITATIVE LAYER
# Ingests AI ensemble vote, Monte Carlo, and IV data for final confidence
# ============================================================================

import logging
import numpy as np
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class ProbabilityScore:
    """
    Final probability score for a trade setup.
    
    Attributes:
        score: Final confidence 0-100%
        ai_ensemble_weight: AI signal confidence (0-100)
        monte_carlo_weight: Monte Carlo win probability (0-100)
        iv_sentiment_weight: IV sentiment bias (-50 to +50)
        risk_reward_weight: Risk/reward ratio component
        conviction_level: Agreement between all signals 0-1
        individual_scores: Dict of all component scores
        rationale: Text explanation of the score
        timestamp: When calculated
    """
    score: float
    ai_ensemble_weight: float
    monte_carlo_weight: float
    iv_sentiment_weight: float
    risk_reward_weight: float
    conviction_level: float
    individual_scores: Dict
    rationale: List[str]
    timestamp: datetime


class ProbabilityScorer:
    """
    Probability Scoring Engine - Combines Multiple Signals.
    
    Ingests signals from:
    1. AI/ML Ensemble: Multiple algo votes (probability 0-1)
    2. Monte Carlo: Historical win probability from simulations
    3. IV Sentiment: Fear/greed indicator from option IV
    4. Risk/Reward: Trade geometry (reward/risk ratio)
    
    Outputs single 0-100% confidence score.
    
    Weighting philosophy (tunable):
    - 40% AI Ensemble (primary signal generator)
    - 35% Monte Carlo (historical validation)
    - 15% IV Sentiment (macro context)
    - 10% Risk/Reward (edge protection)
    """
    
    def __init__(
        self,
        ai_weight: float = 0.40,
        mc_weight: float = 0.35,
        iv_weight: float = 0.15,
        rr_weight: float = 0.10,
        min_score_threshold: float = 65.0
    ):
        """
        Initialize Probability Scorer.
        
        Args:
            ai_weight: Weight for AI ensemble signals
            mc_weight: Weight for Monte Carlo win rate
            iv_weight: Weight for IV sentiment
            rr_weight: Weight for risk/reward ratio
            min_score_threshold: Minimum score to consider signal valid
        """
        # Validate weights sum to 1.0
        total_weight = ai_weight + mc_weight + iv_weight + rr_weight
        
        self.ai_weight = ai_weight / total_weight
        self.mc_weight = mc_weight / total_weight
        self.iv_weight = iv_weight / total_weight
        self.rr_weight = rr_weight / total_weight
        
        self.min_score_threshold = min_score_threshold
        
        logger.info(f"✓ ProbabilityScorer initialized")
        logger.info(f"  - AI Weight: {self.ai_weight:.1%}")
        logger.info(f"  - MC Weight: {self.mc_weight:.1%}")
        logger.info(f"  - IV Weight: {self.iv_weight:.1%}")
        logger.info(f"  - RR Weight: {self.rr_weight:.1%}")
    
    # ========================================================================
    # SIGNAL NORMALIZATION
    # ========================================================================
    
    def normalize_ai_ensemble(self, ensemble_probability: float) -> float:
        """
        Normalize AI ensemble signal to 0-100 confidence score.
        
        AI model outputs probability 0-1, convert to 0-100 confidence.
        
        Args:
            ensemble_probability: Model output 0-1 (0% to 100%)
            
        Returns:
            Normalized confidence 0-100
        """
        # Clip to valid range
        prob = np.clip(ensemble_probability, 0.0, 1.0)
        
        # Convert to 0-100
        score = prob * 100
        
        return score
    
    def normalize_monte_carlo(
        self,
        win_rate: float,
        profit_factor: float,
        sharpe_ratio: float
    ) -> float:
        """
        Normalize Monte Carlo metrics to 0-100 confidence.
        
        Weights:
        - Win rate: 50% (how often we win)
        - Profit factor: 30% (reward/risk balance)
        - Sharpe: 20% (risk-adjusted returns)
        
        Args:
            win_rate: % wins (0-100)
            profit_factor: Reward/risk (typically 0.5-3.0)
            sharpe_ratio: Risk-adjusted return (-3 to +3 typical)
            
        Returns:
            Normalized score 0-100
        """
        # Normalize win rate (0-100 -> 0-100, but capped at 95 for safety)
        win_score = min(win_rate, 95.0)
        
        # Normalize profit factor
        # PF of 1.5 = 60 score, PF of 2.0 = 75 score, PF of 3.0 = 100 score
        pf_score = np.clip((profit_factor - 1.0) / 2.0 * 100, 0, 100)
        
        # Normalize Sharpe ratio
        # Sharpe 0.0 = 50, Sharpe 1.0 = 70, Sharpe 2.0 = 90
        sharpe_score = 50 + (sharpe_ratio * 10)
        sharpe_score = np.clip(sharpe_score, 20, 100)
        
        # Weighted average
        mc_score = (
            win_score * 0.5 +
            pf_score * 0.3 +
            sharpe_score * 0.2
        )
        
        return mc_score
    
    def normalize_iv_sentiment(
        self,
        iv_percentile: float,
        directional_bias: str = 'neutral'
    ) -> float:
        """
        Normalize IV sentiment to -50 to +50 signal.
        
        - Extreme fear (IV at 1-year highs): -50 (contrarian bullish)
        - Extreme greed (IV at lows): +50 (contrarian bearish)
        - Neutral: 0
        
        Args:
            iv_percentile: IV percentile 0-100
            directional_bias: Trade direction ('long', 'short', 'neutral')
            
        Returns:
            IV signal -50 to +50
        """
        # Map IV percentile to fear/greed
        # Low IV (0-20) = greed = contrarian bearish = +50
        # High IV (80-100) = fear = contrarian bullish = -50
        
        if iv_percentile < 20:
            # Extreme greed
            iv_signal = 50.0
        elif iv_percentile < 40:
            # Moderate greed
            iv_signal = 25.0
        elif iv_percentile > 80:
            # Extreme fear
            iv_signal = -50.0
        elif iv_percentile > 60:
            # Moderate fear
            iv_signal = -25.0
        else:
            # Neutral (40-60 range)
            iv_signal = 0.0
        
        return iv_signal
    
    def normalize_risk_reward(
        self,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        min_rr_ratio: float = 1.5
    ) -> float:
        """
        Normalize risk/reward geometry to 0-100 score.
        
        Good risk/reward = Score increases.
        Bad risk/reward = Score decreases (but doesn't negate signal).
        
        Args:
            entry_price: Trade entry level
            stop_loss: Stop loss level
            take_profit: Take profit level
            min_rr_ratio: Minimum acceptable R/R ratio (1.5:1 = 1.5)
            
        Returns:
            R/R score 0-100
        """
        # Calculate distances
        risk = abs(entry_price - stop_loss)
        reward = abs(take_profit - entry_price)
        
        if risk <= 0:
            return 50.0  # Default if SL = entry
        
        # Calculate actual R/R ratio
        rr_ratio = reward / risk
        
        if rr_ratio < 1.0:
            # Negative R/R - losing setup
            return 20.0
        elif rr_ratio < min_rr_ratio:
            # Below minimum - mediocre
            rr_score = 40 + (rr_ratio / min_rr_ratio) * 30
        elif rr_ratio >= 3.0:
            # Excellent - 3:1 or better
            return 100.0
        else:
            # Linear between min_rr and 3.0
            rr_score = 70 + ((rr_ratio - min_rr_ratio) / (3.0 - min_rr_ratio)) * 30
        
        return np.clip(rr_score, 20, 100)
    
    # ========================================================================
    # CONVICTION CALCULATION
    # ========================================================================
    
    def calculate_conviction(self, individual_scores: Dict) -> float:
        """
        Calculate conviction level based on signal agreement.
        
        Conviction = 1.0 if all signals agree
        Conviction = lower if signals contradict
        
        Args:
            individual_scores: Dict of normalized scores from all components
            
        Returns:
            Conviction 0-1 (1 = perfect alignment, 0 = contradiction)
        """
        scores = np.array(list(individual_scores.values()))
        
        if len(scores) == 0:
            return 0.5
        
        # Standard deviation of scores
        score_std = np.std(scores)
        
        # Conviction inversely related to disagreement
        # std=0 -> conviction=1.0
        # std=20 -> conviction~0.7
        # std=50 -> conviction~0.2
        
        conviction = 1.0 / (1.0 + (score_std / 20.0))
        
        return np.clip(conviction, 0.0, 1.0)
    
    # ========================================================================
    # COMPOSITE SCORING
    # ========================================================================
    
    def calculate_probability_score(
        self,
        ai_ensemble_probability: float,
        monte_carlo_win_rate: float,
        monte_carlo_profit_factor: float,
        monte_carlo_sharpe: float,
        iv_percentile: float,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        min_rr_ratio: float = 1.5
    ) -> ProbabilityScore:
        """
        Calculate composite probability score from all signals.
        
        Main formula:
        Score = (ai * 0.40) + (mc * 0.35) + ((iv + 50) * 0.15 / 100) + (rr * 0.10)
        
        All components normalized to 0-100, then weighted.
        
        Args:
            ai_ensemble_probability: AI model output 0-1
            monte_carlo_win_rate: % wins from simulation
            monte_carlo_profit_factor: Win/loss ratio
            monte_carlo_sharpe: Risk-adjusted returns
            iv_percentile: IV percentile 0-100
            entry_price: Trade entry level
            stop_loss: Stop loss level
            take_profit: Take profit level
            min_rr_ratio: Minimum R/R threshold
            
        Returns:
            ProbabilityScore with full breakdown
        """
        # ====================================================================
        # NORMALIZE EACH SIGNAL TO 0-100
        # ====================================================================
        
        ai_score = self.normalize_ai_ensemble(ai_ensemble_probability)
        
        mc_score = self.normalize_monte_carlo(
            monte_carlo_win_rate,
            monte_carlo_profit_factor,
            monte_carlo_sharpe
        )
        
        iv_signal = self.normalize_iv_sentiment(iv_percentile)
        # Convert -50 to +50 to 0 to 100 scale
        iv_score = (iv_signal + 50)  # Now 0-100
        
        rr_score = self.normalize_risk_reward(
            entry_price, stop_loss, take_profit, min_rr_ratio
        )
        
        # ====================================================================
        # CALCULATE CONVICTION
        # ====================================================================
        
        individual_scores = {
            'ai': ai_score,
            'monte_carlo': mc_score,
            'iv_sentiment': iv_score,
            'risk_reward': rr_score
        }
        
        conviction = self.calculate_conviction(individual_scores)
        
        # ====================================================================
        # WEIGHTED COMPOSITE SCORE
        # ====================================================================
        
        composite_score = (
            (ai_score * self.ai_weight) +
            (mc_score * self.mc_weight) +
            (iv_score * self.iv_weight) +
            (rr_score * self.rr_weight)
        )
        
        # Apply conviction damping (weak conviction = lower final score)
        # Conviction curve: 0.5 conviction = -10% penalty, 1.0 = no penalty
        conviction_modifier = (conviction * 0.5) + 0.5  # Range 0.5 to 1.0
        final_score = composite_score * conviction_modifier
        
        # ====================================================================
        # GENERATE RATIONALE
        # ====================================================================
        
        rationale = []
        
        if ai_score >= 75:
            rationale.append(f"✓ Strong AI ensemble signal ({ai_score:.0f})")
        elif ai_score >= 60:
            rationale.append(f"○ Moderate AI signal ({ai_score:.0f})")
        else:
            rationale.append(f"✗ Weak AI signal ({ai_score:.0f})")
        
        if mc_score >= 70:
            rationale.append(f"✓ Excellent Monte Carlo stats (WR:{monte_carlo_win_rate:.0f}%, PF:{monte_carlo_profit_factor:.2f})")
        elif mc_score >= 55:
            rationale.append(f"○ Acceptable Monte Carlo validation")
        else:
            rationale.append(f"✗ Poor Monte Carlo metrics")
        
        if iv_signal < -25:
            rationale.append(f"✓ Extreme fear environment - contrarian bullish")
        elif iv_signal > 25:
            rationale.append(f"✗ Extreme greed environment - contrarian bearish")
        else:
            rationale.append(f"○ Normal IV environment")
        
        if rr_score >= 80:
            rationale.append(f"✓ Excellent R/R geometry")
        elif rr_score >= 60:
            rationale.append(f"○ Acceptable R/R ratio")
        else:
            rationale.append(f"✗ Poor R/R ratio - risk > reward")
        
        if conviction >= 0.8:
            rationale.append(f"✓ All signals aligned - HIGH CONVICTION")
        elif conviction >= 0.6:
            rationale.append(f"○ Moderate signal agreement")
        else:
            rationale.append(f"✗ Signals conflicting - LOW CONVICTION")
        
        return ProbabilityScore(
            score=final_score,
            ai_ensemble_weight=ai_score,
            monte_carlo_weight=mc_score,
            iv_sentiment_weight=iv_signal,
            risk_reward_weight=rr_score,
            conviction_level=conviction,
            individual_scores=individual_scores,
            rationale=rationale,
            timestamp=datetime.now()
        )
    
    # ========================================================================
    # SIGNAL VALIDATION
    # ========================================================================
    
    def is_valid_signal(self, prob_score: ProbabilityScore) -> Tuple[bool, str]:
        """
        Determine if probability score is above acceptance threshold.
        
        Args:
            prob_score: ProbabilityScore result
            
        Returns:
            Tuple of (is_valid, reason_text)
        """
        if prob_score.score >= self.min_score_threshold:
            reason = f"Score {prob_score.score:.1f}% >= threshold {self.min_score_threshold:.0f}%"
            return True, reason
        else:
            reason = f"Score {prob_score.score:.1f}% < threshold {self.min_score_threshold:.0f}% - REJECTED"
            return False, reason
    
    # ========================================================================
    # SIMPLIFIED BACKTEST SCORING
    # ========================================================================
    
    def score_setup(
        self,
        ai_signal: int,
        ai_confidence: float,
        regime: int,
        momentum: float,
        price: float,
        symbol: str
    ) -> float:
        """
        Simplified probability scoring for backtest.
        
        Returns a single 0-1 confidence score based on:
        - AI signal direction and confidence
        - Market regime (trending=1, ranging=0)
        - Momentum direction
        
        Args:
            ai_signal: Trade direction (-1/0/1)
            ai_confidence: AI model confidence (0-1)
            regime: Market regime (0=ranging, 1=trending)
            momentum: Momentum value (positive=up, negative=down)
            price: Current price
            symbol: Symbol being traded
            
        Returns:
            Probability score 0-1 (0.78 = 78% confidence)
        """
        if ai_signal == 0:
            return 0.0
        
        # Base score from AI confidence
        base_score = ai_confidence
        
        # Regime boost: +10% if trending, -5% if ranging
        regime_multiplier = 1.10 if regime == 1 else 0.95
        
        # Momentum confirmation: +5% if aligned with signal
        momentum_aligned = (
            (ai_signal == 1 and momentum > 0) or
            (ai_signal == -1 and momentum < 0)
        )
        momentum_multiplier = 1.05 if momentum_aligned else 1.0
        
        # Final probability
        final_score = base_score * regime_multiplier * momentum_multiplier
        
        return np.clip(final_score, 0.0, 1.0)

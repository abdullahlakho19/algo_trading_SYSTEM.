# ============================================================================
# BLACK-SCHOLES & IMPLIED VOLATILITY - L4 QUANTITATIVE LAYER
# Calculates option pricing, IV to measure market fear/greed
# ============================================================================

import logging
import numpy as np
from scipy import stats
from scipy.optimize import brentq
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class BlackScholesResult:
    """
    Black-Scholes option pricing result.
    
    Attributes:
        call_price: European call option price
        put_price: European put option price
        call_delta: Call option delta (rate of price change)
        put_delta: Put option delta
        call_gamma: Call option gamma (delta acceleration)
        put_gamma: Put option gamma
        call_vega: Call option vega (IV sensitivity)
        put_vega: Put option vega
        call_theta: Call option theta (time decay)
        put_theta: Put option theta
        call_rho: Call option rho (interest rate sensitivity)
        put_rho: Put option rho
        theoretical_price: Fair value based on option
    """
    call_price: float
    put_price: float
    call_delta: float
    put_delta: float
    call_gamma: float
    put_gamma: float
    call_vega: float
    put_vega: float
    call_theta: float
    put_theta: float
    call_rho: float
    put_rho: float
    theoretical_price: float = 0.0


class BlackScholes:
    """
    Black-Scholes Option Pricing & Implied Volatility Engine.
    
    Uses scipy.stats.norm for mathematical precision:
    - Calculates option Greeks (Delta, Gamma, Vega, Theta, Rho)
    - Extracts Implied Volatility from option prices
    - Measures market fear (IV rank) and greed
    - Provides theoretical fair value for options
    
    Mathematical foundation:
    - Call = S*N(d1) - K*e^(-rt)*N(d2)
    - Put = K*e^(-rt)*N(-d2) - S*N(-d1)
    - Where N(x) is the cumulative normal distribution
    """
    
    def __init__(self):
        """Initialize Black-Scholes calculator."""
        logger.info("✓ BlackScholes initialized")
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    def calculate_d1_d2(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float
    ) -> Tuple[float, float]:
        """
        Calculate d1 and d2 parameters for Black-Scholes.
        
        d1 = [ln(S/K) + (r + σ²/2)*T] / (σ*√T)
        d2 = d1 - σ*√T
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate (annual)
            volatility: Annualized volatility
            
        Returns:
            Tuple of (d1, d2)
        """
        if time_to_expiry <= 0 or volatility <= 0:
            d1 = 0.0
            d2 = 0.0
        else:
            sqrt_t = np.sqrt(time_to_expiry)
            
            d1 = (
                np.log(spot_price / strike_price) +
                (risk_free_rate + 0.5 * volatility ** 2) * time_to_expiry
            ) / (volatility * sqrt_t)
            
            d2 = d1 - volatility * sqrt_t
        
        return d1, d2
    
    # ========================================================================
    # OPTION PRICING
    # ========================================================================
    
    def call_price(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float
    ) -> float:
        """
        Calculate European call option price using Black-Scholes.
        
        C = S*N(d1) - K*e^(-rt)*N(d2)
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            volatility: Annualized volatility
            
        Returns:
            Call option price
        """
        d1, d2 = self.calculate_d1_d2(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility
        )
        
        call = (
            spot_price * stats.norm.cdf(d1) -
            strike_price * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(d2)
        )
        
        return call
    
    def put_price(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float
    ) -> float:
        """
        Calculate European put option price using Black-Scholes.
        
        P = K*e^(-rt)*N(-d2) - S*N(-d1)
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            volatility: Annualized volatility
            
        Returns:
            Put option price
        """
        d1, d2 = self.calculate_d1_d2(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility
        )
        
        put = (
            strike_price * np.exp(-risk_free_rate * time_to_expiry) * stats.norm.cdf(-d2) -
            spot_price * stats.norm.cdf(-d1)
        )
        
        return put
    
    # ========================================================================
    # GREEKS CALCULATION
    # ========================================================================
    
    def calculate_greeks(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float
    ) -> Dict[str, float]:
        """
        Calculate all option Greeks (sensitivities).
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            volatility: Annualized volatility
            
        Returns:
            Dict with all Greeks
        """
        d1, d2 = self.calculate_d1_d2(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility
        )
        
        sqrt_t = np.sqrt(time_to_expiry) if time_to_expiry > 0 else 1e-9
        
        # Common terms
        pdf_d1 = stats.norm.pdf(d1)
        exp_rt = np.exp(-risk_free_rate * time_to_expiry)
        
        # ==================================================================
        # DELTA: Rate of change of option price with respect to stock price
        # ==================================================================
        call_delta = stats.norm.cdf(d1)
        put_delta = call_delta - 1
        
        # ==================================================================
        # GAMMA: Rate of change of delta with respect to stock price
        # ==================================================================
        gamma = pdf_d1 / (spot_price * volatility * sqrt_t)
        
        # ==================================================================
        # VEGA: Sensitivity to volatility (per 1% change in IV)
        # ==================================================================
        vega = spot_price * pdf_d1 * sqrt_t / 100
        
        # ==================================================================
        # THETA: Time decay per day
        # ==================================================================
        call_theta = (
            -spot_price * pdf_d1 * volatility / (2 * sqrt_t) -
            risk_free_rate * strike_price * exp_rt * stats.norm.cdf(d2)
        ) / 365
        
        put_theta = (
            -spot_price * pdf_d1 * volatility / (2 * sqrt_t) +
            risk_free_rate * strike_price * exp_rt * stats.norm.cdf(-d2)
        ) / 365
        
        # ==================================================================
        # RHO: Sensitivity to interest rates (per 1% change)
        # ==================================================================
        call_rho = strike_price * time_to_expiry * exp_rt * stats.norm.cdf(d2) / 100
        put_rho = -strike_price * time_to_expiry * exp_rt * stats.norm.cdf(-d2) / 100
        
        return {
            'call_delta': call_delta,
            'put_delta': put_delta,
            'gamma': gamma,
            'vega': vega,
            'call_theta': call_theta,
            'put_theta': put_theta,
            'call_rho': call_rho,
            'put_rho': put_rho
        }
    
    # ========================================================================
    # FULL PRICING
    # ========================================================================
    
    def price_option(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        volatility: float
    ) -> BlackScholesResult:
        """
        Calculate complete option pricing and Greeks.
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            volatility: Annualized volatility
            
        Returns:
            BlackScholesResult with pricing and all Greeks
        """
        call = self.call_price(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility
        )
        put = self.put_price(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility
        )
        
        greeks = self.calculate_greeks(
            spot_price, strike_price, time_to_expiry, risk_free_rate, volatility
        )
        
        return BlackScholesResult(
            call_price=call,
            put_price=put,
            call_delta=greeks['call_delta'],
            put_delta=greeks['put_delta'],
            call_gamma=greeks['gamma'],
            put_gamma=greeks['gamma'],
            call_vega=greeks['vega'],
            put_vega=greeks['vega'],
            call_theta=greeks['call_theta'],
            put_theta=greeks['put_theta'],
            call_rho=greeks['call_rho'],
            put_rho=greeks['put_rho'],
            theoretical_price=(call + put) / 2
        )
    
    # ========================================================================
    # IMPLIED VOLATILITY CALCULATION
    # ========================================================================
    
    def implied_volatility_call(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        market_call_price: float,
        initial_guess: float = 0.2
    ) -> float:
        """
        Calculate implied volatility from option market price using bisection.
        
        Uses scipy.optimize.brentq for numerical root finding:
        Solves for σ such that BS(σ) = market_call_price
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            market_call_price: Observed market price of call
            initial_guess: Starting IV for bisection
            
        Returns:
            Implied volatility (as decimal, 0.2 = 20%)
        """
        def objective(vol):
            return (
                self.call_price(
                    spot_price, strike_price, time_to_expiry, risk_free_rate, vol
                ) - market_call_price
            )
        
        try:
            # Brent's method: robust root-finding algorithm
            # Problem: boundary conditions must bracket the root
            # Try to find bracket by expanding bounds
            lower_vol = 0.001
            upper_vol = 5.0  # 500% volatility (extreme)
            
            # Check if root is bracketed
            if objective(lower_vol) * objective(upper_vol) > 0:
                # Root not bracketed - return initial guess or NaN
                return np.nan
            
            iv = brentq(objective, lower_vol, upper_vol, xtol=1e-6)
            return iv
        except Exception as e:
            logger.warning(f"⚠️  IV calculation failed: {e}")
            return np.nan
    
    def implied_volatility_put(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        market_put_price: float,
        initial_guess: float = 0.2
    ) -> float:
        """
        Calculate implied volatility from put option market price.
        
        Args:
            spot_price: Current stock price
            strike_price: Option strike price
            time_to_expiry: Time to expiration in years
            risk_free_rate: Risk-free interest rate
            market_put_price: Observed market price of put
            initial_guess: Starting IV for bisection
            
        Returns:
            Implied volatility (as decimal)
        """
        def objective(vol):
            return (
                self.put_price(
                    spot_price, strike_price, time_to_expiry, risk_free_rate, vol
                ) - market_put_price
            )
        
        try:
            lower_vol = 0.001
            upper_vol = 5.0
            
            if objective(lower_vol) * objective(upper_vol) > 0:
                return np.nan
            
            iv = brentq(objective, lower_vol, upper_vol, xtol=1e-6)
            return iv
        except Exception as e:
            logger.warning(f"⚠️  IV calculation failed: {e}")
            return np.nan
    
    # ========================================================================
    # IV ANALYSIS FOR MARKET SENTIMENT
    # ========================================================================
    
    def calculate_iv_percentile(
        self,
        current_iv: float,
        iv_history: np.ndarray,
        lookback_periods: int = 252
    ) -> float:
        """
        Calculate IV percentile to measure market fear vs. greed.
        
        IV Percentile: % of historical IV values that are below current IV
        - Low percentile (0-20%) = Fear, opportunity to sell volatility
        - High percentile (80-100%) = Greed, opportunity to buy volatility
        
        Args:
            current_iv: Current implied volatility
            iv_history: Historical IV values (recent lookback)
            lookback_periods: Number of periods to use for percentile (252 = 1 year)
            
        Returns:
            IV percentile 0-100
        """
        iv_history = iv_history[-lookback_periods:] if len(iv_history) > lookback_periods else iv_history
        percentile = stats.percentileofscore(iv_history, current_iv)
        return percentile
    
    def calculate_iv_rank(
        self,
        current_iv: float,
        iv_high_52w: float,
        iv_low_52w: float
    ) -> float:
        """
        Calculate IV Rank: where current IV stands relative to 52-week range.
        
        IV Rank = (Current IV - 52w Low) / (52w High - 52w Low) * 100
        
        - Low rank (0-20%) = Volatility at 52-week lows (potential expansion)
        - High rank (80-100%) = Volatility at 52-week highs (potential contraction)
        
        Args:
            current_iv: Current IV level
            iv_high_52w: 52-week high IV
            iv_low_52w: 52-week low IV
            
        Returns:
            IV rank 0-100
        """
        if iv_high_52w <= iv_low_52w:
            return 50.0
        
        iv_rank = ((current_iv - iv_low_52w) / (iv_high_52w - iv_low_52w)) * 100
        return np.clip(iv_rank, 0.0, 100.0)
    
    def get_market_sentiment(self, iv_percentile: float) -> Dict[str, float]:
        """
        Interpret IV percentile into actionable market sentiment.
        
        Args:
            iv_percentile: IV percentile 0-100
            
        Returns:
            Dict with sentiment bias (-1 to +1 scale)
        """
        # Extreme fear (IV at 1-year highs): bullish contrarian signal
        if iv_percentile > 80:
            fear_score = 1.0  # Maximum fear = contrarian bullish
            greed_score = 0.0
            sentiment = "extreme_fear"
        # Above normal fear
        elif iv_percentile > 60:
            fear_score = (iv_percentile - 60) / 20
            greed_score = 0.0
            sentiment = "fear"
        # Below normal greed
        elif iv_percentile < 40:
            fear_score = 0.0
            greed_score = (40 - iv_percentile) / 40
            sentiment = "greed"
        # Above normal greed
        elif iv_percentile < 20:
            fear_score = 0.0
            greed_score = 1.0  # Maximum greed = contrarian bearish
            sentiment = "extreme_greed"
        else:
            fear_score = 0.0
            greed_score = 0.0
            sentiment = "neutral"
        
        # Convert to -1 to +1 bias (fear is bullish for contrarians)
        bias = fear_score - greed_score
        
        return {
            'sentiment': sentiment,
            'iv_fear_bias': fear_score,
            'iv_greed_bias': greed_score,
            'net_bias': bias,
            'percentile': iv_percentile
        }

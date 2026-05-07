# ============================================================================
# ENSEMBLE_VOTER.PY - AI/ML Consensus Engine
# Loads pre-trained models and performs weighted voting on market signals
# ============================================================================

import logging
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class EnsembleVote:
    """Result of ensemble voting."""
    direction: int  # 1 = bullish, 0 = bearish
    confidence: float  # 0-1 probability
    xgboost_vote: int
    rf_vote: int
    gb_vote: int
    xgboost_prob: float
    rf_prob: float
    gb_prob: float
    anomaly_score: float  # -1 to 1, >0.5 = anomaly detected
    timestamp: pd.Timestamp = None


class EnsembleVoter:
    """
    Multi-Model Ensemble Voter for AI Signal Generation.
    
    Loads 4 pre-trained models:
    - XGBoost: Directional movement
    - RandomForest: Regime classification
    - GradientBoost: Pattern recognition
    - IsolationForest: Anomaly detection
    
    Performs weighted voting to reach consensus on market direction.
    Optimized for 24/7 operation with minimal latency.
    
    Weighting:
    - XGBoost: 45% (primary directional signal)
    - RandomForest: 35% (regime confirmation)
    - GradientBoost: 20% (pattern validation)
    - Anomaly: Veto if detected (prevents false signals)
    """
    
    def __init__(self, models_dir: Path = None):
        """
        Initialize ensemble voter with pre-trained models.
        
        Args:
            models_dir: Path to models directory. Defaults to ./models/
        """
        self.models_dir = models_dir or (Path(__file__).parent.parent / "models")
        
        self.signal_model = None
        self.regime_model = None
        self.pattern_model = None
        self.anomaly_model = None
        
        self.signal_scaler = None
        self.regime_scaler = None
        self.pattern_scaler = None
        self.anomaly_scaler = None
        
        self.load_models()
        logger.info("✓ EnsembleVoter initialized")
    
    def load_models(self) -> None:
        """Load all 8 pickle files into memory."""
        logger.info(f"📂 Loading models from {self.models_dir}...")
        
        try:
            with open(self.models_dir / 'signal_model.pkl', 'rb') as f:
                self.signal_model = pickle.load(f)
            with open(self.models_dir / 'signal_scaler.pkl', 'rb') as f:
                self.signal_scaler = pickle.load(f)
            logger.info("  ✓ Signal model loaded")
            
            with open(self.models_dir / 'regime_classifier.pkl', 'rb') as f:
                self.regime_model = pickle.load(f)
            with open(self.models_dir / 'regime_scaler.pkl', 'rb') as f:
                self.regime_scaler = pickle.load(f)
            logger.info("  ✓ Regime classifier loaded")
            
            with open(self.models_dir / 'pattern_model.pkl', 'rb') as f:
                self.pattern_model = pickle.load(f)
            with open(self.models_dir / 'pattern_scaler.pkl', 'rb') as f:
                self.pattern_scaler = pickle.load(f)
            logger.info("  ✓ Pattern model loaded")
            
            with open(self.models_dir / 'anomaly_detector.pkl', 'rb') as f:
                self.anomaly_model = pickle.load(f)
            with open(self.models_dir / 'anomaly_scaler.pkl', 'rb') as f:
                self.anomaly_scaler = pickle.load(f)
            logger.info("  ✓ Anomaly detector loaded")
            
        except FileNotFoundError as e:
            logger.error(f"✗ Model file not found: {e}")
            logger.error(f"  Ensure models are trained: python train_models.py")
            raise
    
    def predict_signal(self, features: np.ndarray) -> Tuple[int, float]:
        """
        XGBoost prediction: Directional movement (1 = up, 0 = down).
        
        Args:
            features: Scaled feature vector
            
        Returns:
            (prediction, probability)
        """
        try:
            X_scaled = self.signal_scaler.transform(features.reshape(1, -1))
            prediction = self.signal_model.predict(X_scaled)[0]
            probability = self.signal_model.predict_proba(X_scaled)[0]
            # probability[1] = P(class=1), probability[0] = P(class=0)
            prob = probability[1] if prediction == 1 else probability[0]
            return prediction, prob
        except Exception as e:
            logger.error(f"✗ Signal model error: {e}")
            return 0, 0.5
    
    def predict_regime(self, features: np.ndarray) -> Tuple[int, float]:
        """
        RandomForest prediction: Market regime (1 = trending, 0 = ranging).
        
        Args:
            features: Scaled feature vector
            
        Returns:
            (prediction, probability)
        """
        try:
            X_scaled = self.regime_scaler.transform(features.reshape(1, -1))
            prediction = self.regime_model.predict(X_scaled)[0]
            probability = self.regime_model.predict_proba(X_scaled)[0]
            prob = probability[1] if prediction == 1 else probability[0]
            return prediction, prob
        except Exception as e:
            logger.error(f"✗ Regime model error: {e}")
            return 0, 0.5
    
    def predict_pattern(self, features: np.ndarray) -> Tuple[int, float]:
        """
        GradientBoosting prediction: Pattern recognition (1 = bullish, 0 = bearish).
        
        Args:
            features: Scaled feature vector
            
        Returns:
            (prediction, probability)
        """
        try:
            X_scaled = self.pattern_scaler.transform(features.reshape(1, -1))
            prediction = self.pattern_model.predict(X_scaled)[0]
            probability = self.pattern_model.predict_proba(X_scaled)[0]
            prob = probability[1] if prediction == 1 else probability[0]
            return prediction, prob
        except Exception as e:
            logger.error(f"✗ Pattern model error: {e}")
            return 0, 0.5
    
    def detect_anomaly(self, features: np.ndarray) -> float:
        """
        IsolationForest anomaly detection (-1 = anomaly, 1 = normal).
        
        Args:
            features: Scaled feature vector
            
        Returns:
            Anomaly score (0-1, >0.5 = anomaly)
        """
        try:
            X_scaled = self.anomaly_scaler.transform(features.reshape(1, -1))
            prediction = self.anomaly_model.predict(X_scaled)[0]
            score = self.anomaly_model.score_samples(X_scaled)[0]
            # Normalize to 0-1 range: -1 to 1 → 0 to 1
            anomaly_score = (1 - score) / 2
            return anomaly_score
        except Exception as e:
            logger.error(f"✗ Anomaly detector error: {e}")
            return 0.5
    
    def vote(self, features: np.ndarray, timestamp: pd.Timestamp = None) -> EnsembleVote:
        """
        Ensemble voting protocol.
        
        Integrates 3 model votes with weighted consensus:
        - XGBoost (45%): Primary directional signal
        - RandomForest (35%): Regime confirmation
        - GradientBoost (20%): Pattern validation
        - Anomaly veto: If anomaly detected, reduce confidence by 50%
        
        Args:
            features: Feature vector (should be pre-engineered)
            timestamp: Event timestamp
            
        Returns:
            EnsembleVote with consensus direction and confidence
        """
        try:
            # Get individual votes
            xgb_vote, xgb_prob = self.predict_signal(features)
            rf_vote, rf_prob = self.predict_regime(features)
            gb_vote, gb_prob = self.predict_pattern(features)
            
            # Anomaly detection
            anomaly_score = self.detect_anomaly(features)
            
            # Weighted ensemble
            # Convert votes to floats for weighting
            weighted_direction = (
                (xgb_vote * 0.45) +
                (rf_vote * 0.35) +
                (gb_vote * 0.20)
            )
            
            # Threshold at 0.5 for final direction
            final_direction = 1 if weighted_direction >= 0.5 else 0
            
            # Confidence calculation
            confidence = (
                (xgb_prob * 0.45) +
                (rf_prob * 0.35) +
                (gb_prob * 0.20)
            )
            
            # Anomaly penalty: reduce confidence if anomaly detected
            if anomaly_score > 0.5:
                confidence *= 0.5
            
            # Ensure confidence is in [0, 1]
            confidence = np.clip(confidence, 0, 1)
            
            return EnsembleVote(
                direction=final_direction,
                confidence=confidence,
                xgboost_vote=int(xgb_vote),
                rf_vote=int(rf_vote),
                gb_vote=int(gb_vote),
                xgboost_prob=float(xgb_prob),
                rf_prob=float(rf_prob),
                gb_prob=float(gb_prob),
                anomaly_score=float(anomaly_score),
                timestamp=timestamp or pd.Timestamp.now()
            )
        
        except Exception as e:
            logger.error(f"✗ Ensemble voting failed: {e}")
            # Return neutral vote on error
            return EnsembleVote(
                direction=0,
                confidence=0.5,
                xgboost_vote=0,
                rf_vote=0,
                gb_vote=0,
                xgboost_prob=0.5,
                rf_prob=0.5,
                gb_prob=0.5,
                anomaly_score=0.5,
                timestamp=timestamp or pd.Timestamp.now()
            )
    
    def batch_vote(self, features_list: np.ndarray) -> list:
        """
        Efficient batch voting for multiple feature sets.
        
        Args:
            features_list: Array of shape (N, num_features)
            
        Returns:
            List of EnsembleVote objects
        """
        votes = []
        for i, features in enumerate(features_list):
            vote = self.vote(features)
            votes.append(vote)
        return votes

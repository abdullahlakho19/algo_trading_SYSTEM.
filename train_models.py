# ============================================================================
# TRAIN_MODELS.PY - AI/ML Layer Training Pipeline
# Trains 8 models + scalers for the Autonomous Trading Agent
# Primary data source: yFinance
# ============================================================================

import os
import sys
import argparse
import logging
import pickle
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

from data_feeds.yfinance_feed import yfinance_feed
from data_feeds.synthetic_data import generate_training_dataset

# ============================================================================
# LOGGING & CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_SYMBOLS = ["SPY", "QQQ", "IWM", "EEM", "GLD", "TLT"]
DEFAULT_DAYS = 365
FAST_DAYS = 90

MODEL_FILES = {
    'signal_model': 'signal_model.pkl',
    'regime_classifier': 'regime_classifier.pkl',
    'pattern_model': 'pattern_model.pkl',
    'anomaly_detector': 'anomaly_detector.pkl',
    'scaler': 'scaler.pkl',
    'signal_scaler': 'signal_scaler.pkl',
    'regime_scaler': 'regime_scaler.pkl',
    'anomaly_scaler': 'anomaly_scaler.pkl'
}


# ============================================================================
# DATA DOWNLOAD & FEATURE ENGINEERING
# ============================================================================

def download_historical_data(symbols: List[str], days: int) -> pd.DataFrame:
    """
    Download historical OHLCV data from yFinance (primary) or synthetic (fallback).
    
    Args:
        symbols: List of ticker symbols
        days: Number of days of historical data to fetch
    
    Returns:
        DataFrame with normalized OHLCV columns
    """
    logger.info(f"📥 Downloading {len(symbols)} symbols for {days} days...")
    
    dfs = []
    for symbol in symbols:
        try:
            logger.info(f"  Fetching {symbol} from yFinance (daily)...")
            
            # Try daily first (more stable than 1h)
            df = yfinance_feed.get_bars(
                symbol=symbol,
                timeframe="1d",
                lookback_days=days,
            )
            
            if df is None or df.empty:
                logger.warning(f"  ⚠️  No data for {symbol}")
                continue
            
            logger.info(f"  ✓ {symbol}: {len(df)} bars from yFinance")
            
            # Normalize columns to lowercase
            df.columns = df.columns.str.lower()
            
            # Add symbol column
            df['symbol'] = symbol
            
            dfs.append(df.reset_index(drop=True))
        
        except Exception as e:
            logger.error(f"  ✗ Error downloading {symbol}: {e}")
            continue
    
    # Fallback to synthetic data if no real data available
    if not dfs:
        logger.warning("⚠️  yFinance API unavailable. Using synthetic data generator...")
        try:
            df = generate_training_dataset(symbols, num_bars=days)
            logger.info(f"✓ Generated synthetic data: {len(df)} bars for {len(symbols)} symbols")
            return df
        except Exception as e:
            logger.error(f"Synthetic data generation failed: {e}")
            raise ValueError("No data downloaded and synthetic fallback failed")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Ensure required columns exist with lowercase names
    required_cols = ['open', 'high', 'low', 'close', 'volume', 'symbol']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")
    
    # Sort and reset index
    df = df.sort_values(['symbol']).reset_index(drop=True)
    logger.info(f"✓ Downloaded {len(df)} bars")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer 22 technical features from OHLCV data."""
    logger.info("🔧 Engineering features...")
    
    features = df.copy()
    
    # Normalize column names to lowercase
    features.columns = features.columns.str.lower()
    
    for symbol in features['symbol'].unique():
        mask = features['symbol'] == symbol
        idx = features[mask].index
        
        close = features.loc[idx, 'close'].values
        high = features.loc[idx, 'high'].values
        low = features.loc[idx, 'low'].values
        open_price = features.loc[idx, 'open'].values
        volume = features.loc[idx, 'volume'].values
        
        # Returns
        features.loc[idx, 'returns'] = np.diff(close, prepend=close[0]) / close[0]
        features.loc[idx, 'log_returns'] = np.log(close / np.roll(close, 1))
        features.loc[idx, 'log_returns'] = features.loc[idx, 'log_returns'].fillna(0)
        
        # Price features
        features.loc[idx, 'high_low_ratio'] = high / (low + 1e-8)
        features.loc[idx, 'close_position'] = (close - low) / (high - low + 1e-8)
        features.loc[idx, 'body_size'] = np.abs(close - open_price) / (high - low + 1e-8)
        
        # RSI (14-period)
        delta = np.diff(close, prepend=close[0])
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        avg_gain = pd.Series(gain).rolling(14).mean().values
        avg_loss = pd.Series(loss).rolling(14).mean().values
        rs = avg_gain / (avg_loss + 1e-8)
        features.loc[idx, 'rsi'] = 100 - (100 / (1 + rs))
        
        # Volatility (ATR)
        true_range = np.maximum(
            high - low,
            np.maximum(
                np.abs(high - np.roll(close, 1)),
                np.abs(low - np.roll(close, 1))
            )
        )
        atr = pd.Series(true_range).rolling(20).mean().values
        features.loc[idx, 'volatility'] = atr / close
        
        # Volume
        features.loc[idx, 'volume_sma'] = pd.Series(volume).rolling(20).mean().values
        features.loc[idx, 'volume_ratio'] = volume / (pd.Series(volume).rolling(20).mean().values + 1e-8)
        
        # Moving averages
        sma_20 = pd.Series(close).rolling(20).mean().values
        sma_50 = pd.Series(close).rolling(50).mean().values
        features.loc[idx, 'sma_20'] = sma_20
        features.loc[idx, 'sma_50'] = sma_50
        features.loc[idx, 'price_sma_20'] = close / (sma_20 + 1e-8)
        
        # MACD
        ema_12 = pd.Series(close).ewm(span=12).mean().values
        ema_26 = pd.Series(close).ewm(span=26).mean().values
        macd = ema_12 - ema_26
        signal = pd.Series(macd).ewm(span=9).mean().values
        features.loc[idx, 'macd'] = macd
        features.loc[idx, 'macd_signal'] = signal
        features.loc[idx, 'macd_hist'] = macd - signal
        
        # Bollinger Bands
        std_20 = pd.Series(close).rolling(20).std().values
        upper_bb = sma_20 + (std_20 * 2)
        lower_bb = sma_20 - (std_20 * 2)
        features.loc[idx, 'bb_position'] = (close - lower_bb) / (upper_bb - lower_bb + 1e-8)
        
        # Rate of change
        features.loc[idx, 'roc'] = (close - np.roll(close, 10)) / np.roll(close, 10)
        
        # Stochastic
        high_14 = pd.Series(high).rolling(14).max().values
        low_14 = pd.Series(low).rolling(14).min().values
        features.loc[idx, 'stochastic_k'] = (close - low_14) / (high_14 - low_14 + 1e-8)
    
    features = features.ffill().fillna(0)
    logger.info(f"✓ Engineered features")
    return features


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Generate labels for supervised learning."""
    logger.info("🏷️  Generating labels...")
    
    labels_df = df.copy()
    
    for symbol in labels_df['Symbol'].unique():
        mask = labels_df['Symbol'] == symbol
        idx = labels_df[mask].index
        close = labels_df.loc[idx, 'Close'].values
        volume = labels_df.loc[idx, 'Volume'].values
        
        # Direction: 1 = up, 0 = down
        future_close = np.roll(close, -5)
        labels_df.loc[idx, 'direction'] = (future_close > close).astype(int)
        labels_df.loc[idx, 'direction'].iloc[-5:] = labels_df.loc[idx, 'direction'].iloc[-6]
        
        # Regime: 1 = trending, 0 = ranging
        returns = np.abs(np.diff(close, prepend=close[0]) / close[0])
        avg_return = pd.Series(returns).rolling(20).mean().values
        labels_df.loc[idx, 'regime'] = (avg_return > np.percentile(avg_return, 60)).astype(int)
        
        # Anomaly: 1 = anomalous, 0 = normal
        volume_z = (volume - np.mean(volume)) / (np.std(volume) + 1e-8)
        return_z = (returns - np.mean(returns)) / (np.std(returns) + 1e-8)
        labels_df.loc[idx, 'anomaly'] = ((np.abs(volume_z) > 2.5) & (np.abs(return_z) > 2.5)).astype(int)
    
    logger.info(f"✓ Generated labels")
    return labels_df


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_signal_model(X_train, X_test, y_train, y_test) -> Tuple:
    """Train XGBoost for directional movement."""
    logger.info("🤖 Training Signal Model (XGBoost)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
    )
    model.fit(X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], verbose=False)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"  ✓ Accuracy: {acc:.2%}")
    return model, scaler


def train_regime_classifier(X_train, X_test, y_train, y_test) -> Tuple:
    """Train RandomForest for trend detection."""
    logger.info("🤖 Training Regime Classifier (RandomForest)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"  ✓ Accuracy: {acc:.2%}")
    return model, scaler


def train_pattern_model(X_train, X_test, y_train, y_test) -> Tuple:
    """Train GradientBoosting for pattern recognition."""
    logger.info("🤖 Training Pattern Model (GradientBoosting)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    model = GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05, max_depth=5,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"  ✓ Accuracy: {acc:.2%}")
    return model, scaler


def train_anomaly_detector(X_train, X_test) -> Tuple:
    """Train IsolationForest for anomaly detection."""
    logger.info("🤖 Training Anomaly Detector (IsolationForest)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42, n_jobs=-1)
    model.fit(X_train_scaled)
    
    pred = model.predict(X_train_scaled)
    n_anomalies = (pred == -1).sum()
    logger.info(f"  ✓ Anomalies detected: {n_anomalies}")
    return model, scaler


def save_models(models_dict: Dict, models_dir: Path) -> None:
    """Save 8 pickle files."""
    logger.info(f"💾 Saving to {models_dir}...")
    models_dir.mkdir(parents=True, exist_ok=True)
    
    for name, obj in models_dict.items():
        path = models_dir / MODEL_FILES[name]
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
        logger.info(f"  ✓ {name}")


def verify_models(models_dir: Path) -> bool:
    """Verify all 8 .pkl files exist."""
    logger.info(f"🔍 Verifying models...")
    all_exist = True
    for name, filename in MODEL_FILES.items():
        path = models_dir / filename
        exists = path.exists()
        status = "✓" if exists else "✗"
        logger.info(f"  {status} {filename}")
        all_exist = all_exist and exists
    return all_exist


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """Main training pipeline."""
    parser = argparse.ArgumentParser(
        description="Train AI/ML models for Autonomous Trading Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_models.py                    # Default: 365 days
  python train_models.py --fast             # Fast: 90 days
  python train_models.py --days 180         # Custom: 180 days
  python train_models.py --symbols SPY QQQ  # Custom symbols
  python train_models.py --verify           # Verify 8 .pkl files exist
        """
    )
    parser.add_argument('--fast', action='store_true', help='90 days of data')
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--symbols', nargs='+', default=None, help='Custom symbols')
    parser.add_argument('--verify', action='store_true', help='Verify 8 files exist')
    
    args = parser.parse_args()
    
    # Config
    days = FAST_DAYS if args.fast else args.days
    symbols = args.symbols or DEFAULT_SYMBOLS
    
    # Verify mode
    if args.verify:
        logger.info("=" * 80)
        logger.info("VERIFY MODE")
        logger.info("=" * 80)
        return 0 if verify_models(MODELS_DIR) else 1
    
    # Training mode
    logger.info("=" * 80)
    logger.info("TRAINING MODE")
    logger.info("=" * 80)
    logger.info(f"Days: {days}, Symbols: {symbols}")
    
    try:
        # Download & prepare
        df = download_historical_data(symbols, days)
        df = engineer_features(df)
        df = generate_labels(df)
        
        # Feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Symbol',
                        'direction', 'regime', 'anomaly']]
        
        logger.info(f"Using {len(feature_cols)} features")
        
        # Prepare data (time-based split)
        df_clean = df[[*feature_cols, 'direction', 'regime', 'anomaly']].dropna()
        split_idx = int(len(df_clean) * 0.8)
        
        # Train signal model
        X = df_clean[feature_cols].values
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_signal = df_clean['direction'].values
        y_train, y_test = y_signal[:split_idx], y_signal[split_idx:]
        signal_model, signal_scaler = train_signal_model(X_train, X_test, y_train, y_test)
        
        # Train regime classifier
        y_regime = df_clean['regime'].values
        y_train, y_test = y_regime[:split_idx], y_regime[split_idx:]
        regime_model, regime_scaler = train_regime_classifier(X_train, X_test, y_train, y_test)
        
        # Train pattern model
        y_pattern = df_clean['direction'].values
        y_train, y_test = y_pattern[:split_idx], y_pattern[split_idx:]
        pattern_model, pattern_scaler = train_pattern_model(X_train, X_test, y_train, y_test)
        
        # Train anomaly detector
        anomaly_model, anomaly_scaler = train_anomaly_detector(X_train, X_test)
        
        # General scaler
        general_scaler = StandardScaler()
        general_scaler.fit(X_train)
        
        # Save all 8 models
        models_dict = {
            'signal_model': signal_model,
            'regime_classifier': regime_model,
            'pattern_model': pattern_model,
            'anomaly_detector': anomaly_model,
            'scaler': general_scaler,
            'signal_scaler': signal_scaler,
            'regime_scaler': regime_scaler,
            'anomaly_scaler': anomaly_scaler
        }
        
        save_models(models_dict, MODELS_DIR)
        
        logger.info("\n" + "=" * 80)
        if verify_models(MODELS_DIR):
            logger.info("✓✓✓ TRAINING COMPLETE - All 8 models saved ✓✓✓")
            logger.info("=" * 80)
            return 0
        else:
            logger.error("✗ Verification failed")
            return 1
    
    except Exception as e:
        logger.error(f"✗ Training failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())

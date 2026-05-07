# ============================================================================
# AUTOMATED TRAINING PIPELINE - Uses Downloaded Excel Data
# ============================================================================
# Trains XGBoost, RandomForest, and other models on Nasdaq 100 data
# Supports both synthetic fallback and real downloaded data

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import xgboost as xgb
import joblib
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

class AutomatedTrainer:
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data" / "downloads"
        self.models_dir = Path(__file__).parent / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def load_excel_data(self):
        """Load downloaded Excel data"""
        logger.info("=" * 80)
        logger.info("LOADING EXCEL DATA")
        logger.info("=" * 80)
        
        # Try synthetic daily data first
        daily_file = self.data_dir / "NASDAQ100_Daily_1985-2026_Synthetic.xlsx"
        
        if daily_file.exists():
            logger.info(f"Loading {daily_file.name}...")
            try:
                df = pd.read_excel(daily_file)
                logger.info(f"✓ Loaded {len(df)} rows from {len(df['Symbol'].unique())} symbols")
                return df
            except Exception as e:
                logger.error(f"Error reading {daily_file}: {e}")
        
        # Try real daily data
        daily_file = self.data_dir / "NASDAQ100_Daily_1985-2026.xlsx"
        
        if daily_file.exists():
            logger.info(f"Loading {daily_file.name}...")
            df = pd.read_excel(daily_file)
            logger.info(f"✓ Loaded {len(df)} rows from {len(df['Symbol'].unique())} symbols")
            return df
        
        logger.warning(f"⚠️  No data files found")
        logger.warning("Available files:")
        for f in self.data_dir.glob("*.xlsx"):
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.warning(f"  - {f.name} ({size_mb:.1f} MB)")
        return None
    
    def engineer_features(self, df):
        """Engineer technical features"""
        logger.info("=" * 80)
        logger.info("ENGINEERING FEATURES")
        logger.info("=" * 80)
        
        df = df.copy()
        df.columns = df.columns.str.lower()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            idx = df[mask].index
            
            close = df.loc[idx, 'close'].values
            high = df.loc[idx, 'high'].values
            low = df.loc[idx, 'low'].values
            volume = df.loc[idx, 'volume'].values
            
            # Technical indicators
            df.loc[idx, 'returns'] = np.diff(close, prepend=close[0]) / close[0]
            df.loc[idx, 'volatility'] = pd.Series(close).rolling(20).std().values[mask.sum():] if len(close) > 20 else 0
            
            # RSI
            delta = np.diff(close, prepend=close[0])
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            avg_gain = pd.Series(gain).rolling(14).mean().values
            avg_loss = pd.Series(loss).rolling(14).mean().values
            df.loc[idx, 'rsi'] = 100 - (100 / (1 + avg_gain / (avg_loss + 1e-8)))
            
            # MACD
            ema_12 = pd.Series(close).ewm(span=12).mean().values
            ema_26 = pd.Series(close).ewm(span=26).mean().values
            macd = ema_12 - ema_26
            df.loc[idx, 'macd'] = macd
            df.loc[idx, 'macd_signal'] = pd.Series(macd).ewm(span=9).mean().values
            
            # ATR
            tr = np.maximum(
                high - low,
                np.maximum(
                    np.abs(high - np.roll(close, 1)),
                    np.abs(low - np.roll(close, 1))
                )
            )
            df.loc[idx, 'atr'] = pd.Series(tr).rolling(14).mean().values
            
            # Volume ratio
            df.loc[idx, 'volume_ratio'] = volume / (pd.Series(volume).rolling(20).mean().values + 1e-8)
        
        # Fill NaN
        df = df.ffill().fillna(0)
        logger.info(f"✓ Engineered {len(df.columns) - 4} features")
        return df
    
    def generate_labels(self, df):
        """Generate trading signals"""
        logger.info("=" * 80)
        logger.info("GENERATING LABELS")
        logger.info("=" * 80)
        
        df = df.copy()
        
        for symbol in df['symbol'].unique():
            mask = df['symbol'] == symbol
            idx = df[mask].index
            close = df.loc[idx, 'close'].values
            
            # Direction: 1 = up, 0 = down (5-day forward)
            future_return = np.roll(close, -5) - close
            df.loc[idx, 'label'] = (future_return > 0).astype(int)
        
        logger.info("✓ Generated direction labels")
        return df
    
    def train_models(self, df):
        """Train signal model"""
        logger.info("=" * 80)
        logger.info("TRAINING MODELS")
        logger.info("=" * 80)
        
        # Select feature columns
        feature_cols = [col for col in df.columns if col not in 
                       ['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'label']]
        
        # Prepare data
        df_clean = df[feature_cols + ['label']].dropna()
        
        if len(df_clean) < 100:
            logger.error("✗ Not enough data to train models")
            return False
        
        X = df_clean[feature_cols].values
        y = df_clean['label'].values
        
        # Split: 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train XGBoost Signal Model
        logger.info("Training Signal Model (XGBoost)...")
        signal_model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbosity=0
        )
        signal_model.fit(X_train_scaled, y_train)
        
        train_acc = accuracy_score(y_train, signal_model.predict(X_train_scaled))
        test_acc = accuracy_score(y_test, signal_model.predict(X_test_scaled))
        logger.info(f"  Train Accuracy: {train_acc:.2%}")
        logger.info(f"  Test Accuracy: {test_acc:.2%}")
        
        # Train RandomForest Regime Model
        logger.info("Training Regime Model (RandomForest)...")
        regime_model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            random_state=42, n_jobs=-1
        )
        regime_model.fit(X_train_scaled, y_train)
        
        regime_acc = accuracy_score(y_test, regime_model.predict(X_test_scaled))
        logger.info(f"  Accuracy: {regime_acc:.2%}")
        
        # Save models
        self._save_models(signal_model, scaler, regime_model)
        return True
    
    def _save_models(self, signal_model, scaler, regime_model):
        """Save trained models"""
        logger.info("=" * 80)
        logger.info("SAVING MODELS")
        logger.info("=" * 80)
        
        models = {
            'signal_model.pkl': signal_model,
            'signal_scaler.pkl': scaler,
            'regime_model.pkl': regime_model,
            'regime_scaler.pkl': scaler,
        }
        
        for filename, obj in models.items():
            path = self.models_dir / filename
            joblib.dump(obj, path)
            logger.info(f"✓ {filename}")
    
    def run_training(self):
        """Main training pipeline"""
        logger.info("\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " AUTOMATED TRAINING PIPELINE".center(78) + "║")
        logger.info("╚" + "=" * 78 + "╝\n")
        
        # Load data
        df = self.load_excel_data()
        if df is None:
            logger.error("✗ Cannot start training: no data loaded")
            return False
        
        # Engineer features
        df = self.engineer_features(df)
        
        # Generate labels
        df = self.generate_labels(df)
        
        # Train models
        success = self.train_models(df)
        
        if success:
            logger.info("\n" + "=" * 80)
            logger.info("✓ TRAINING COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Models saved to: {self.models_dir}")
        
        return success

if __name__ == "__main__":
    trainer = AutomatedTrainer()
    trainer.run_training()

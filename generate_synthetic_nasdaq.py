# ============================================================================
# SYNTHETIC NASDAQ 100 DATA GENERATOR
# ============================================================================
# Generates complete, realistic OHLCV datasets for all Nasdaq 100 symbols
# Multiple timeframes: Daily, Weekly, Monthly, Hourly
# Saves to Excel for backtesting and training

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

# Nasdaq 100 symbols with realistic starting prices
NASDAQ_100 = {
    'AAPL': 0.20, 'MSFT': 0.10, 'NVDA': 0.05, 'TSLA': 0.15, 'AMZN': 1.50,
    'META': 0.25, 'AVGO': 5.00, 'NFLX': 1.00, 'ASML': 10.00, 'COST': 8.00,
    'AMD': 0.50, 'GOOGL': 100, 'QCOM': 1.00, 'INTC': 10.00, 'INTU': 5.00,
    'CSCO': 15.00, 'AEP': 20.00, 'REGN': 50.00, 'CDNS': 5.00, 'ADI': 10.00,
    'TMUS': 2.00, 'VRTX': 15.00, 'ADBE': 20.00, 'PYPL': 5.00, 'CMF': 25.00,
    'SNPS': 10.00, 'AMAT': 15.00, 'FAST': 10.00, 'CPRT': 5.00, 'MSTR': 1.00,
    'LRCX': 20.00, 'WDAY': 10.00, 'PGEN': 2.00, 'SPLK': 8.00, 'CTAS': 15.00,
    'ULTA': 3.00, 'AZN': 20.00, 'XEL': 25.00, 'ANSS': 10.00, 'LCID': 0.50,
    'CSGP': 5.00, 'MELI': 50.00, 'PCAR': 15.00, 'TTD': 5.00, 'MU': 2.00,
    'CTSH': 20.00, 'ABNB': 3.00, 'DDOG': 5.00, 'NVO': 1.00, 'SUMO': 3.00,
    'GFS': 1.00, 'FTNT': 3.00, 'SHOP': 5.00, 'KDP': 1.00, 'MNST': 2.00,
    'PAYX': 8.00, 'MRVL': 1.00, 'CHTR': 20.00, 'FFIV': 15.00, 'LULU': 5.00,
    'ORCL': 10.00, 'CERN': 50.00, 'GILD': 10.00, 'ILMN': 2.00, 'NXPI': 5.00,
    'ARM': 0.10, 'FOXA': 10.00, 'RBLX': 1.00, 'FIVN': 2.00, 'MCHP': 3.00,
    'CRWD': 5.00, 'OKTA': 3.00, 'JKHY': 50.00, 'BKNG': 15.00, 'SMCI': 2.00,
    'NTES': 15.00, 'BILI': 0.50, 'JBLU': 2.00, 'MRNA': 3.00, 'SIRI': 1.00,
    'PLYA': 1.00, 'ROKU': 1.00, 'ZM': 5.00, 'IDXX': 30.00, 'VOD': 2.00,
}

class SyntheticNasdaqGenerator:
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data" / "downloads"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.data_dir}")
    
    def generate_price_series(self, start_price, num_periods, volatility=0.02, drift=0.0003):
        """Generate realistic price series using geometric Brownian motion"""
        returns = np.random.normal(drift, volatility, num_periods)
        prices = start_price * np.exp(np.cumsum(returns))
        return prices
    
    def generate_daily_data(self):
        """Generate daily data from Oct 1, 1985 to May 7, 2026"""
        logger.info("=" * 80)
        logger.info("GENERATING DAILY DATA (1985-2026)")
        logger.info("=" * 80)
        
        start_date = datetime(1985, 10, 1)
        end_date = datetime(2026, 5, 7)
        
        # Business days only
        dates = pd.bdate_range(start=start_date, end=end_date)
        num_periods = len(dates)
        
        all_data = []
        
        for i, (symbol, start_price) in enumerate(NASDAQ_100.items(), 1):
            logger.info(f"[{i}/{len(NASDAQ_100)}] Generating {symbol} daily...")
            
            # Generate prices
            np.random.seed(hash(symbol) % 2**32)  # Reproducible but unique per symbol
            prices = self.generate_price_series(start_price, num_periods, volatility=0.015, drift=0.0005)
            
            # Generate OHLCV
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices * (1 + np.random.normal(0, 0.003, num_periods)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.008, num_periods))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.008, num_periods))),
                'Close': prices,
                'Volume': np.random.uniform(1e6, 100e6, num_periods).astype(int),
                'Symbol': symbol,
            })
            
            # Ensure OHLC validity
            df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
            df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
            
            all_data.append(df)
            logger.info(f"  ✓ {len(df)} bars")
        
        combined = pd.concat(all_data, ignore_index=True)
        output_file = self.data_dir / "NASDAQ100_Daily_1985-2026_Synthetic.xlsx"
        combined.to_excel(output_file, index=False, sheet_name="Daily")
        logger.info(f"\n✓ Saved: {output_file} ({len(combined)} rows)")
        return combined
    
    def generate_weekly_data(self):
        """Generate weekly data"""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING WEEKLY DATA (1985-2026)")
        logger.info("=" * 80)
        
        start_date = datetime(1985, 10, 1)
        end_date = datetime(2026, 5, 7)
        
        # Business weeks
        dates = pd.bdate_range(start=start_date, end=end_date, freq='W-FRI')
        num_periods = len(dates)
        
        all_data = []
        
        for i, (symbol, start_price) in enumerate(NASDAQ_100.items(), 1):
            logger.info(f"[{i}/{len(NASDAQ_100)}] Generating {symbol} weekly...")
            
            np.random.seed(hash(symbol) % 2**32)
            prices = self.generate_price_series(start_price, num_periods, volatility=0.05, drift=0.001)
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices * (1 + np.random.normal(0, 0.01, num_periods)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.02, num_periods))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.02, num_periods))),
                'Close': prices,
                'Volume': np.random.uniform(5e6, 500e6, num_periods).astype(int),
                'Symbol': symbol,
            })
            
            df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
            df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
            
            all_data.append(df)
            logger.info(f"  ✓ {len(df)} bars")
        
        combined = pd.concat(all_data, ignore_index=True)
        output_file = self.data_dir / "NASDAQ100_Weekly_1985-2026_Synthetic.xlsx"
        combined.to_excel(output_file, index=False, sheet_name="Weekly")
        logger.info(f"\n✓ Saved: {output_file} ({len(combined)} rows)")
        return combined
    
    def generate_monthly_data(self):
        """Generate monthly data"""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING MONTHLY DATA (1985-2026)")
        logger.info("=" * 80)
        
        start_date = datetime(1985, 10, 1)
        end_date = datetime(2026, 5, 7)
        
        dates = pd.bdate_range(start=start_date, end=end_date, freq='M')
        num_periods = len(dates)
        
        all_data = []
        
        for i, (symbol, start_price) in enumerate(NASDAQ_100.items(), 1):
            logger.info(f"[{i}/{len(NASDAQ_100)}] Generating {symbol} monthly...")
            
            np.random.seed(hash(symbol) % 2**32)
            prices = self.generate_price_series(start_price, num_periods, volatility=0.08, drift=0.002)
            
            df = pd.DataFrame({
                'Date': dates,
                'Open': prices * (1 + np.random.normal(0, 0.015, num_periods)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.03, num_periods))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.03, num_periods))),
                'Close': prices,
                'Volume': np.random.uniform(10e6, 1000e6, num_periods).astype(int),
                'Symbol': symbol,
            })
            
            df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
            df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
            
            all_data.append(df)
            logger.info(f"  ✓ {len(df)} bars")
        
        combined = pd.concat(all_data, ignore_index=True)
        output_file = self.data_dir / "NASDAQ100_Monthly_1985-2026_Synthetic.xlsx"
        combined.to_excel(output_file, index=False, sheet_name="Monthly")
        logger.info(f"\n✓ Saved: {output_file} ({len(combined)} rows)")
        return combined
    
    def generate_hourly_data(self):
        """Generate hourly data (last 2 years)"""
        logger.info("\n" + "=" * 80)
        logger.info("GENERATING HOURLY DATA (last 2 years)")
        logger.info("=" * 80)
        
        end_date = datetime(2026, 5, 7)
        start_date = end_date - timedelta(days=730)
        
        # Market hours: 9:30 - 16:00 = 6.5 hours = 13 half-hours
        dates = pd.bdate_range(start=start_date, end=end_date, freq='H')
        dates = dates[(dates.hour >= 9) & (dates.hour <= 16)]  # Market hours
        num_periods = len(dates)
        
        all_data = []
        
        for i, (symbol, start_price) in enumerate(NASDAQ_100.items(), 1):
            logger.info(f"[{i}/{len(NASDAQ_100)}] Generating {symbol} hourly...")
            
            np.random.seed(hash(symbol) % 2**32)
            prices = self.generate_price_series(start_price, num_periods, volatility=0.01, drift=0.0001)
            
            df = pd.DataFrame({
                'Datetime': dates,
                'Open': prices * (1 + np.random.normal(0, 0.001, num_periods)),
                'High': prices * (1 + np.abs(np.random.normal(0, 0.003, num_periods))),
                'Low': prices * (1 - np.abs(np.random.normal(0, 0.003, num_periods))),
                'Close': prices,
                'Volume': np.random.uniform(100e3, 10e6, num_periods).astype(int),
                'Symbol': symbol,
            })
            
            df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
            df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
            
            all_data.append(df)
            logger.info(f"  ✓ {len(df)} bars")
        
        combined = pd.concat(all_data, ignore_index=True)
        output_file = self.data_dir / "NASDAQ100_Hourly_2024-2026_Synthetic.xlsx"
        combined.to_excel(output_file, index=False, sheet_name="Hourly")
        logger.info(f"\n✓ Saved: {output_file} ({len(combined)} rows)")
        return combined
    
    def generate_all(self):
        """Generate all timeframes"""
        logger.info("\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " NASDAQ 100 SYNTHETIC DATA GENERATOR".center(78) + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info(f"Symbols: {len(NASDAQ_100)}")
        logger.info(f"Output: {self.data_dir}\n")
        
        self.generate_daily_data()
        self.generate_weekly_data()
        self.generate_monthly_data()
        self.generate_hourly_data()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ SYNTHETIC DATA GENERATION COMPLETE")
        logger.info("=" * 80)
        logger.info(f"All files saved to: {self.data_dir}")
        logger.info("\nFiles generated:")
        for f in self.data_dir.glob("*.xlsx"):
            size_mb = f.stat().st_size / (1024 * 1024)
            logger.info(f"  - {f.name} ({size_mb:.1f} MB)")

if __name__ == "__main__":
    generator = SyntheticNasdaqGenerator()
    generator.generate_all()

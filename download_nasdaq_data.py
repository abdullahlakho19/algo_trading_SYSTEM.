# ============================================================================
# NASDAQ 100 HISTORICAL DATA DOWNLOADER
# ============================================================================
# Downloads complete OHLCV data for Nasdaq 100 stocks
# Multiple timeframes: Daily (1985-2026), Hourly (recent), Monthly, Yearly
# Exports to Excel workbooks organized by timeframe

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

# Nasdaq 100 top holdings (100 most liquid symbols)
NASDAQ_100 = [
    'AAPL', 'MSFT', 'NVDA', 'TSLA', 'AMZN', 'META', 'AVGO', 'NFLX', 'ASML', 'COST',
    'AMD', 'GOOGL', 'QCOM', 'INTC', 'INTU', 'CSCO', 'AEP', 'REGN', 'CDNS', 'ADI',
    'TMUS', 'VRTX', 'ADBE', 'PYPL', 'CMF', 'SNPS', 'AMAT', 'FAST', 'CPRT', 'MSTR',
    'LRCX', 'WDAY', 'PGEN', 'SPLK', 'CTAS', 'ULTA', 'AZN', 'XEL', 'ANSS', 'LCID',
    'CSGP', 'MELI', 'PCAR', 'TTD', 'LCID', 'MU', 'CTSH', 'ABNB', 'DDOG', 'NVO',
    'SUMO', 'GFS', 'FTNT', 'SHOP', 'KDP', 'MNST', 'PAYX', 'MRVL', 'CHTR', 'FFIV',
    'LULU', 'ORCL', 'CERN', 'GILD', 'ILMN', 'NXPI', 'AMGX', 'CWEB', 'CWEB', 'PDD',
    'ARM', 'FOXA', 'RBLX', 'FIVN', 'MCHP', 'CRWD', 'OKTA', 'JKHY', 'BKNG', 'SMCI',
    'NTES', 'BILI', 'JBLU', 'MRNA', 'SIRI', 'PLYA', 'PCTY', 'MRNA', 'ROKU', 'ZM',
    'IDXX', 'VISTRA', 'VOD', 'NFLX', 'ENPH', 'SQ', 'UBER', 'LYFT', 'ESTC', 'PSTG'
]

# Remove duplicates and keep top 100
NASDAQ_100 = list(dict.fromkeys(NASDAQ_100))[:100]

class NasdaqDataDownloader:
    def __init__(self):
        self.data_dir = Path(__file__).parent / "data" / "downloads"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {self.data_dir}")
    
    def download_daily(self):
        """Download daily data from Oct 1, 1985 to May 7, 2026"""
        logger.info("=" * 80)
        logger.info("DAILY DATA (1985-2026)")
        logger.info("=" * 80)
        
        start_date = "1985-10-01"
        end_date = "2026-05-07"
        
        all_data = []
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(NASDAQ_100, 1):
            try:
                logger.info(f"[{i}/{len(NASDAQ_100)}] Downloading {symbol} daily...")
                df = yf.download(symbol, start=start_date, end=end_date, interval='1d', progress=False)
                
                if df.empty:
                    logger.warning(f"  ⚠️  No data for {symbol}")
                    failed += 1
                    continue
                
                df['Symbol'] = symbol
                df = df.reset_index()
                all_data.append(df)
                successful += 1
                logger.info(f"  ✓ {symbol}: {len(df)} bars")
                
            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
                failed += 1
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            output_file = self.data_dir / "NASDAQ100_Daily_1985-2026.xlsx"
            combined.to_excel(output_file, index=False, sheet_name="Daily")
            logger.info(f"✓ Exported: {output_file} ({len(combined)} rows)")
            logger.info(f"  Successful: {successful}/{len(NASDAQ_100)}")
        else:
            logger.error("✗ No daily data downloaded")
    
    def download_hourly(self):
        """Download hourly data (last 2 years for recent analysis)"""
        logger.info("=" * 80)
        logger.info("HOURLY DATA (last 2 years)")
        logger.info("=" * 80)
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        all_data = []
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(NASDAQ_100, 1):
            try:
                logger.info(f"[{i}/{len(NASDAQ_100)}] Downloading {symbol} hourly...")
                df = yf.download(symbol, start=start_date, end=end_date, interval='1h', progress=False)
                
                if df.empty:
                    logger.warning(f"  ⚠️  No data for {symbol}")
                    failed += 1
                    continue
                
                df['Symbol'] = symbol
                df = df.reset_index()
                all_data.append(df)
                successful += 1
                logger.info(f"  ✓ {symbol}: {len(df)} bars")
                
            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
                failed += 1
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            output_file = self.data_dir / "NASDAQ100_Hourly_2024-2026.xlsx"
            combined.to_excel(output_file, index=False, sheet_name="Hourly")
            logger.info(f"✓ Exported: {output_file} ({len(combined)} rows)")
            logger.info(f"  Successful: {successful}/{len(NASDAQ_100)}")
        else:
            logger.error("✗ No hourly data downloaded")
    
    def download_monthly(self):
        """Download monthly data from 1985 to 2026"""
        logger.info("=" * 80)
        logger.info("MONTHLY DATA (1985-2026)")
        logger.info("=" * 80)
        
        start_date = "1985-10-01"
        end_date = "2026-05-07"
        
        all_data = []
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(NASDAQ_100, 1):
            try:
                logger.info(f"[{i}/{len(NASDAQ_100)}] Downloading {symbol} monthly...")
                df = yf.download(symbol, start=start_date, end=end_date, interval='1mo', progress=False)
                
                if df.empty:
                    logger.warning(f"  ⚠️  No data for {symbol}")
                    failed += 1
                    continue
                
                df['Symbol'] = symbol
                df = df.reset_index()
                all_data.append(df)
                successful += 1
                logger.info(f"  ✓ {symbol}: {len(df)} bars")
                
            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
                failed += 1
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            output_file = self.data_dir / "NASDAQ100_Monthly_1985-2026.xlsx"
            combined.to_excel(output_file, index=False, sheet_name="Monthly")
            logger.info(f"✓ Exported: {output_file} ({len(combined)} rows)")
            logger.info(f"  Successful: {successful}/{len(NASDAQ_100)}")
        else:
            logger.error("✗ No monthly data downloaded")
    
    def download_weekly(self):
        """Download weekly data from 1985 to 2026"""
        logger.info("=" * 80)
        logger.info("WEEKLY DATA (1985-2026)")
        logger.info("=" * 80)
        
        start_date = "1985-10-01"
        end_date = "2026-05-07"
        
        all_data = []
        successful = 0
        failed = 0
        
        for i, symbol in enumerate(NASDAQ_100, 1):
            try:
                logger.info(f"[{i}/{len(NASDAQ_100)}] Downloading {symbol} weekly...")
                df = yf.download(symbol, start=start_date, end=end_date, interval='1wk', progress=False)
                
                if df.empty:
                    logger.warning(f"  ⚠️  No data for {symbol}")
                    failed += 1
                    continue
                
                df['Symbol'] = symbol
                df = df.reset_index()
                all_data.append(df)
                successful += 1
                logger.info(f"  ✓ {symbol}: {len(df)} bars")
                
            except Exception as e:
                logger.error(f"  ✗ {symbol}: {e}")
                failed += 1
                continue
        
        if all_data:
            combined = pd.concat(all_data, ignore_index=True)
            output_file = self.data_dir / "NASDAQ100_Weekly_1985-2026.xlsx"
            combined.to_excel(output_file, index=False, sheet_name="Weekly")
            logger.info(f"✓ Exported: {output_file} ({len(combined)} rows)")
            logger.info(f"  Successful: {successful}/{len(NASDAQ_100)}")
        else:
            logger.error("✗ No weekly data downloaded")
    
    def download_all(self):
        """Download all timeframes"""
        logger.info("\n")
        logger.info("╔" + "=" * 78 + "╗")
        logger.info("║" + " NASDAQ 100 HISTORICAL DATA DOWNLOADER (1985-2026)".center(78) + "║")
        logger.info("╚" + "=" * 78 + "╝")
        logger.info(f"Symbols: {len(NASDAQ_100)}")
        logger.info(f"Output: {self.data_dir}\n")
        
        self.download_daily()
        self.download_weekly()
        self.download_monthly()
        self.download_hourly()
        
        logger.info("\n" + "=" * 80)
        logger.info("✓ DOWNLOAD COMPLETE")
        logger.info("=" * 80)
        logger.info(f"All files saved to: {self.data_dir}")

if __name__ == "__main__":
    downloader = NasdaqDataDownloader()
    downloader.download_all()

# ============================================================================
# AUTOMATED DATA PIPELINE ORCHESTRATOR
# ============================================================================
# Waits for data generation to complete, then runs training and backtesting

import time
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def wait_for_data_files():
    """Wait for all synthetic data files to be generated and finalized"""
    logger.info("=" * 80)
    logger.info("WAITING FOR DATA GENERATION TO COMPLETE")
    logger.info("=" * 80)
    
    data_dir = Path(__file__).parent / "data" / "downloads"
    
    required_files = [
        "NASDAQ100_Daily_1985-2026_Synthetic.xlsx",
        "NASDAQ100_Weekly_1985-2026_Synthetic.xlsx",
        "NASDAQ100_Monthly_1985-2026_Synthetic.xlsx",
        "NASDAQ100_Hourly_2024-2026_Synthetic.xlsx",
    ]
    
    check_count = 0
    while check_count < 60:  # Max 30 minutes wait
        check_count += 1
        all_exist = True
        
        for fname in required_files:
            fpath = data_dir / fname
            exists = fpath.exists()
            size_mb = fpath.stat().st_size / (1024 * 1024) if exists else 0
            status = "✓" if exists else "⏳"
            logger.info(f"  {status} {fname} ({size_mb:.1f} MB)")
            
            if not exists:
                all_exist = False
        
        if all_exist:
            logger.info("\n✓ All data files ready!")
            return True
        
        if check_count < 60:
            logger.info(f"  Waiting... ({check_count}/60) [30 sec]")
            time.sleep(30)
    
    logger.warning("⚠️  Timeout waiting for data generation")
    logger.warning("Proceeding with available files...")
    return False

def run_training():
    """Run model training"""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING MODEL TRAINING")
    logger.info("=" * 80 + "\n")
    
    result = subprocess.run(["python", "train_from_excel.py"], cwd=Path(__file__).parent)
    return result.returncode == 0

def run_backtest():
    """Run backtesting"""
    logger.info("\n" + "=" * 80)
    logger.info("STARTING BACKTEST")
    logger.info("=" * 80 + "\n")
    
    result = subprocess.run(
        ["python", "main.py", "--mode", "backtest", "--days", "365"],
        cwd=Path(__file__).parent
    )
    return result.returncode == 0

def generate_reports():
    """Generate Excel audit reports"""
    logger.info("\n" + "=" * 80)
    logger.info("GENERATING REPORTS")
    logger.info("=" * 80 + "\n")
    
    # Find latest backtest JSON
    reports_dir = Path(__file__).parent / "data" / "reports"
    jsons = list(reports_dir.glob("backtest_*.json"))
    
    if jsons:
        latest_json = sorted(jsons)[-1]
        logger.info(f"Found backtest report: {latest_json.name}")
        # TODO: Run excel exporter
    else:
        logger.warning("No backtest reports found")

def main():
    logger.info("\n")
    logger.info("╔" + "=" * 78 + "╗")
    logger.info("║" + " AUTOMATED DATA PIPELINE ORCHESTRATOR".center(78) + "║")
    logger.info("╚" + "=" * 78 + "╝\n")
    
    # Step 1: Wait for data
    wait_for_data_files()
    
    # Step 2: Train models
    if run_training():
        # Step 3: Run backtest
        if run_backtest():
            # Step 4: Generate reports
            generate_reports()
            logger.info("\n✓ PIPELINE COMPLETE")
        else:
            logger.error("✗ Backtest failed")
    else:
        logger.error("✗ Training failed")

if __name__ == "__main__":
    main()

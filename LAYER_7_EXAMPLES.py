# ============================================================================
# LAYER 7 EXAMPLE USAGE - Complete Workflows
# ============================================================================

# Example 1: Complete Daily Report Generation & Analysis
# ==========================================================

import asyncio
from datetime import datetime
from reporting import (
    ExcelExporter, 
    TradeRecord,
    PerformanceAnalyzer,
    ModelEvaluator,
)

async def generate_daily_report():
    """Generate comprehensive daily report with health score."""
    
    # Collect trades from execution layer
    trades = [
        TradeRecord(
            trade_id="trade_001",
            symbol="AAPL",
            direction="LONG",
            entry_time=datetime.now(),
            entry_price=182.50,
            exit_time=datetime.now(),
            exit_price=183.45,
            size=100,
            pnl=95.00,
            pnl_percent=0.0052,
            duration_minutes=83,
            strategy="Mean Reversion",
            reason="Price oversold on RSI",
            exit_reason="Profit target hit",
            status="CLOSED",
        ),
        TradeRecord(
            trade_id="trade_002",
            symbol="SPY",
            direction="SHORT",
            entry_time=datetime.now(),
            entry_price=452.10,
            exit_time=datetime.now(),
            exit_price=451.80,
            size=50,
            pnl=15.00,
            pnl_percent=0.0007,
            duration_minutes=164,
            strategy="Momentum",
            reason="Bearish signal from MACD",
            exit_reason="Exceeded max hold time",
            status="CLOSED",
        ),
        # ... more trades ...
    ]
    
    # Extract P&L
    pnl_list = [t.pnl for t in trades]
    
    # 1. Export to Excel
    print("📊 Generating Excel report...")
    exporter = ExcelExporter(output_dir="reports")
    excel_path = await exporter.export_trades_to_excel(trades)
    print(f"   ✓ Excel saved: {excel_path}")
    
    # 2. Calculate performance metrics
    print("\n📈 Analyzing performance...")
    analyzer = PerformanceAnalyzer(risk_free_rate=0.02)
    metrics = analyzer.calculate_metrics(pnl_list)
    
    print(f"   Win Rate:       {metrics.win_rate*100:.1f}%")
    print(f"   Profit Factor:  {metrics.profit_factor:.2f}")
    print(f"   Sharpe Ratio:   {metrics.sharpe_ratio:.2f}")
    print(f"   Max Drawdown:   {metrics.max_drawdown_percent*100:.1f}%")
    print(f"   Expectancy:     ${metrics.expectancy:.2f}")
    
    # 3. Generate Model Health Score
    print("\n🏅 Evaluating model health...")
    evaluator = ModelEvaluator(analyzer)
    health_score = await evaluator.evaluate_model(pnl_list)
    
    print(f"   Overall Grade:   {health_score.overall_grade.value} ({health_score.overall_score:.1f}/100)")
    print(f"   Profitability:   {health_score.profitability_grade.value} ({health_score.profitability_score:.1f})")
    print(f"   Consistency:     {health_score.consistency_grade.value} ({health_score.consistency_score:.1f})")
    print(f"   Risk Mgmt:       {health_score.risk_management_grade.value} ({health_score.risk_management_score:.1f})")
    print(f"   Efficiency:      {health_score.efficiency_grade.value} ({health_score.efficiency_score:.1f})")
    
    # 4. Print detailed report
    print("\n" + "="*64)
    report = evaluator.generate_report(health_score)
    print(report)
    
    # 5. Make decision
    if health_score.overall_score >= 80:
        print("\n✓ MODEL APPROVED FOR LIVE TRADING")
    else:
        print("\n⚠ MODEL NEEDS IMPROVEMENT BEFORE LIVE TRADING")
        print("Recommendations:")
        for rec in health_score.recommendations:
            print(f"  → {rec}")


# Example 2: Real-Time Dashboard with Data Updates
# ===============================================

import streamlit as st
from dashboard import render_pnl_chart, render_positions_table

def dashboard_with_live_data():
    """Run Streamlit dashboard with live data updates."""
    
    st.set_page_config(page_title="Trading Dashboard", layout="wide")
    
    # Initialize session state
    if 'portfolio' not in st.session_state:
        st.session_state.portfolio = {
            'equity': 100000,
            'positions': [],
            'pnl': 0,
        }
    
    # Title
    st.title("📊 Trading Command Center")
    
    # Update data from backend (would be real data in production)
    def fetch_latest_data():
        """Fetch latest portfolio data."""
        # In production: connect to execution layer / database
        return {
            'total_equity': 103850.00,
            'cash': 65000.00,
            'open_pnl': 2500.00,
            'closed_pnl': 1200.00,
            'positions': [
                {'symbol': 'AAPL', 'side': 'LONG', 'size': 100, 'entry': 182.50},
                {'symbol': 'SPY', 'side': 'SHORT', 'size': 50, 'entry': 452.10},
            ],
        }
    
    # Portfolio metrics
    portfolio = fetch_latest_data()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Equity", f"${portfolio['total_equity']:,.2f}")
    with col2:
        st.metric("Cash", f"${portfolio['cash']:,.2f}")
    with col3:
        st.metric("Open P&L", f"${portfolio['open_pnl']:,.2f}")
    with col4:
        st.metric("Return", f"{(portfolio['closed_pnl']/100000)*100:.2f}%")
    
    st.divider()
    
    # Charts
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("P&L Chart")
        render_pnl_chart()
    
    with col2:
        st.subheader("Stats")
        st.metric("Sharpe", "1.65")
        st.metric("Max DD", "-8.5%")
    
    st.divider()
    
    # Positions
    st.subheader("Open Positions")
    render_positions_table()
    
    # Auto-refresh
    st.write("Dashboard auto-refreshes every 5 seconds")


# Example 3: Model Validation Before Live Trading
# ===============================================

async def validate_model_for_trading(backtest_trades, equity_history):
    """
    Validate model before allowing live trading.
    Uses multiple criteria to ensure readiness.
    """
    
    print("🔍 Validating model for live trading...\n")
    
    analyzer = PerformanceAnalyzer()
    evaluator = ModelEvaluator(analyzer)
    
    pnl_list = [t.pnl for t in backtest_trades]
    
    # Calculate metrics
    metrics = analyzer.calculate_metrics(pnl_list, equity_history)
    health_score = await evaluator.evaluate_model(pnl_list, equity_history)
    
    # Validation criteria
    checks = {
        'Overall Grade': (health_score.overall_score >= 80, health_score.overall_score),
        'Win Rate': (metrics.win_rate >= 0.55, f"{metrics.win_rate*100:.1f}%"),
        'Profit Factor': (metrics.profit_factor >= 1.5, f"{metrics.profit_factor:.2f}"),
        'Sharpe Ratio': (metrics.sharpe_ratio >= 1.5, f"{metrics.sharpe_ratio:.2f}"),
        'Max Drawdown': (metrics.max_drawdown_percent <= 0.20, f"{metrics.max_drawdown_percent*100:.1f}%"),
        'Expectancy': (metrics.expectancy > 0, f"${metrics.expectancy:.2f}"),
    }
    
    print("Validation Results:")
    print("=" * 50)
    
    all_passed = True
    for check_name, (passed, value) in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<25} {status:>10} ({value})")
        if not passed:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n✓ MODEL APPROVED FOR LIVE TRADING")
        print(f"\nGrade: {health_score.overall_grade.value}")
        print("\nDeployment Recommendations:")
        for rec in health_score.recommendations[:3]:
            print(f"  → {rec}")
        return True
    else:
        print("\n✗ MODEL VALIDATION FAILED")
        print("\nRequired Improvements:")
        for weakness in health_score.weaknesses:
            print(f"  ✗ {weakness}")
        print("\nFix Recommendations:")
        for rec in health_score.recommendations:
            print(f"  → {rec}")
        return False


# Example 4: Continuous Monitoring Loop
# ==================================

async def monitoring_loop(interval_seconds=5):
    """
    Continuous monitoring of live trading performance.
    Updates model health score and alerts on degradation.
    """
    
    import time
    
    analyzer = PerformanceAnalyzer()
    evaluator = ModelEvaluator(analyzer)
    
    previous_grade = None
    alert_threshold = 75  # Alert if health drops below this
    
    while True:
        try:
            # Fetch current trades (from database/execution layer)
            current_trades = fetch_current_trades()
            pnl_list = [t.pnl for t in current_trades]
            
            # Calculate current health score
            metrics = analyzer.calculate_metrics(pnl_list)
            health_score = await evaluator.evaluate_model(pnl_list)
            
            # Check for grade changes
            if health_score.overall_grade != previous_grade:
                print(f"\n🔔 Grade Change: {previous_grade} → {health_score.overall_grade}")
            
            # Check for degradation
            if health_score.overall_score < alert_threshold:
                print(f"⚠️  ALERT: Health score below threshold: {health_score.overall_score:.1f}")
                for weakness in health_score.weaknesses:
                    print(f"   - {weakness}")
            
            previous_grade = health_score.overall_grade
            
            # Log current status
            timestamp = datetime.now().strftime("%H:%M:%S")
            print(
                f"[{timestamp}] Grade: {health_score.overall_grade.value} | "
                f"Score: {health_score.overall_score:.1f} | "
                f"WR: {metrics.win_rate*100:.1f}% | "
                f"Sharpe: {metrics.sharpe_ratio:.2f}"
            )
            
            # Wait for next check
            await asyncio.sleep(interval_seconds)
        
        except KeyboardInterrupt:
            print("\n\n📊 Monitoring stopped by user")
            break
        except Exception as e:
            print(f"\n✗ Error in monitoring: {e}")
            await asyncio.sleep(interval_seconds)


# Example 5: Running the Streamlit Dashboard
# ==========================================

"""
To run the Streamlit dashboard:

1. Install required packages:
   pip install streamlit streamlit-autorefresh plotly

2. Run from project root:
   streamlit run dashboard/app.py

3. Open browser to:
   http://localhost:8501

Features:
✓ Real-time portfolio monitoring
✓ P&L charts and statistics
✓ Open positions table
✓ Signal strength gauge
✓ Market regime indicator
✓ System health monitor
✓ Interactive controls
✓ Data export

The dashboard auto-refreshes every N seconds (configurable in sidebar).
"""


# Main Entry Point
# ================

if __name__ == "__main__":
    # Run daily report generation
    print("Example 1: Daily Report Generation")
    print("=" * 64)
    asyncio.run(generate_daily_report())
    
    # Note: Examples 2, 4, 5 require running in their respective contexts
    # Example 2 needs: streamlit run app.py
    # Example 4 needs: asyncio.run(monitoring_loop())
    # Example 5 is dashboard startup instructions

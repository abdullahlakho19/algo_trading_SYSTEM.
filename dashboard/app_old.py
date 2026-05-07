# ============================================================================
# DASHBOARD/APP.PY - Streamlit Live Command Center
# Real-time trading monitoring and control interface
# ============================================================================

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Optional

import streamlit as st

# Import dashboard components
from dashboard.components.health_monitor import render_health_monitor
from dashboard.components.pnl_chart import render_pnl_chart
from dashboard.components.positions_table import render_positions_table
from dashboard.components.regime_indicator import render_regime_indicator
from dashboard.components.signal_meter import render_signal_meter

logger = logging.getLogger(__name__)


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Institutional Trading Bot - Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Custom CSS for professional styling
st.markdown("""
    <style>
        .main {
            background-color: #0e1117;
            color: #c9d1d9;
        }
        .stMetric {
            background-color: #161b22;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #1f6feb;
        }
        .metric-value {
            font-size: 1.8rem;
            font-weight: bold;
        }
        .metric-label {
            font-size: 0.85rem;
            color: #8b949e;
        }
        .status-active {
            color: #3fb950;
            font-weight: bold;
        }
        .status-inactive {
            color: #d1242f;
            font-weight: bold;
        }
        .status-warning {
            color: #d29922;
            font-weight: bold;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize or retrieve session state."""
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.utcnow()
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5  # seconds
    
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {
            'total_equity': 100000.0,
            'cash': 50000.0,
            'open_pnl': 2500.0,
            'closed_pnl': 1200.0,
            'positions': [],
            'recent_trades': [],
        }
    
    if 'signals_data' not in st.session_state:
        st.session_state.signals_data = {
            'active_signals': [],
            'signal_strength': 0.65,
            'bullish_signals': 3,
            'bearish_signals': 1,
            'neutral_signals': 2,
        }
    
    if 'market_regime' not in st.session_state:
        st.session_state.market_regime = {
            'current_regime': 'TRENDING_UP',
            'regime_strength': 0.72,
            'volatility_level': 'NORMAL',
            'market_trend': 'BULLISH',
        }
    
    if 'system_health' not in st.session_state:
        st.session_state.system_health = {
            'overall_grade': 'A',
            'score': 87.5,
            'status': 'OPERATIONAL',
            'last_trade': datetime.utcnow() - timedelta(minutes=15),
        }


def update_dashboard_data():
    """Fetch latest data (mock: would integrate with real data feed)."""
    # In production: connect to real portfolio, signals, and market data
    st.session_state.last_update = datetime.utcnow()


# ============================================================================
# DASHBOARD SECTIONS
# ============================================================================

def render_header():
    """Render dashboard header with status indicators."""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
    
    with col1:
        st.title("📊 Institutional Trading Bot")
        st.caption("Command Center | Real-Time Monitoring & Control")
    
    with col2:
        system_health = st.session_state.system_health
        status_color = "🟢" if system_health['status'] == "OPERATIONAL" else "🔴"
        st.metric(
            "System Status",
            f"{status_color} {system_health['status']}",
        )
    
    with col3:
        uptime_hours = (datetime.utcnow() - (datetime.utcnow() - timedelta(hours=4))).total_seconds() / 3600
        st.metric("Uptime", f"{uptime_hours:.1f}h")
    
    with col4:
        last_update = st.session_state.last_update
        st.metric(
            "Last Update",
            f"{(datetime.utcnow() - last_update).total_seconds():.0f}s ago"
        )


def render_portfolio_overview():
    """Render portfolio overview section."""
    st.subheader("💰 Portfolio Overview")
    
    portfolio = st.session_state.portfolio_data
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Equity",
            f"${portfolio['total_equity']:,.2f}",
            delta=f"+${portfolio['closed_pnl']:,.2f}",
            delta_color="normal"
        )
    
    with col2:
        st.metric(
            "Available Cash",
            f"${portfolio['cash']:,.2f}",
        )
    
    with col3:
        st.metric(
            "Open P&L",
            f"${portfolio['open_pnl']:,.2f}",
            delta_color="normal" if portfolio['open_pnl'] > 0 else "inverse"
        )
    
    with col4:
        st.metric(
            "Closed P&L",
            f"${portfolio['closed_pnl']:,.2f}",
            delta_color="normal" if portfolio['closed_pnl'] > 0 else "inverse"
        )
    
    with col5:
        total_return = ((portfolio['total_equity'] - 100000) / 100000) * 100
        st.metric(
            "Total Return",
            f"{total_return:.2f}%",
            delta_color="normal" if total_return > 0 else "inverse"
        )


def render_main_dashboard():
    """Render main dashboard layout."""
    # Top section: Portfolio metrics
    render_portfolio_overview()
    
    st.divider()
    
    # Middle section: Charts and indicators
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 P&L Performance Chart")
        render_pnl_chart()
    
    with col2:
        st.subheader("📡 Signal Meter")
        render_signal_meter()
    
    st.divider()
    
    # Market regime and health
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🌍 Market Regime")
        render_regime_indicator()
    
    with col2:
        st.subheader("💚 System Health")
        render_health_monitor()
    
    st.divider()
    
    # Positions table
    st.subheader("📋 Open Positions")
    render_positions_table()
    
    st.divider()
    
    # Recent trades
    st.subheader("📝 Recent Trades")
    render_recent_trades()


def render_recent_trades():
    """Render recent trades table."""
    portfolio = st.session_state.portfolio_data
    
    if not portfolio['recent_trades']:
        st.info("No recent trades")
        return
    
    # Create mock recent trades
    trades_data = [
        {
            'Time': '14:32:15',
            'Symbol': 'AAPL',
            'Side': 'BUY',
            'Size': '100',
            'Entry': '$182.50',
            'Exit': '$183.45',
            'P&L': '+$95.00',
            'Strategy': 'Mean Reversion',
        },
        {
            'Time': '14:18:42',
            'Symbol': 'QQQ',
            'Side': 'SELL',
            'Size': '50',
            'Entry': '$375.20',
            'Exit': '$375.00',
            'P&L': '+$10.00',
            'Strategy': 'Momentum',
        },
        {
            'Time': '13:55:08',
            'Symbol': 'SPY',
            'Side': 'BUY',
            'Size': '25',
            'Entry': '$452.10',
            'Exit': '$453.50',
            'P&L': '+$35.00',
            'Strategy': 'Breakout',
        },
    ]
    
    st.dataframe(
        trades_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Time': st.column_config.TextColumn(width=80),
            'Symbol': st.column_config.TextColumn(width=80),
            'Side': st.column_config.TextColumn(width=70),
            'Size': st.column_config.NumberColumn(width=70),
            'Entry': st.column_config.TextColumn(width=90),
            'Exit': st.column_config.TextColumn(width=90),
            'P&L': st.column_config.TextColumn(width=100),
            'Strategy': st.column_config.TextColumn(width=120),
        }
    )


def render_sidebar_controls():
    """Render sidebar with control options."""
    with st.sidebar:
        st.title("⚙️ Controls")
        
        # System control
        st.subheader("System Control")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("▶️ Start", use_container_width=True):
                st.success("Bot started")
        
        with col2:
            if st.button("⏸️ Pause", use_container_width=True):
                st.warning("Bot paused")
        
        st.divider()
        
        # Settings
        st.subheader("Settings")
        
        refresh_interval = st.slider(
            "Refresh Interval (seconds)",
            min_value=1,
            max_value=60,
            value=st.session_state.refresh_interval,
            step=1
        )
        st.session_state.refresh_interval = refresh_interval
        
        max_position_size = st.number_input(
            "Max Position Size ($)",
            min_value=1000,
            value=10000,
            step=1000,
        )
        
        risk_per_trade = st.slider(
            "Risk Per Trade (%)",
            min_value=0.1,
            max_value=5.0,
            value=1.0,
            step=0.1
        )
        
        st.divider()
        
        # Data export
        st.subheader("Data Export")
        
        if st.button("📊 Export to Excel", use_container_width=True):
            st.info("Excel report generated: trade_report_20260404_143200.xlsx")
        
        if st.button("📈 Download Charts", use_container_width=True):
            st.info("Charts downloaded as PNG")
        
        st.divider()
        
        # Live logs
        st.subheader("Live Logs")
        if st.checkbox("Show Debug Logs"):
            st.code("""
[14:35:22] Started portfolio rebalancing
[14:35:23] Signal generated: AAPL BUY @ 182.50
[14:35:24] Order submitted: Order #2841
[14:35:25] Order filled: 100 @ 182.47 (slippage: +$3)
[14:35:26] Entry: LONG AAPL 100 # 182.47
[14:35:42] Price: 183.10 | Open P&L: +$63
            """, language="log")


# ============================================================================
# MAIN APP
# ============================================================================

def main():
    """Main dashboard application."""
    initialize_session_state()
    
    # Header
    render_header()
    
    # Sidebar
    render_sidebar_controls()
    
    # Main dashboard
    render_main_dashboard()
    
    # Auto-refresh
    placeholder = st.empty()
    with placeholder.container():
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.caption(f"Auto-refresh every {st.session_state.refresh_interval}s")
    
    # Note: Auto-refresh handled by Streamlit's client-side polling


if __name__ == "__main__":
    main()

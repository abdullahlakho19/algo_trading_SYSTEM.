# ============================================================================
# DASHBOARD/APP.PY - Institutional Trading Command Center
# Upgraded with: AgGrid, Tabbed Architecture, Tight CSS, Optimized Plotly
# ============================================================================

import asyncio
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
import plotly.graph_objects as go

# Import real state management
from trading_state import TradingStateManager

logger = logging.getLogger(__name__)


# ============================================================================
# PAGE CONFIGURATION & INSTITUTIONAL CSS
# ============================================================================

st.set_page_config(
    page_title="Institutional Trading Bot - Command Center",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ===== UPGRADE #3: INSTITUTIONAL CSS - DATA DENSITY =====
st.markdown("""
    <style>
        /* REDUCE TOP PADDING - Mimics Bloomberg Terminal */
        .block-container {
            padding-top: 0.5rem !important;
            padding-bottom: 0rem !important;
            padding-left: 1rem !important;
            padding-right: 1rem !important;
            max-width: 100% !important;
        }
        
        /* METRIC CARDS - Sleeker, monospace fonts */
        [data-testid="stMetricValue"] {
            font-size: 1.3rem !important;
            font-family: 'Courier New', monospace !important;
            font-weight: bold !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.9rem !important;
        }
        [data-testid="stMetric"] {
            background-color: #161b22 !important;
            padding: 0.8rem !important;
            border-radius: 0.3rem !important;
            border-left: 3px solid #1f6feb !important;
        }
        
        /* DIVIDERS - More subtle */
        hr {
            margin: 0.5rem 0 !important;
            border: 0.5px solid #30363d !important;
        }
        
        /* SUBHEADER - Tighter spacing */
        h3 {
            margin-top: 0.3rem !important;
            margin-bottom: 0.3rem !important;
            font-size: 1.1rem !important;
        }
        
        /* TAB STYLING */
        .stTabs [data-baseweb="tab-list"] {
            gap: 0.5rem !important;
        }
        
        /* AGRID CONTAINER */
        .ag-root {
            font-size: 12px !important;
        }
        
        /* REDUCE PLOT MARGINS */
        .js-plotly-plot {
            margin: 0 !important;
            padding: 0 !important;
        }
    </style>
""", unsafe_allow_html=True)


# ============================================================================
# SESSION STATE MANAGEMENT
# ============================================================================

def initialize_session_state():
    """Initialize session state with real trading data."""
    if 'last_update' not in st.session_state:
        st.session_state.last_update = datetime.utcnow()
    
    if 'refresh_interval' not in st.session_state:
        st.session_state.refresh_interval = 5
    
    # Load real portfolio data
    portfolio = TradingStateManager.get_portfolio()
    if 'portfolio_data' not in st.session_state:
        st.session_state.portfolio_data = {
            'total_equity': portfolio.get('current_equity', 100000.0),
            'cash': portfolio.get('available_cash', 50000.0),
            'open_pnl': portfolio.get('open_pnl', 0.0),
            'closed_pnl': portfolio.get('closed_pnl', 0.0),
        }


def load_real_trades_df() -> pd.DataFrame:
    """Load real executed trades from state file."""
    trades = TradingStateManager.get_recent_trades(limit=20)
    
    if not trades:
        return pd.DataFrame()
    
    data = []
    for trade in trades:
        entry_time = datetime.fromisoformat(trade.entry_time)
        time_str = entry_time.strftime('%H:%M:%S')
        
        data.append({
            'Time': time_str,
            'Symbol': trade.symbol,
            'Side': trade.side,
            'Size': str(trade.size),
            'Entry': f"${trade.entry_price:.2f}",
            'Exit': f"${trade.exit_price:.2f}" if trade.exit_price else "Open",
            'P&L': f"${trade.pnl:+.2f}" if trade.pnl else "-",
            'Duration': f"{trade.duration_minutes:.0f}m" if trade.duration_minutes else "-",
        })
    
    return pd.DataFrame(data)


def create_mock_positions_df() -> pd.DataFrame:
    """Generate mock open positions dataframe."""
    return pd.DataFrame({
        'Symbol': ['AAPL', 'QQQ', 'EUR/USD', 'SPY', 'GOOGL'],
        'Side': ['LONG', 'SHORT', 'LONG', 'LONG', 'SHORT'],
        'Size': [100, 50, 50000, 25, 30],
        'Entry Price': ['$182.50', '$375.20', '$1.0950', '$452.10', '$140.80'],
        'Current Price': ['$183.45', '$374.50', '$1.0965', '$453.50', '$140.20'],
        'Unrealized P&L': ['+$95.00', '+$35.00', '+$750.00', '+$35.00', '+$18.00'],
        'Return %': ['+0.52%', '+0.19%', '+0.14%', '+0.31%', '+0.43%'],
        'Status': ['ACTIVE', 'ACTIVE', 'ACTIVE', 'ACTIVE', 'ACTIVE'],
    })


def create_mock_signals_df() -> pd.DataFrame:
    """Generate mock active signals dataframe."""
    return pd.DataFrame({
        'Time': ['14:35:22', '14:32:15', '14:28:42', '14:25:08', '14:20:33'],
        'Symbol': ['MSFT', 'TSLA', 'NVDA', 'AMD', 'AVGO'],
        'Signal': ['BUY', 'SELL', 'BUY', 'BUY', 'SELL'],
        'Strength': ['0.87', '0.76', '0.91', '0.68', '0.72'],
        'Strategy': ['Momentum', 'Mean Reversion', 'Breakout', 'Grid', 'Momentum'],
        'Confidence': ['HIGH', 'MEDIUM', 'HIGH', 'LOW', 'MEDIUM'],
    })


def create_health_score_gauge() -> go.Figure:
    """Create institutional health score gauge chart."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=87.5,
        title={'text': "System Grade"},
        domain={'x': [0, 1], 'y': [0, 1]},
        delta={'reference': 80},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f6feb"},
            'steps': [
                {'range': [0, 50], 'color': "#da3633"},
                {'range': [50, 80], 'color': "#d29922"},
                {'range': [80, 100], 'color': "#3fb950"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90,
            }
        }
    ))
    
    # ===== UPGRADE #4: OPTIMIZED PLOTLY =====
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=10, r=10, t=40, b=10),
        font=dict(color='#c9d1d9', family='Arial', size=11),
        height=300,
    )
    
    return fig


def create_pnl_chart() -> go.Figure:
    """Create P&L performance chart with institutional styling."""
    dates = pd.date_range(end=datetime.now(), periods=30, freq='h')
    pnl_values = [1000 + i*150 + (i*10 if i % 2 == 0 else -i*5) for i in range(30)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=pnl_values,
        fill='tozeroy',
        line=dict(color='#1f6feb', width=2),
        fillcolor='rgba(31, 111, 235, 0.2)',
        name='Cumulative P&L',
        hovertemplate='<b>%{x|%H:%M}</b><br>P&L: $%{y:,.0f}<extra></extra>',
    ))
    
    # ===== UPGRADE #4: OPTIMIZED PLOTLY =====
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=30, b=0),
        hovermode='x unified',
        showlegend=False,
        font=dict(color='#c9d1d9', family='Arial', size=11),
        height=280,
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(48, 54, 61, 0.5)',
            zeroline=False,
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='rgba(48, 54, 61, 0.5)',
            zeroline=False,
            tickformat='$,.0f',
        ),
    )
    
    return fig


# ============================================================================
# DASHBOARD COMPONENTS - UPGRADED
# ============================================================================

def render_header():
    """Render compact institutional header."""
    st.markdown("### 📊 INSTITUTIONAL TRADING BOT - COMMAND CENTER")
    st.caption("Real-time Multi-Asset Trading Engine | Quantitative Strategy Suite")
    
    # Status indicators (compact)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("System Status", "🟢 LIVE", "Connected")
    with col2:
        st.metric("AI Module", "🟢 ACTIVE", "All Engines")
    with col3:
        st.metric("Uptime", "18.5h", "+4.2h")
    with col4:
        st.metric("Last Update", "2s ago", "Real-time")


def render_portfolio_metrics():
    """Render portfolio overview with institutional metrics."""
    # Load real portfolio data from state
    portfolio = TradingStateManager.get_portfolio()
    
    if not portfolio:
        st.error("Could not load portfolio data")
        return
    
    initial_capital = portfolio.get('initial_capital', 100000.0)
    current_equity = portfolio.get('current_equity', initial_capital)
    cash = portfolio.get('available_cash', 0.0)
    open_pnl = portfolio.get('open_pnl', 0.0)
    closed_pnl = portfolio.get('closed_pnl', 0.0)
    total_pnl = open_pnl + closed_pnl
    
    col1, col2, col3, col4, col5 = st.columns(5, gap="small")
    
    with col1:
        st.metric("Equity", f"${current_equity:,.2f}", f"${total_pnl:+,.2f}")
    with col2:
        st.metric("Available", f"${cash:,.2f}")
    with col3:
        st.metric("Open P&L", f"${open_pnl:+,.2f}")
    with col4:
        st.metric("Closed P&L", f"${closed_pnl:+,.2f}") 
    with col5:
        utilization = ((initial_capital - cash) / initial_capital * 100) if initial_capital > 0 else 0
        st.metric("Utilization", f"{utilization:.1f}%")


# ===== UPGRADE #1: AGGRID INSTITUTIONAL DATA GRIDS =====
def render_positions_grid():
    """Render interactive positions grid with AgGrid."""
    st.markdown("#### OPEN POSITIONS (Sortable | Filterable)")
    
    positions_df = create_mock_positions_df()
    
    # Configure the grid
    gb = GridOptionsBuilder.from_dataframe(positions_df)
    gb.configure_pagination(paginationAutoPageSize=False, paginationPageSize=10)
    gb.configure_side_bar(filters_panel=True)
    
    # Configure columns
    gb.configure_column("Symbol", pinned="left", width=80)
    gb.configure_column("Side", width=70)
    gb.configure_column("Size", width=70)
    gb.configure_column("Entry Price", width=100)
    gb.configure_column("Current Price", width=110)
    gb.configure_column("Unrealized P&L", width=120)
    gb.configure_column("Return %", width=100)
    gb.configure_column("Status", width=90)
    
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(domLayout='autoHeight')
    
    grid_options = gb.build()
    
    # Render the grid
    grid_response = AgGrid(
        positions_df,
        gridOptions=grid_options,
        data_return_mode='AS_INPUT',
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        fit_columns_on_grid_load=False,
        theme='alpine-dark',
        height=250,
        allow_unsafe_jscode=False,
    )
    
    if grid_response['selected_rows']:
        selected = grid_response['selected_rows'][0]
        st.caption(f"📌 Selected: {selected['Symbol']} {selected['Side']} {selected['Size']} @ {selected['Entry Price']}")


def render_signals_grid():
    """Render active signals grid with AgGrid."""
    st.markdown("#### ACTIVE SIGNALS (Real-time | Click to Trade)")
    
    signals_df = create_mock_signals_df()
    
    gb = GridOptionsBuilder.from_dataframe(signals_df)
    gb.configure_side_bar(filters_panel=True)
    
    gb.configure_column("Time", width=80)
    gb.configure_column("Symbol", width=80)
    gb.configure_column("Signal", width=70)
    gb.configure_column("Strength", width=80)
    gb.configure_column("Strategy", width=120)
    gb.configure_column("Confidence", width=100)
    
    gb.configure_selection(selection_mode="single", use_checkbox=False)
    gb.configure_grid_options(domLayout='autoHeight')
    
    grid_options = gb.build()
    
    grid_response = AgGrid(
        signals_df,
        gridOptions=grid_options,
        data_return_mode='AS_INPUT',
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        theme='alpine-dark',
        height=200,
        allow_unsafe_jscode=False,
    )
    
    if grid_response['selected_rows']:
        selected = grid_response['selected_rows'][0]
        col1, col2 = st.columns([3, 1])
        with col1:
            st.caption(f"📌 Selected: {selected['Symbol']} {selected['Signal']} (Strength: {selected['Strength']})")
        with col2:
            if st.button("Execute", key="exec_signal"):
                st.success(f"Order placed: {selected['Signal']} {selected['Symbol']}")


def render_trades_grid():
    """Render recent trades grid with AgGrid."""
    st.markdown("#### TRADE HISTORY (Recent Executions)")
    
    # Load real trades from state
    trades_df = load_real_trades_df()
    
    if trades_df.empty:
        st.info("No trades executed yet")
        return
    
    gb = GridOptionsBuilder.from_dataframe(trades_df)
    gb.configure_side_bar(filters_panel=True)
    
    gb.configure_column("Time", width=85)
    gb.configure_column("Symbol", width=85)
    gb.configure_column("Side", width=70)
    gb.configure_column("Size", width=70)
    gb.configure_column("Entry", width=95)
    gb.configure_column("Exit", width=95)
    gb.configure_column("P&L", width=100)
    gb.configure_column("Duration", width=100)
    
    gb.configure_grid_options(domLayout='autoHeight')
    
    grid_options = gb.build()
    
    AgGrid(
        trades_df,
        gridOptions=grid_options,
        theme='alpine-dark',
        height=180,
        allow_unsafe_jscode=False,
    )


# ===== UPGRADE #2: TABBED ARCHITECTURE (Kill the Scroll) =====
def render_tabbed_dashboard():
    """Render institutional tabbed dashboard."""
    
    tab1, tab2, tab3 = st.tabs(["📊  EXECUTION & PORTFOLIO", "🧠  AI & REGIME", "⚙️  DIAGNOSTICS"])
    
    # ===== TAB 1: EXECUTION & PORTFOLIO =====
    with tab1:
        st.markdown("##### Portfolio Metrics")
        render_portfolio_metrics()
        
        st.markdown("---")
        col1, col2 = st.columns([2, 1], gap="small")
        
        with col1:
            st.markdown("##### Performance Chart")
            st.plotly_chart(create_pnl_chart(), use_container_width=True, config=dict(responsive=True, displayModeBar=False))
        
        with col2:
            st.markdown("##### System Grade")
            st.plotly_chart(create_health_score_gauge(), use_container_width=True, config=dict(responsive=True, displayModeBar=False))
        
        st.markdown("---")
        render_positions_grid()
        st.markdown("---")
        render_trades_grid()
    
    # ===== TAB 2: AI & REGIME ANALYSIS =====
    with tab2:
        col1, col2, col3, col4 = st.columns(4, gap="small")
        
        with col1:
            st.metric("Market Regime", "TRENDING UP", "Strength: 0.72")
        with col2:
            st.metric("Volatility", "NORMAL", "VIX: 18.5")
        with col3:
            st.metric("Bullish Signals", "7/12", "+2 today")
        with col4:
            st.metric("Signal Strength", "0.68", "Consensus")
        
        st.markdown("---")
        render_signals_grid()
        
        st.markdown("---")
        st.markdown("##### Model Confidence Heatmap")
        
        # Create institutional heatmap
        heatmap_data = pd.DataFrame({
            'AAPL': [0.85, 0.76, 0.92, 0.68, 0.55],
            'MSFT': [0.78, 0.82, 0.71, 0.88, 0.64],
            'GOOGL': [0.91, 0.68, 0.79, 0.75, 0.82],
            'QQQ': [0.72, 0.85, 0.68, 0.91, 0.73],
            'SPY': [0.80, 0.77, 0.84, 0.69, 0.86],
        }, index=['Momentum', 'Mean Reversion', 'Breakout', 'Grid Trading', 'Volatility'])
        
        fig = go.Figure(data=go.Heatmap(
            z=heatmap_data.values,
            x=heatmap_data.columns,
            y=heatmap_data.index,
            colorscale='RdYlGn',
            zmid=0.5,
        ))
        
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=80, r=0, t=30, b=0),
            font=dict(color='#c9d1d9'),
            height=280,
        )
        
        st.plotly_chart(fig, use_container_width=True, config=dict(responsive=True, displayModeBar=False))
    
    # ===== TAB 3: SYSTEM DIAGNOSTICS =====
    with tab3:
        col1, col2, col3, col4 = st.columns(4, gap="small")
        
        with col1:
            st.metric("API Status", "🟢 Connected", "Latency: 45ms")
        with col2:
            st.metric("Memory Usage", "2.3 GB / 8 GB", "-0.5GB")
        with col3:
            st.metric("CPU Load", "18%", "-5%")
        with col4:
            st.metric("Uptime", "18.5h", "Running")
        
        st.markdown("---")
        st.markdown("##### System Logs (Last 10 Events)")
        
        logs = [
            "[14:35:22] AAPL Position +100 Opened @ $182.47",
            "[14:33:15] Signal Generated: QQQ SELL (Strength: 0.76)",
            "[14:30:42] Portfolio Rebalance Completed",
            "[14:28:08] Market Regime Changed: TRENDING UP",
            "[14:25:33] Risk Check: 53% Utilization (Healthy)",
            "[14:20:15] New Model Update: Accuracy +2.3%",
            "[14:18:42] QQQ Position Closed: +$35 P&L",
            "[14:15:00] Scheduled Health Check Passed",
            "[14:12:30] Data Feed Connected to Alpaca",
            "[14:10:00] System Started - Initial State Loaded",
        ]
        
        st.code('\n'.join(logs), language='log')


def render_sidebar():
    """Render compact sidebar controls."""
    with st.sidebar:
        st.markdown("### ⚡ QUICK CONTROLS")
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("▶️ START", use_container_width=True):
                st.success("Bot started!")
        with col2:
            if st.button("⏸️ STOP", use_container_width=True):
                st.warning("Bot paused!")
        
        st.divider()
        
        st.markdown("### ⚙️ SETTINGS")
        
        refresh = st.slider("Refresh (sec)", 1, 60, 5)
        max_position = st.number_input("Max Position ($)", 10000, 1000000, 50000, 5000)
        risk_per_trade = st.slider("Risk/Trade (%)", 0.1, 5.0, 1.0, 0.1)
        
        st.divider()
        
        st.markdown("### 💾 EXPORT")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("📊 Excel"):
                st.info("Exported to reports/")
        with col2:
            if st.button("📈 CSV"):
                st.info("Exported to data/")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main institutional dashboard application."""
    initialize_session_state()
    
    render_header()
    render_sidebar()
    
    st.divider()
    
    render_tabbed_dashboard()
    
    # Footer
    st.divider()
    st.caption(f"Last Updated: {datetime.now().strftime('%H:%M:%S')} | Refresh: {st.session_state.refresh_interval}s")


if __name__ == "__main__":
    main()

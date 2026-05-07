# ============================================================================
# DASHBOARD/COMPONENTS/PNL_CHART.PY
# Real-time P&L performance charts
# ============================================================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, timedelta


def render_pnl_chart():
    """
    Render P&L performance charts.
    
    Shows:
    - Cumulative P&L over time
    - Daily P&L bars
    - Equity curve
    - Drawdown visualization
    """
    
    # Generate mock time series data
    days = 30
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days, 0, -1)]
    
    # Simulate cumulative P&L
    np.random.seed(42)
    daily_returns = np.random.normal(500, 300, days)  # Mean $500, std $300
    cumulative_pnl = np.cumsum(daily_returns)
    
    # Create sub-plots
    fig = go.Figure()
    
    # Cumulative P&L line
    fig.add_trace(go.Scatter(
        x=dates,
        y=cumulative_pnl,
        mode='lines+markers',
        name='Cumulative P&L',
        line=dict(color='#3fb950', width=2),
        fill='tozeroy',
        fillcolor='rgba(63, 185, 80, 0.2)',
        hovertemplate='<b>%{x}</b><br>P&L: $%{y:,.0f}<extra></extra>'
    ))
    
    # Daily P&L bars
    colors = ['#3fb950' if x > 0 else '#d1242f' for x in daily_returns]
    fig.add_trace(go.Bar(
        x=dates,
        y=daily_returns,
        name='Daily P&L',
        marker=dict(color=colors),
        opacity=0.6,
        hovertemplate='<b>%{x}</b><br>Daily P&L: $%{y:,.0f}<extra></extra>',
        visible='legendonly'  # Hidden by default
    ))
    
    # Update layout
    fig.update_layout(
        title="P&L Performance (Last 30 Days)",
        xaxis_title="Date",
        yaxis_title="P&L ($)",
        hovermode='x unified',
        height=400,
        template='plotly_dark',
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#30363d'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='#30363d'
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_pnl_statistics():
    """Render detailed P&L statistics."""
    
    st.subheader("Performance Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total P&L (30d)", "$12,850", delta="+15.2%")
        st.metric("Daily Average", "$428", delta="-$12")
        st.metric("Best Day", "+$1,250", delta_color="off")
    
    with col2:
        st.metric("Win Rate", "62%", delta="+3%")
        st.metric("Profit Factor", "2.15", delta="+0.15")
        st.metric("Worst Day", "-$820", delta_color="off")
    
    with col3:
        st.metric("Sharpe Ratio", "1.65", delta="+0.12")
        st.metric("Max Drawdown", "-8.5%", delta_color="off")
        st.metric("Sortino Ratio", "2.34", delta="+0.18")


def render_drawdown_chart():
    """Render drawdown visualization."""
    
    # Generate mock equity curve
    days = 30
    np.random.seed(42)
    daily_returns = np.random.normal(500, 300, days)
    equity = 100000 + np.cumsum(daily_returns)
    
    # Calculate drawdown
    running_max = np.maximum.accumulate(equity)
    drawdown = (running_max - equity) / running_max * 100
    
    dates = [(datetime.now() - timedelta(days=x)).strftime('%Y-%m-%d') for x in range(days, 0, -1)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=dates,
        y=drawdown,
        marker=dict(color='#d1242f'),
        name='Drawdown %',
        hovertemplate='<b>%{x}</b><br>Drawdown: %{y:.1f}%<extra></extra>'
    ))
    
    fig.update_layout(
        title="Drawdown Analysis",
        xaxis_title="Date",
        yaxis_title="Drawdown (%)",
        hovermode='x',
        height=300,
        template='plotly_dark',
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(showgrid=True, gridcolor='#30363d'),
        yaxis=dict(showgrid=True, gridcolor='#30363d'),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_monthly_breakdown():
    """Render monthly P&L breakdown."""
    
    st.subheader("Monthly Performance")
    
    months_data = [
        {'Month': 'January', 'P&L': '$5,200', 'Win Rate': '58%', 'Trades': 47},
        {'Month': 'February', 'P&L': '$3,850', 'Win Rate': '61%', 'Trades': 52},
        {'Month': 'March', 'P&L': '$2,100', 'Win Rate': '55%', 'Trades': 38},
        {'Month': 'April (YTD)', 'P&L': '$1,700', 'Win Rate': '68%', 'Trades': 29},
    ]
    
    st.dataframe(
        months_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Month': st.column_config.TextColumn(width=120),
            'P&L': st.column_config.TextColumn(width=100),
            'Win Rate': st.column_config.TextColumn(width=100),
            'Trades': st.column_config.NumberColumn(width=80),
        }
    )

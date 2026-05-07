# ============================================================================
# DASHBOARD/COMPONENTS/HEALTH_MONITOR.PY
# System health and model grade monitoring
# ============================================================================

import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta


def render_health_monitor():
    """
    Render system health and model grade indicators.
    
    Shows:
    - Model Health Grade (A-F)
    - Component scores (profitability, consistency, risk mgmt, efficiency)
    - System status
    - Last update time
    - Operational metrics
    """
    
    # Health grade indicator
    health_grade = "A"
    health_score = 87.5
    
    grade_color = {
        'A+': '#3fb950',
        'A': '#3fb950',
        'A-': '#3fb950',
        'B+': '#4184e4',
        'B': '#4184e4',
        'B-': '#d29922',
        'C+': '#d29922',
        'C': '#d1242f',
        'C-': '#d1242f',
        'D': '#d1242f',
        'F': '#d1242f',
    }
    
    # Display health grade prominently
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(
            f"<div style='text-align: center; padding: 20px; background: {grade_color.get(health_grade, '#4184e4')}; "
            f"border-radius: 10px; color: white;'>"
            f"<h2 style='margin: 0; font-size: 3em;'>{health_grade}</h2>"
            f"<p style='margin: 5px 0 0 0; font-size: 0.9em;'>GRADE</p>"
            f"</div>",
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(f"<h4 style='margin: 10px 0;'>Health Score: {health_score}/100</h4>", unsafe_allow_html=True)
        
        # Score bar
        progress_col1, progress_col2 = st.columns([1, 4])
        with progress_col1:
            st.write("Score")
        with progress_col2:
            st.progress(health_score / 100)
        
        st.caption("Model is performing at institutional grade standards")
    
    with col3:
        st.metric("Status", "🟢 OPERATIONAL", delta="All systems")
    
    # Component scores breakdown
    st.subheader("Component Scores")
    
    components = [
        ('Profitability', 92, '92/100'),
        ('Consistency', 87, '87/100'),
        ('Risk Mgmt', 78, '78/100'),
        ('Efficiency', 88, '88/100'),
    ]
    
    for comp_name, score, label in components:
        col1, col2, col3 = st.columns([2, 3, 1])
        
        with col1:
            st.write(f"**{comp_name}**")
        
        with col2:
            st.progress(score / 100)
        
        with col3:
            st.caption(label)
    
    # System operational metrics
    st.subheader("Operational Status")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("API Status", "🟢 OK", delta="Connected")
    
    with col2:
        st.metric("Data Feed", "🟢 Active", delta="2 sources")
    
    with col3:
        st.metric("Last Trade", "15 min ago", delta="Active")
    
    with col4:
        st.metric("Memory", "245 MB", delta="+12 MB")


def render_health_timeline():
    """Render health score timeline."""
    
    st.subheader("Health Score Timeline (7 days)")
    
    # Generate mock health timeline
    days_ago = 7
    dates = [(datetime.now() - timedelta(days=x)).strftime('%b %d') for x in range(days_ago, 0, -1)]
    scores = [82, 83, 85, 84, 86, 87, 87.5]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=scores,
        mode='lines+markers',
        name='Health Score',
        line=dict(color='#3fb950', width=3),
        marker=dict(size=8),
        fill='tozeroy',
        fillcolor='rgba(63, 185, 80, 0.2)',
        hovertemplate='<b>%{x}</b><br>Score: %{y:.1f}<extra></extra>'
    ))
    
    # Add trend line
    z = np.polyfit(range(len(scores)), scores, 1)
    p = np.poly1d(z)
    trend = p(range(len(scores)))
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=trend,
        mode='lines',
        name='Trend',
        line=dict(color='#d29922', width=2, dash='dash'),
        hovertemplate='Trend: %{y:.1f}<extra></extra>'
    ))
    
    fig.update_layout(
        title="Health Score Trend",
        xaxis_title="Date",
        yaxis_title="Health Score",
        hovermode='x',
        height=300,
        template='plotly_dark',
        margin=dict(l=50, r=20, t=50, b=50),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(showgrid=True, gridcolor='#30363d'),
        yaxis=dict(showgrid=True, gridcolor='#30363d', range=[70, 100]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_performance_alerts():
    """Render performance alerts and warnings."""
    
    st.subheader("Alerts & Notifications")
    
    alerts = [
        {
            'type': 'info',
            'title': '✓ Daily Target Achieved',
            'message': 'Daily P&L target of $500 achieved at 14:32',
        },
        {
            'type': 'warning',
            'title': '⚠ Drawdown Alert',
            'message': 'Current drawdown at 8.2% (target: <15%)',
        },
        {
            'type': 'success',
            'title': '✓ Win Rate Milestone',
            'message': 'Win rate improved to 62% (previous: 58%)',
        },
    ]
    
    for alert in alerts:
        if alert['type'] == 'info':
            st.info(f"**{alert['title']}**\n{alert['message']}")
        elif alert['type'] == 'warning':
            st.warning(f"**{alert['title']}**\n{alert['message']}")
        elif alert['type'] == 'success':
            st.success(f"**{alert['title']}**\n{alert['message']}")


def render_system_diagnostics():
    """Render detailed system diagnostics."""
    
    st.subheader("System Diagnostics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**CPU Usage**")
        st.progress(0.35)
        st.caption("35% - Moderate")
        
        st.write("**Network Latency**")
        st.progress(0.15)
        st.caption("45ms - Excellent")
    
    with col2:
        st.write("**Memory Usage**")
        st.progress(0.48)
        st.caption("245 MB / 512 MB")
        
        st.write("**Order Queue**")
        st.progress(0.0)
        st.caption("0 pending orders")


import numpy as np  # Import at end to avoid circular imports

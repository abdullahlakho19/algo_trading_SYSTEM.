# ============================================================================
# DASHBOARD/COMPONENTS/SIGNAL_METER.PY
# Signal strength gauge showing active signals and sentiment
# ============================================================================

import streamlit as st
import plotly.graph_objects as go


def render_signal_meter():
    """
    Render signal strength gauge and sentiment distribution.
    
    Shows:
    - Overall signal strength (0-100 scale)
    - Bullish vs Bearish vs Neutral signals count
    - Signal sources (strategies generating signals)
    """
    
    # Signal strength gauge
    signal_strength = 65
    
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number+delta",
        value=signal_strength,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Signal Strength"},
        delta={'reference': 60, 'suffix': " points"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#3fb950"},
            'steps': [
                {'range': [0, 25], 'color': "#d1242f"},
                {'range': [25, 50], 'color': "#d29922"},
                {'range': [50, 75], 'color': "#3fb950"},
                {'range': [75, 100], 'color': "#1f6feb"},
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 70
            }
        }
    )])
    
    fig.update_layout(
        height=300,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="#0e1117",
        font=dict(color="#c9d1d9"),
        plot_bgcolor="#0e1117",
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Signal sentiment breakdown
    st.subheader("Signal Sentiment")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🟢 Bullish Signals", "3", delta="+1")
    
    with col2:
        st.metric("⚫ Neutral Signals", "2", delta="No change")
    
    with col3:
        st.metric("🔴 Bearish Signals", "1", delta="-1")
    
    # Active signals table
    st.subheader("Active Signals")
    
    signals_data = [
        {
            'Strategy': 'Mean Reversion',
            'Signal': 'BUY',
            'Confidence': '85%',
            'Time': '2m ago',
            'Symbol': 'AAPL',
        },
        {
            'Strategy': 'Momentum',
            'Signal': 'BUY',
            'Confidence': '72%',
            'Time': '5m ago',
            'Symbol': 'AAPL',
        },
        {
            'Strategy': 'Volume Profile',
            'Signal': 'BUY',
            'Confidence': '68%',
            'Time': '8m ago',
            'Symbol': 'AAPL',
        },
        {
            'Strategy': 'Breakout',
            'Signal': 'SELL',
            'Confidence': '55%',
            'Time': '12m ago',
            'Symbol': 'QQQ',
        },
        {
            'Strategy': 'SMC/ICT',
            'Signal': 'NEUTRAL',
            'Confidence': '45%',
            'Time': '3m ago',
            'Symbol': 'SPY',
        },
    ]
    
    st.dataframe(
        signals_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Strategy': st.column_config.TextColumn(width=130),
            'Signal': st.column_config.TextColumn(width=80),
            'Confidence': st.column_config.TextColumn(width=90),
            'Time': st.column_config.TextColumn(width=90),
            'Symbol': st.column_config.TextColumn(width=80),
        }
    )
    
    # Signal quality metrics
    st.subheader("Signal Quality")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Signal/Noise Ratio", "3.2", delta="+0.3")
    
    with col2:
        st.metric("Avg Confidence", "65%", delta="-2%")
    
    with col3:
        st.metric("Hit Rate", "62%", delta="+4%")
    
    with col4:
        st.metric("False Positives", "38%", delta="-3%")

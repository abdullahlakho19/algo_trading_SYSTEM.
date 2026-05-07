# ============================================================================
# DASHBOARD/COMPONENTS/REGIME_INDICATOR.PY
# Market regime and volatility indicator
# ============================================================================

import streamlit as st
import plotly.graph_objects as go
import numpy as np


def render_regime_indicator():
    """
    Render market regime and volatility indicators.
    
    Shows:
    - Current market regime (trending up/down, ranging, etc.)
    - Regime strength
    - Volatility level (low, normal, high)
    - Market trend sentiment
    """
    
    # Regime gauge
    regime_strength = 0.72  # 72%
    
    fig = go.Figure(data=[go.Indicator(
        mode="gauge+number",
        value=regime_strength * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Regime Strength"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "#1f6feb"},
            'steps': [
                {'range': [0, 30], 'color': "#d1242f"},
                {'range': [30, 70], 'color': "#d29922"},
                {'range': [70, 100], 'color': "#3fb950"},
            ],
            'threshold': {
                'line': {'color': "white", 'width': 2},
                'thickness': 0.75,
                'value': 50
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
    
    # Regime details
    st.subheader("Regime Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Current Regime", "TRENDING UP", delta="Confirmed")
        st.metric("Trend Duration", "5 days", delta="+2 days")
        st.metric("Resistance Level", "$454.25", delta="-$2.10")
    
    with col2:
        st.metric("Volatility Level", "NORMAL", delta="Stable")
        st.metric("VIX Equivalent", "16.4", delta="+0.8")
        st.metric("Support Level", "$448.50", delta="+$1.80")
    
    # Regime transition probability
    st.subheader("Regime Transition Probabilities")
    
    transition_data = [
        {'Regime': 'Trending Up', 'Probability': '72%', 'Duration': '5 days'},
        {'Regime': 'Ranging', 'Probability': '18%', 'Duration': '~3 days'},
        {'Regime': 'Trending Down', 'Probability': '8%', 'Duration': '~2 days'},
        {'Regime': 'Gap Up', 'Probability': '2%', 'Duration': 'Immediate'},
    ]
    
    st.dataframe(
        transition_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Regime': st.column_config.TextColumn(width=130),
            'Probability': st.column_config.TextColumn(width=120),
            'Duration': st.column_config.TextColumn(width=120),
        }
    )


def render_volatility_indicator():
    """Render volatility time series."""
    
    st.subheader("Volatility Time Series")
    
    # Generate mock volatility data
    days = 20
    dates = [(f"Day {i}") for i in range(1, days + 1)]
    volatility = np.sin(np.linspace(0, 3*np.pi, days)) * 10 + 15
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=dates,
        y=volatility,
        mode='lines+markers',
        name='Volatility %',
        line=dict(color='#d29922', width=2),
        fill='tozeroy',
        fillcolor='rgba(210, 153, 34, 0.2)',
        hovertemplate='<b>%{x}</b><br>Vol: %{y:.1f}%<extra></extra>'
    ))
    
    fig.add_hline(
        y=20,
        line_dash="dash",
        line_color="#3fb950",
        annotation_text="Normal Range",
        annotation_position="right"
    )
    
    fig.add_hline(
        y=10,
        line_dash="dash",
        line_color="#d1242f",
        annotation_text="Low Signal",
        annotation_position="right"
    )
    
    fig.update_layout(
        title="Historical Volatility (20-day)",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode='x',
        height=300,
        template='plotly_dark',
        margin=dict(l=50, r=100, t=50, b=50),
        paper_bgcolor='#0e1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        xaxis=dict(showgrid=True, gridcolor='#30363d'),
        yaxis=dict(showgrid=True, gridcolor='#30363d'),
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_market_microstructure():
    """Render market microstructure analysis."""
    
    st.subheader("Market Microstructure")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Bid-Ask Spread", "0.02%", delta="-0.01%")
        st.metric("Spread Change", "-1 bp", delta_color="off")
        st.metric("Average Spread", "0.025%", delta_color="off")
    
    with col2:
        st.metric("Volume Profile", "Balanced", delta="Normal")
        st.metric("Buy/Sell Ratio", "1.15", delta="+0.05")
        st.metric("Imbalance", "15% Buy", delta="+3%")

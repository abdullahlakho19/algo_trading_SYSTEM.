# ============================================================================
# DASHBOARD/COMPONENTS/POSITIONS_TABLE.PY
# Real-time open positions table with P&L tracking
# ============================================================================

import streamlit as st
from datetime import datetime


def render_positions_table():
    """
    Render open positions table with real-time P&L.
    
    Shows:
    - Symbol, Side, Size, Entry Price, Current Price
    - Unrealized P&L ($, %), Time in Trade, Exit Price
    - Quick action buttons (Close, Modify SL/TP)
    """
    
    # Mock open positions
    positions_data = [
        {
            'Symbol': 'AAPL',
            'Side': 'LONG',
            'Size': '100',
            'Entry': '$182.50',
            'Current': '$183.45',
            'Unrealized P&L': '+$95.00',
            'Unrealized %': '+0.52%',
            'Time': '1h 23m',
            'Strategy': 'Mean Reversion',
            'Stop Loss': '$180.50',
            'Take Profit': '$185.00',
        },
        {
            'Symbol': 'SPY',
            'Side': 'SHORT',
            'Size': '50',
            'Entry': '$452.10',
            'Current': '$451.80',
            'Unrealized P&L': '+$15.00',
            'Unrealized %': '+0.07%',
            'Time': '2h 44m',
            'Strategy': 'Momentum',
            'Stop Loss': '$453.50',
            'Take Profit': '$450.00',
        },
        {
            'Symbol': 'QQQ',
            'Side': 'LONG',
            'Size': '25',
            'Entry': '$375.20',
            'Current': '$374.50',
            'Unrealized P&L': '-$17.50',
            'Unrealized %': '-0.19%',
            'Time': '45m',
            'Strategy': 'Breakout',
            'Stop Loss': '$373.50',
            'Take Profit': '$378.50',
        },
    ]
    
    if not positions_data:
        st.info("No open positions")
        return
    
    # Display as table
    st.dataframe(
        positions_data,
        use_container_width=True,
        hide_index=True,
        column_config={
            'Symbol': st.column_config.TextColumn(width=80),
            'Side': st.column_config.TextColumn(width=70),
            'Size': st.column_config.NumberColumn(width=70),
            'Entry': st.column_config.TextColumn(width=90),
            'Current': st.column_config.TextColumn(width=90),
            'Unrealized P&L': st.column_config.TextColumn(width=120),
            'Unrealized %': st.column_config.TextColumn(width=100),
            'Time': st.column_config.TextColumn(width=80),
            'Strategy': st.column_config.TextColumn(width=120),
            'Stop Loss': st.column_config.TextColumn(width=100),
            'Take Profit': st.column_config.TextColumn(width=100),
        }
    )
    
    # Position management section
    st.subheader("Position Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        selected_position = st.selectbox("Select Position", [p['Symbol'] for p in positions_data])
    
    with col2:
        action = st.selectbox("Action", ["Close Position", "Modify Stop Loss", "Modify Take Profit", "Add to Position"])
    
    with col3:
        if action == "Close Position":
            if st.button("Execute Close", key="close_position"):
                st.success(f"Position {selected_position} closed at market")
        elif action == "Modify Stop Loss":
            new_sl = st.number_input("New Stop Loss Price", value=100.0, step=0.01)
            if st.button("Update SL", key="update_sl"):
                st.success(f"Stop Loss updated for {selected_position} to ${new_sl:.2f}")
        elif action == "Modify Take Profit":
            new_tp = st.number_input("New Take Profit Price", value=100.0, step=0.01)
            if st.button("Update TP", key="update_tp"):
                st.success(f"Take Profit updated for {selected_position} to ${new_tp:.2f}")
        else:
            add_qty = st.number_input("Additional Quantity", value=1, step=1)
            if st.button("Add Position", key="add_position"):
                st.success(f"Added {add_qty} shares to {selected_position}")


def render_positions_summary():
    """Render summary statistics of all positions."""
    
    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Open Positions", 3)
    
    with col2:
        st.metric("Long Positions", 2)
    
    with col3:
        st.metric("Short Positions", 1)
    
    with col4:
        st.metric("Total Exposure", "$37,500")

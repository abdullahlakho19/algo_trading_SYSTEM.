"""
dashboard/pages/2_Manual_Trading.py
─────────────────────────────────────────────────────────────────────────────
Manual Trading Panel.
Trade manually alongside the bot — full risk calculation,
position management, and trade history.
─────────────────────────────────────────────────────────────────────────────
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

st.set_page_config(page_title="Manual Trading", page_icon="🖐️", layout="wide")

st.markdown("""
<style>
.stApp,.main{background:#0A0E1A!important}
h1,h2,h3,h4,p,label,div{color:#D0D6E0!important}
.stTextInput input,.stNumberInput input{background:#0D1B2A!important;
  color:#D0D6E0!important;border:1px solid #1E2A3A!important}
.stSelectbox>div>div{background:#0D1B2A!important;color:#D0D6E0!important}
.buy-btn>button{background:#00C853!important;color:#000!important;
  font-weight:bold!important;font-size:16px!important}
.sell-btn>button{background:#D50000!important;color:#fff!important;
  font-weight:bold!important;font-size:16px!important}
.close-btn>button{background:#FF9100!important;color:#000!important;font-weight:bold!important}
.pos-row{background:#0D1B2A;border-radius:6px;padding:10px 14px;margin:5px 0}
.risk-box{background:#0D2A0D;border:1px solid #00E676;border-radius:8px;padding:12px;margin:8px 0}
.warn-box{background:#2A0D0D;border:1px solid #FF5252;border-radius:8px;padding:12px;margin:8px 0}
.trade-hist{background:#0A0E1A;border:1px solid #1E2A3A;border-radius:6px;
            padding:6px 12px;margin:3px 0;font-size:12px}
</style>
""", unsafe_allow_html=True)


# ── Helpers ────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_sim():
    try:
        from execution.paper_simulator import PaperSimulator
        return PaperSimulator(initial_capital=100000.0, commission_rate=0.0005, slippage_bps=2.0)
    except Exception as e:
        st.error(f"Failed to load simulator: {e}")
        return None


def get_price(symbol: str) -> float:
    try:
        import yfinance as yf
        yf_map = {"EUR/USD":"EURUSD=X", "GBP/USD":"GBPUSD=X", "USD/JPY":"USDJPY=X"}
        sym = yf_map.get(symbol, symbol)
        t   = yf.Ticker(sym)
        h   = t.history(period="1d", interval="1m")
        if not h.empty:
            return float(h["Close"].iloc[-1])
    except Exception:
        pass
    return 0.0


def get_atr(symbol: str) -> float:
    try:
        import yfinance as yf
        yf_map = {"EUR/USD":"EURUSD=X", "GBP/USD":"GBPUSD=X", "USD/JPY":"USDJPY=X"}
        sym = yf_map.get(symbol, symbol)
        t   = yf.Ticker(sym)
        df  = t.history(period="14d", interval="1h")
        if df.empty: return 0.0
        df.columns=[c.lower() for c in df.columns]
        tr  = pd.concat([df["high"]-df["low"],
                         (df["high"]-df["close"].shift()).abs(),
                         (df["low"] -df["close"].shift()).abs()],axis=1).max(axis=1)
        return float(tr.ewm(span=14,adjust=False).mean().iloc[-1])
    except Exception:
        return 0.0


def calc_risk(entry, sl, qty, equity):
    sl_dist = abs(entry - sl)
    risk_amt = sl_dist * qty
    risk_pct = (risk_amt / equity * 100) if equity > 0 else 0
    return risk_amt, risk_pct


# ══════════════════════════════════════════════════════════════════════════════
st.markdown("## 🖐️ Manual Trading Panel")
st.caption("Trade manually alongside the bot · Full risk calculation · Paper mode")

sim = get_sim()
if sim is None:
    st.error("⚠️ Paper simulator not loaded. Make sure you are running from the trading_agent folder.")
    st.stop()

# ── Account banner ─────────────────────────────────────────────────────────────
status = sim.get_status()
equity = float(status.get("equity", 100000))
cash   = float(status.get("cash",   100000))
pnl    = float(status.get("total_pnl", 0))
pnl_pct= float(status.get("total_pnl_pct", 0))

ac1,ac2,ac3,ac4,ac5 = st.columns(5)
ac1.metric("Mode",    "📄 PAPER")
ac2.metric("Equity",  f"${equity:,.2f}")
ac3.metric("Cash",    f"${cash:,.2f}")
ac4.metric("P&L",     f"${pnl:,.2f}", f"{pnl_pct*100:+.2f}%",
           delta_color="normal" if pnl>=0 else "inverse")
ac5.metric("Positions", str(len(sim.open_positions)))

st.markdown("---")

# ══════════════════════════════════════════════════════════════════════════════
left_col, right_col = st.columns([1,1])

# ─────────────────────────────────────────────────────────────────────────────
# LEFT — ORDER ENTRY
# ─────────────────────────────────────────────────────────────────────────────
with left_col:
    st.markdown("### 📋 Place Manual Order")

    mkt = st.selectbox("Market", ["Stocks","Forex"], key="mkt")
    sym_map = {
        "Stocks":["AAPL","MSFT","NVDA","TSLA","GOOGL","AMZN","META","SPY","QQQ"],
        "Forex": ["EUR/USD","GBP/USD","USD/JPY","AUD/USD","GBP/JPY","USD/CAD"],
    }
    symbol = st.selectbox("Symbol", sym_map[mkt], key="sym")

    # Fetch live price
    if st.button("📡 Fetch Live Price", use_container_width=True):
        with st.spinner("Fetching..."):
            p = get_price(symbol)
            st.session_state["live_price"] = p
            st.session_state["live_atr"]   = get_atr(symbol)
            if p > 0:
                st.success(f"Live price: {p:,.6f}")
            else:
                st.warning("Could not fetch live price. Enter manually.")

    live_p   = st.session_state.get("live_price", 0.0)
    live_atr = st.session_state.get("live_atr", 0.0)

    entry_price = st.number_input("Entry Price", value=float(live_p) if live_p>0 else 0.0,
                                   format="%.6f", step=0.0001)
    direction = st.radio("Direction", ["LONG (Buy)", "SHORT (Sell)"],
                          horizontal=True)
    is_long = "LONG" in direction

    # Auto-suggest SL/TP from ATR
    atr_val = live_atr if live_atr > 0 else entry_price * 0.015
    default_sl = round(entry_price - atr_val*1.5, 6) if is_long and entry_price>0 else round(entry_price + atr_val*1.5, 6)
    default_tp = round(entry_price + atr_val*3.0, 6) if is_long and entry_price>0 else round(entry_price - atr_val*3.0, 6)

    col_sl, col_tp = st.columns(2)
    with col_sl:
        sl = st.number_input("Stop Loss", value=float(default_sl), format="%.6f", step=0.0001)
    with col_tp:
        tp = st.number_input("Take Profit", value=float(default_tp), format="%.6f", step=0.0001)

    # Quantity / Risk selector
    size_method = st.radio("Size by", ["Units","Risk %","Dollar Amount"], horizontal=True)

    if size_method == "Units":
        qty = st.number_input("Quantity (units)", value=0.01, min_value=0.0001,
                               format="%.4f", step=0.0001)
    elif size_method == "Risk %":
        risk_pct_input = st.slider("Risk % of equity", 0.1, 5.0, 1.0, 0.1)
        sl_dist = abs(entry_price - sl) if entry_price > 0 and sl > 0 else 1
        qty = round((equity * risk_pct_input/100) / sl_dist, 6) if sl_dist > 0 else 0.01
        st.info(f"Auto quantity: **{qty:.6f}** units")
    else:
        dollar_amt = st.number_input("Dollar amount ($)", value=1000.0, min_value=10.0)
        qty = round(dollar_amt / entry_price, 6) if entry_price > 0 else 0.01
        st.info(f"Auto quantity: **{qty:.6f}** units")

    # Risk calculation display
    if entry_price > 0 and sl > 0 and qty > 0:
        risk_amt, risk_pct_calc = calc_risk(entry_price, sl, qty, equity)
        rr_val = abs(tp - entry_price) / abs(entry_price - sl) if abs(entry_price-sl)>0 else 0
        pos_val = qty * entry_price

        if risk_pct_calc <= 2.0:
            box_class = "risk-box"
            icon = "✅"
        else:
            box_class = "warn-box"
            icon = "⚠️"

        st.markdown(f"""
        <div class="{box_class}">
            <b>{icon} Risk Analysis</b><br>
            Position Value: <b>${pos_val:,.2f}</b> &nbsp;|&nbsp;
            Risk Amount: <b>${risk_amt:,.2f}</b> &nbsp;|&nbsp;
            Risk: <b>{risk_pct_calc:.2f}%</b><br>
            Risk/Reward: <b>1:{rr_val:.2f}</b> &nbsp;|&nbsp;
            {"<span style='color:#00E676'>R/R is acceptable ✓</span>" if rr_val>=2 else "<span style='color:#FF5252'>R/R below 1:2 ✗</span>"}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("")

    # BUY / SELL buttons
    bc, sc = st.columns(2)
    with bc:
        st.markdown('<div class="buy-btn">', unsafe_allow_html=True)
        buy_clicked = st.button("🟢 BUY / LONG", use_container_width=True,
                                key="buy_btn", type="primary")
        st.markdown('</div>', unsafe_allow_html=True)
    with sc:
        st.markdown('<div class="sell-btn">', unsafe_allow_html=True)
        sell_clicked = st.button("🔴 SELL / SHORT", use_container_width=True,
                                  key="sell_btn")
        st.markdown('</div>', unsafe_allow_html=True)

    # Execute order
    if (buy_clicked or sell_clicked) and entry_price > 0 and qty > 0:
        side      = "buy"  if buy_clicked  else "sell"
        direction_str = "long" if buy_clicked else "short"
        with st.spinner("Executing order..."):
            order = sim.place_order(
                symbol=symbol, side=side, qty=qty,
                order_type="market", stop_loss=sl,
                take_profit=tp, current_price=entry_price,
                market=mkt.lower(), direction=direction_str,
            )
        if order and order.status == "filled":
            fill = order.filled_price or entry_price
            st.success(f"""
            ✅ **Order Filled!**
            {symbol} {side.upper()} {qty:.6f} @ {fill:,.6f}
            SL: {sl:,.6f} | TP: {tp:,.6f}
            """)
            st.cache_data.clear()
        else:
            st.error(f"Order rejected: {order.status if order else 'unknown error'}")

    st.markdown("---")

    # Close position section
    st.markdown("### 🔒 Close a Position")
    if sim.open_positions:
        pos_to_close = st.selectbox(
            "Select position to close",
            list(sim.open_positions.keys()),
            key="close_sel"
        )
        close_price = st.number_input("Close at price (0 = market)", value=0.0,
                                       format="%.6f", step=0.0001)
        if st.button("❌ Close Position", use_container_width=True):
            pos = sim.open_positions.get(pos_to_close)
            if pos:
                cp = close_price if close_price > 0 else float(pos.current_price or entry_price)
                trade = sim._close_position(pos_to_close, cp, "manual_close")
                if trade:
                    color = "success" if trade.pnl >= 0 else "error"
                    getattr(st, color)(
                        f"Position closed: {pos_to_close} | P&L: ${trade.pnl:,.2f}"
                    )
    else:
        st.info("No open positions to close.")


# ─────────────────────────────────────────────────────────────────────────────
# RIGHT — POSITIONS + HISTORY
# ─────────────────────────────────────────────────────────────────────────────
with right_col:
    st.markdown("### 🔓 Open Positions")

    if sim.open_positions:
        for sym_key, pos in sim.open_positions.items():
            pnl_color = "#00E676" if pos.unrealised_pnl >= 0 else "#FF5252"
            progress  = 0.5
            if pos.direction == "long" and pos.take_profit > pos.stop_loss:
                rng = pos.take_profit - pos.stop_loss
                progress = max(0, min(1, (pos.current_price - pos.stop_loss)/rng))
            elif pos.direction == "short" and pos.stop_loss > pos.take_profit:
                rng = pos.stop_loss - pos.take_profit
                progress = max(0, min(1, (pos.stop_loss - pos.current_price)/rng))

            bar_clr = "#00E676" if progress > 0.5 else "#FF9100" if progress > 0.25 else "#FF5252"
            dir_icon = "▲" if pos.direction == "long" else "▼"
            dir_clr  = "#00D4FF" if pos.direction == "long" else "#FF9100"

            st.markdown(f"""
            <div class="pos-row">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="color:#D0D6E0;font-size:14px;font-weight:bold">{pos.symbol}</span>
                    <span style="color:{dir_clr};font-size:11px;background:#0A0E1A;
                          padding:2px 8px;border-radius:3px">{dir_icon} {pos.direction.upper()}</span>
                    <span style="color:{pnl_color};font-size:15px;font-weight:bold">
                        ${pos.unrealised_pnl:+,.2f}
                    </span>
                </div>
                <div style="display:flex;gap:16px;margin-top:6px;font-size:10px;color:#8A94A8">
                    <span>Entry: <b style="color:#D0D6E0">{pos.entry_price:.5f}</b></span>
                    <span>Now: <b style="color:#D0D6E0">{pos.current_price:.5f}</b></span>
                    <span>Qty: <b style="color:#D0D6E0">{pos.qty:.4f}</b></span>
                    <span>SL: <b style="color:#FF5252">{pos.stop_loss:.5f}</b></span>
                    <span>TP: <b style="color:#00E676">{pos.take_profit:.5f}</b></span>
                </div>
                <div style="margin-top:6px">
                    <div style="background:#1E2A3A;border-radius:3px;height:5px">
                        <div style="background:{bar_clr};width:{int(progress*100)}%;
                                    height:100%;border-radius:3px"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                font-size:9px;color:#4A5A6A;margin-top:2px">
                        <span>SL</span>
                        <span>TP Progress: {int(progress*100)}%</span>
                        <span>TP</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Update prices button
        if st.button("🔄 Update Position Prices", use_container_width=True):
            pf = {}
            for s in sim.open_positions:
                p = get_price(s)
                if p > 0: pf[s] = p
            if pf:
                closed = sim.update_positions(pf)
                for t in closed:
                    if t.pnl >= 0:
                        st.success(f"✅ {t.symbol} TP hit | P&L: ${t.pnl:,.2f}")
                    else:
                        st.warning(f"🛑 {t.symbol} SL hit | P&L: ${t.pnl:,.2f}")
    else:
        st.info("No open positions. Place a trade on the left.")

    st.markdown("---")
    st.markdown("### 📋 Recent Trade History")

    if sim.closed_trades:
        recent = list(reversed(sim.closed_trades[-15:]))
        for t in recent:
            pnl_clr  = "#00E676" if t.pnl >= 0 else "#FF5252"
            outcome_icon = "✅" if t.outcome == "win" else "❌"
            dir_icon = "▲" if t.direction == "long" else "▼"
            ts = t.closed_at.strftime("%m/%d %H:%M") if hasattr(t.closed_at,"strftime") else ""
            st.markdown(f"""
            <div class="trade-hist">
                <span style="color:#C9A84C;font-weight:bold">{outcome_icon} {t.symbol}</span>
                <span style="color:#8A94A8"> {dir_icon} {t.direction.upper()}</span>
                <span style="color:{pnl_clr};font-weight:bold;float:right">${t.pnl:+,.2f}</span>
                <span style="color:#546E7A;font-size:10px;margin-left:8px">
                    {t.entry_price:.5f}→{t.exit_price:.5f} | {t.exit_reason} | {ts}
                </span>
            </div>
            """, unsafe_allow_html=True)

        # Performance summary
        st.markdown("---")
        perf = sim.get_performance()
        p1,p2,p3,p4 = st.columns(4)
        p1.metric("Win Rate",     f"{perf.get('win_rate',0)*100:.1f}%")
        p2.metric("Profit Factor",f"{perf.get('profit_factor',0):.2f}")
        p3.metric("Sharpe",       f"{perf.get('sharpe_ratio',0):.2f}")
        p4.metric("Total Trades", perf.get("total_trades",0))
    else:
        st.info("No closed trades yet.")

    st.markdown("---")
    if st.button("📊 Export to Excel", use_container_width=True):
        if sim.closed_trades:
            try:
                from reporting.excel_exporter import excel_exporter
                path = excel_exporter.export(
                    closed_trades=sim.closed_trades,
                    portfolio_value=sim.equity,
                    initial_capital=sim.initial_capital,
                )
                st.success(f"Exported: {path.name}")
            except Exception as e:
                st.error(f"Export failed: {e}")
        else:
            st.warning("No trades to export.")
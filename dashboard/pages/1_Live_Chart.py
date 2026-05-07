"""
dashboard/pages/1_Live_Chart.py
─────────────────────────────────────────────────────────────────────────────
Live Chart Analysis — TradingView-style.
Shows everything the bot sees:
  Candlesticks · EMAs · VWAP · Bollinger Bands · Volume Profile
  Candlestick Patterns · BoS/CHoCH · Probability Score · Buy/Sell Signals
  RSI · MACD · ADX · Stochastic · Volume · Backtesting overlay
Run:  streamlit run dashboard/app.py
─────────────────────────────────────────────────────────────────────────────
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Load environment variables for API keys
from dotenv import load_dotenv
load_dotenv()

st.set_page_config(
    page_title="Live Chart Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Dark CSS ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp,.main{background:#0A0E1A!important}
h1,h2,h3,h4,p,li,span,label{color:#D0D6E0!important}
.stSelectbox>div>div{background:#0D1B2A!important;color:#D0D6E0!important}
.stSlider>div{color:#C9A84C!important}
.signal-box{padding:10px 14px;border-radius:6px;margin:4px 0;font-size:13px}
.buy-signal{background:#0D2A0D;border:1.5px solid #00E676;color:#00E676}
.sell-signal{background:#2A0D0D;border:1.5px solid #FF5252;color:#FF5252}
.neutral-signal{background:#1A1A0D;border:1.5px solid #C9A84C;color:#C9A84C}
.metric-dark{background:#0D1B2A;border:1px solid #1E2A3A;border-radius:8px;
             padding:12px;text-align:center;margin:4px}
.metric-val{font-size:20px;font-weight:bold;color:#C9A84C}
.metric-lbl{font-size:10px;color:#8A94A8;margin-top:2px}
.prob-bar-bg{background:#1E2A3A;border-radius:4px;height:12px;margin-top:4px}
.indicator-row{background:#0D1B2A;border-bottom:1px solid #1E2A3A;
               padding:8px 12px;display:flex;justify-content:space-between}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING — yFinance (Primary)
# ══════════════════════════════════════════════════════════════════════════════

def _get_market_type(symbol: str) -> str:
    """Determine market type: stocks or forex."""
    if "/" in symbol:
        return "forex"
    else:
        return "stocks"

@st.cache_data(ttl=60, show_spinner=True, max_entries=50)
def load_ohlcv(symbol: str, tf: str) -> pd.DataFrame:
    """
    Load OHLCV data using yFinance.
    
    Args:
        symbol: Ticker (e.g., "AAPL", "EUR/USD=X")
        tf: Timeframe (1m, 5m, 15m, 1h, 4h, 1d, 1wk, 1mo)
    
    Returns:
        DataFrame with OHLCV columns, UTC-localized index, cleaned data
    """
    from data_feeds.yfinance_feed import yfinance_feed
    from datetime import datetime, timedelta
    
    try:
        # Calculate lookback days based on timeframe
        lookback_map = {
            "1m": 7,
            "5m": 60,
            "15m": 60,
            "1h": 365,
            "4h": 365,
            "1d": 730,
            "1wk": 1460,
            "1mo": 2920,
        }
        lookback_days = lookback_map.get(tf, 365)
        
        # Fetch from yFinance
        df = yfinance_feed.get_bars(
            symbol=symbol,
            timeframe=tf,
            lookback_days=lookback_days,
        )
        
        if df is None or df.empty:
            st.toast(f"No data for {symbol}", icon="❌")
            return pd.DataFrame()
        
        # Normalize columns to lowercase
        df.columns = df.columns.str.lower()
        
        # Track data source
        st.session_state['data_source'] = f"yFinance ({_get_market_type(symbol)})"
        
        return df
    
    except Exception as e:
        st.toast(f"yFinance error: {str(e)[:50]}", icon="⚠️")
        return pd.DataFrame()


# ══════════════════════════════════════════════════════════════════════════════
# INDICATOR ENGINE
# ══════════════════════════════════════════════════════════════════════════════
def indicators(df: pd.DataFrame) -> dict:
    c,h,l,v = df["close"],df["high"],df["low"],df["volume"]
    I = {}
    # EMAs
    for s in [9,20,50,200]:
        I[f"ema{s}"] = c.ewm(span=s,adjust=False).mean()
    # VWAP
    tp = (h+l+c)/3
    I["vwap"] = (tp*v).cumsum()/(v.cumsum()+1e-10)
    # Bollinger
    ma20=c.rolling(20).mean(); sd20=c.rolling(20).std()
    I["bb_u"]=ma20+2*sd20; I["bb_l"]=ma20-2*sd20; I["bb_m"]=ma20
    # RSI
    d=c.diff(); g=d.clip(lower=0).ewm(span=14,adjust=False).mean()
    ls=(-d.clip(upper=0)).ewm(span=14,adjust=False).mean()
    I["rsi"]=100-100/(1+g/(ls+1e-10))
    # Stochastic
    ll14=l.rolling(14).min(); hh14=h.rolling(14).max()
    I["stoch_k"]=100*(c-ll14)/(hh14-ll14+1e-10)
    I["stoch_d"]=I["stoch_k"].rolling(3).mean()
    # MACD
    e12=c.ewm(span=12,adjust=False).mean(); e26=c.ewm(span=26,adjust=False).mean()
    I["macd"]=e12-e26; I["macd_sig"]=I["macd"].ewm(span=9,adjust=False).mean()
    I["macd_h"]=I["macd"]-I["macd_sig"]
    # ATR
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    I["atr"]=tr.ewm(span=14,adjust=False).mean()
    # ADX
    pdm=h.diff().clip(lower=0); mdm=(-l.diff()).clip(lower=0)
    atr_s=tr.ewm(span=14,adjust=False).mean()
    pdi=100*pdm.ewm(span=14,adjust=False).mean()/(atr_s+1e-10)
    mdi=100*mdm.ewm(span=14,adjust=False).mean()/(atr_s+1e-10)
    dx=100*(pdi-mdi).abs()/(pdi+mdi+1e-10)
    I["adx"]=dx.ewm(span=14,adjust=False).mean()
    I["pdi"]=pdi; I["mdi"]=mdi
    # OBV
    obv=np.where(c>c.shift(),v,np.where(c<c.shift(),-v,0))
    I["obv"]=pd.Series(obv,index=df.index).cumsum()
    return I


def volume_profile(df: pd.DataFrame, bins=50) -> dict:
    lo,hi = df["low"].min(), df["high"].max()
    edges = np.linspace(lo,hi,bins+1)
    centers=(edges[:-1]+edges[1:])/2
    vols=np.zeros(bins)
    for _,row in df.tail(100).iterrows():
        lo_i=max(0,np.searchsorted(edges,row["low"],"left")-1)
        hi_i=min(bins,np.searchsorted(edges,row["high"],"right"))
        n=hi_i-lo_i
        if n>0: vols[lo_i:hi_i]+=row["volume"]/n
    poc_i=int(np.argmax(vols)); poc=float(centers[poc_i])
    tot=vols.sum(); tgt=tot*0.70
    u=l=poc_i; cur=vols[poc_i]
    while cur<tgt:
        cu=u+1<bins; cd=l-1>=0
        if not cu and not cd: break
        vu=vols[u+1] if cu else -1; vd=vols[l-1] if cd else -1
        if vu>=vd: u+=1; cur+=vu
        else: l-=1; cur+=vd
    return {"poc":poc,"vah":float(centers[u]),"val":float(centers[l]),
            "centers":centers.tolist(),"vols":vols.tolist()}


def detect_patterns(df: pd.DataFrame) -> list[dict]:
    pats=[]
    if len(df)<5: return pats
    for i in range(-min(80,len(df)-2),0):
        c=df.iloc[i]; c1=df.iloc[i-1]
        o,h,l,cl=float(c["open"]),float(c["high"]),float(c["low"]),float(c["close"])
        o1,h1,l1,cl1=float(c1["open"]),float(c1["high"]),float(c1["low"]),float(c1["close"])
        body=abs(cl-o); hl=h-l+1e-10
        up=h-max(o,cl); dn=min(o,cl)-l
        ts=df.index[i]
        if body/hl<0.08:
            pats.append({"t":ts,"p":cl,"n":"Doji","c":"#C9A84C","d":"neutral","i":"◆"})
        if dn>body*2 and up<body*0.5 and cl>o:
            pats.append({"t":ts,"p":l-hl*0.02,"n":"Hammer","c":"#00E676","d":"bullish","i":"▲"})
        if up>body*2 and dn<body*0.5 and cl<o:
            pats.append({"t":ts,"p":h+hl*0.02,"n":"Shooting Star","c":"#FF5252","d":"bearish","i":"▼"})
        if cl1<o1 and cl>o and cl>o1 and o<cl1:
            pats.append({"t":ts,"p":l-hl*0.02,"n":"Bull Engulf","c":"#00E676","d":"bullish","i":"▲▲"})
        if cl1>o1 and cl<o and cl<o1 and o>cl1:
            pats.append({"t":ts,"p":h+hl*0.02,"n":"Bear Engulf","c":"#FF5252","d":"bearish","i":"▼▼"})
        if up<hl*0.03 and dn<hl*0.03 and body>hl*0.90:
            d="bullish" if cl>o else "bearish"
            pats.append({"t":ts,"p":(h+l)/2,"n":"Marubozu",
                         "c":"#00E676" if d=="bullish" else "#FF5252","d":d,"i":"█"})
        if i>=-3:
            c2=df.iloc[i-2]
            o2,cl2=float(c2["open"]),float(c2["close"])
            if cl2<o2 and abs(cl1-o1)/hl<0.15 and cl>o and cl>(o2+cl2)/2:
                pats.append({"t":ts,"p":l-hl*0.02,"n":"Morning Star","c":"#00E676","d":"bullish","i":"★"})
            if cl2>o2 and abs(cl1-o1)/hl<0.15 and cl<o and cl<(o2+cl2)/2:
                pats.append({"t":ts,"p":h+hl*0.02,"n":"Evening Star","c":"#FF5252","d":"bearish","i":"★"})
    return pats[-20:]


def detect_structure(df: pd.DataFrame) -> list[dict]:
    evts=[]; n=5
    if len(df)<n*3: return evts
    highs=df["high"].values; lows=df["low"].values
    for i in range(n,len(df)-n):
        if highs[i]==max(highs[i-n:i+n+1]):
            evts.append({"type":"sh","idx":i,"price":highs[i],"t":df.index[i]})
        if lows[i]==min(lows[i-n:i+n+1]):
            evts.append({"type":"sl","idx":i,"price":lows[i],"t":df.index[i]})
    lc=float(df["close"].iloc[-1])
    sh=[e for e in evts if e["type"]=="sh"]
    sl=[e for e in evts if e["type"]=="sl"]
    if sh and lc>sh[-1]["price"]:
        evts.append({"type":"bos_bull","price":sh[-1]["price"],"t":df.index[-1]})
    if sl and lc<sl[-1]["price"]:
        evts.append({"type":"bos_bear","price":sl[-1]["price"],"t":df.index[-1]})
    return evts


def probability_score(df: pd.DataFrame, I: dict, vp_data: dict) -> dict:
    """Calculate signal probability and generate recommendation."""
    c=df["close"]; lc=float(c.iloc[-1])
    scores={}
    # EMA alignment
    e20=float(I["ema20"].iloc[-1]); e50=float(I["ema50"].iloc[-1]); e200=float(I["ema200"].iloc[-1])
    bull_ema = sum([lc>e20, e20>e50, e50>e200])
    bear_ema = sum([lc<e20, e20<e50, e50<e200])
    scores["ema"]=(bull_ema-bear_ema)/3
    # RSI
    rsi=float(I["rsi"].iloc[-1])
    scores["rsi"]=0.5 if 45<rsi<55 else 0.8 if 50<rsi<70 else -0.8 if 30<rsi<50 else 0.6 if rsi>70 else -0.6
    # MACD
    mh=float(I["macd_h"].iloc[-1]); pmh=float(I["macd_h"].iloc[-2])
    scores["macd"]=0.7 if mh>0 and mh>pmh else -0.7 if mh<0 and mh<pmh else 0.3 if mh>0 else -0.3
    # Volume Profile
    poc=vp_data["poc"]; vah=vp_data["vah"]; val=vp_data["val"]
    if lc>poc: scores["vp"]=0.5
    elif lc<poc: scores["vp"]=-0.5
    else: scores["vp"]=0
    # ADX trend
    adx=float(I["adx"].iloc[-1]); pdi=float(I["pdi"].iloc[-1]); mdi=float(I["mdi"].iloc[-1])
    if adx>25:
        scores["adx"]=0.6 if pdi>mdi else -0.6
    else: scores["adx"]=0
    # VWAP
    vwap=float(I["vwap"].iloc[-1])
    scores["vwap"]=0.4 if lc>vwap else -0.4
    # Stochastic
    sk=float(I["stoch_k"].iloc[-1]); sd=float(I["stoch_d"].iloc[-1])
    scores["stoch"]=0.5 if sk>sd and sk<80 else -0.5 if sk<sd and sk>20 else 0

    # Weighted composite
    weights={"ema":0.25,"rsi":0.15,"macd":0.20,"vp":0.15,"adx":0.10,"vwap":0.10,"stoch":0.05}
    composite=sum(scores[k]*weights[k] for k in weights)
    # Convert to 0-100 probability
    long_prob  = round((composite+1)/2*100,1)
    short_prob = round(100-long_prob,1)
    confluence = sum(1 for v in scores.values() if v>0.3)
    anti_conf  = sum(1 for v in scores.values() if v<-0.3)

    if composite>0.35 and confluence>=4:
        signal="STRONG BUY"; clr="#00E676"; icon="🟢"
    elif composite>0.15 and confluence>=3:
        signal="BUY"; clr="#69F0AE"; icon="🟩"
    elif composite<-0.35 and anti_conf>=4:
        signal="STRONG SELL"; clr="#FF5252"; icon="🔴"
    elif composite<-0.15 and anti_conf>=3:
        signal="SELL"; clr="#FF8A80"; icon="🟥"
    else:
        signal="NEUTRAL / WAIT"; clr="#C9A84C"; icon="🟡"

    # SL/TP suggestions
    atr=float(I["atr"].iloc[-1])
    if "BUY" in signal:
        sl=round(lc-atr*1.5,6); tp=round(lc+atr*3.0,6)
    elif "SELL" in signal:
        sl=round(lc+atr*1.5,6); tp=round(lc-atr*3.0,6)
    else:
        sl=round(lc-atr*1.5,6); tp=round(lc+atr*3.0,6)
    rr=abs(tp-lc)/abs(sl-lc) if abs(sl-lc)>0 else 0

    return {
        "signal":signal,"color":clr,"icon":icon,
        "long_prob":long_prob,"short_prob":short_prob,
        "composite":composite,"confluence":confluence,
        "scores":scores,"rsi":rsi,"adx":adx,
        "sl":sl,"tp":tp,"rr":round(rr,2),
        "entry":round(lc,6),"atr":round(atr,6),
    }


def backtest_signals(df: pd.DataFrame, I: dict) -> pd.DataFrame:
    """Generate historical buy/sell signals for backtest overlay."""
    close=df["close"]; rsi=I["rsi"]; macd_h=I["macd_h"]
    e20=I["ema20"]; e50=I["ema50"]
    signals=[]
    for i in range(50,len(df)):
        lc=float(close.iloc[i]); e2=float(e20.iloc[i]); e5=float(e50.iloc[i])
        r=float(rsi.iloc[i]); mh=float(macd_h.iloc[i]); pmh=float(macd_h.iloc[i-1])
        bull_score=sum([lc>e2,e2>e5,r>50,mh>0,mh>pmh])
        bear_score=sum([lc<e2,e2<e5,r<50,mh<0,mh<pmh])
        if bull_score>=4:
            signals.append({"time":df.index[i],"price":float(df["low"].iloc[i]),"type":"buy"})
        elif bear_score>=4:
            signals.append({"time":df.index[i],"price":float(df["high"].iloc[i]),"type":"sell"})
    return signals[-30:]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHART BUILDER
# ══════════════════════════════════════════════════════════════════════════════
def build_main_chart(df, I, vp_data, patterns, structure, bt_signals, opts):
    nrows=5; heights=[0.50,0.12,0.12,0.13,0.13]
    fig=make_subplots(rows=nrows,cols=1,row_heights=heights,
                      shared_xaxes=True,vertical_spacing=0.01,
                      subplot_titles=("","Volume","RSI","MACD","ADX"))

    # Candlesticks
    fig.add_trace(go.Candlestick(
        x=df.index,open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="OHLC",
        increasing=dict(line=dict(color="#00E676",width=1),fillcolor="#00E676"),
        decreasing=dict(line=dict(color="#FF5252",width=1),fillcolor="#FF5252"),
        showlegend=False,
    ),row=1,col=1)

    # EMAs
    if opts.get("ema"):
        for s,nm,clr,w in [(9,"EMA9","#FF9100",1.2),(20,"EMA20","#00D4FF",1.5),
                           (50,"EMA50","#C9A84C",1.5),(200,"EMA200","#E040FB",1.8)]:
            if s in [9,20] or opts.get("ema_all"):
                fig.add_trace(go.Scatter(x=df.index,y=I[f"ema{s}"],name=nm,
                    line=dict(color=clr,width=w),opacity=0.85),row=1,col=1)

    # VWAP
    if opts.get("vwap"):
        fig.add_trace(go.Scatter(x=df.index,y=I["vwap"],name="VWAP",
            line=dict(color="#E040FB",width=1.5,dash="dash"),opacity=0.9),row=1,col=1)

    # Bollinger
    if opts.get("bb"):
        fig.add_trace(go.Scatter(x=df.index,y=I["bb_u"],name="BB Upper",
            line=dict(color="#546E7A",width=0.8,dash="dot"),opacity=0.7,showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df.index,y=I["bb_l"],name="BB Lower",
            line=dict(color="#546E7A",width=0.8,dash="dot"),opacity=0.7,
            fill="tonexty",fillcolor="rgba(84,110,122,0.08)",showlegend=False),row=1,col=1)

    # Volume Profile lines
    if opts.get("vp"):
        for lvl,clr,lbl,dash in [
            (vp_data["poc"],"#FF9100",f"POC {vp_data['poc']:.2f}","solid"),
            (vp_data["vah"],"#00E676",f"VAH {vp_data['vah']:.2f}","dash"),
            (vp_data["val"],"#FF5252",f"VAL {vp_data['val']:.2f}","dash"),
        ]:
            fig.add_hline(y=lvl,line_dash=dash,line_color=clr,line_width=1.5,
                annotation_text=lbl,annotation_font_color=clr,annotation_font_size=10,
                row=1,col=1)

    # Patterns
    if opts.get("patterns"):
        bull_p=[p for p in patterns if p["d"]=="bullish"]
        bear_p=[p for p in patterns if p["d"]=="bearish"]
        neut_p=[p for p in patterns if p["d"]=="neutral"]
        for grp,sym,clr,pos in [
            (bull_p,"triangle-up","#00E676","top center"),
            (bear_p,"triangle-down","#FF5252","bottom center"),
            (neut_p,"diamond","#C9A84C","top center"),
        ]:
            if grp:
                fig.add_trace(go.Scatter(
                    x=[p["t"] for p in grp],y=[p["p"] for p in grp],
                    mode="markers+text",
                    marker=dict(symbol=sym,size=11,color=clr,
                                line=dict(color="#0A0E1A",width=1)),
                    text=[p["n"] for p in grp],textposition=pos,
                    textfont=dict(size=8,color=clr),
                    name="Pattern",showlegend=False,
                ),row=1,col=1)

    # Structure
    if opts.get("structure"):
        sh=[e for e in structure if e["type"]=="sh"][-5:]
        sl=[e for e in structure if e["type"]=="sl"][-5:]
        if sh:
            fig.add_trace(go.Scatter(x=[e["t"] for e in sh],
                y=[e["price"]*1.003 for e in sh],mode="markers",
                marker=dict(symbol="triangle-down-open",size=10,color="#FF5252"),
                name="Swing H",showlegend=False),row=1,col=1)
        if sl:
            fig.add_trace(go.Scatter(x=[e["t"] for e in sl],
                y=[e["price"]*0.997 for e in sl],mode="markers",
                marker=dict(symbol="triangle-up-open",size=10,color="#00E676"),
                name="Swing L",showlegend=False),row=1,col=1)
        for e in structure:
            if "bos" in e["type"]:
                clr="#00E676" if "bull" in e["type"] else "#FF5252"
                fig.add_hline(y=e["price"],line_dash="dot",line_color=clr,
                    line_width=1,annotation_text="BoS",
                    annotation_font_color=clr,annotation_font_size=9,row=1,col=1)

    # Backtest signals overlay
    if opts.get("backtest") and bt_signals:
        buys=[s for s in bt_signals if s["type"]=="buy"]
        sells=[s for s in bt_signals if s["type"]=="sell"]
        if buys:
            fig.add_trace(go.Scatter(
                x=[s["time"] for s in buys],y=[s["price"] for s in buys],
                mode="markers",name="BT Buy",
                marker=dict(symbol="triangle-up",size=13,color="#00C853",
                            line=dict(color="#0A0E1A",width=1)),
                showlegend=True),row=1,col=1)
        if sells:
            fig.add_trace(go.Scatter(
                x=[s["time"] for s in sells],y=[s["price"] for s in sells],
                mode="markers",name="BT Sell",
                marker=dict(symbol="triangle-down",size=13,color="#D50000",
                            line=dict(color="#0A0E1A",width=1)),
                showlegend=True),row=1,col=1)

    # Volume
    vol_clr=["#00E676" if c>=o else "#FF5252"
              for c,o in zip(df["close"],df["open"])]
    fig.add_trace(go.Bar(x=df.index,y=df["volume"],name="Vol",
        marker_color=vol_clr,opacity=0.7,showlegend=False),row=2,col=1)
    # Volume MA
    fig.add_trace(go.Scatter(x=df.index,y=df["volume"].rolling(20).mean(),
        name="Vol MA",line=dict(color="#FF9100",width=1),showlegend=False),row=2,col=1)

    # RSI
    fig.add_trace(go.Scatter(x=df.index,y=I["rsi"],name="RSI",
        line=dict(color="#00D4FF",width=1.5),showlegend=False),row=3,col=1)
    for lvl,clr in [(70,"#FF5252"),(50,"#546E7A"),(30,"#00E676")]:
        fig.add_hline(y=lvl,line_dash="dash",line_color=clr,
                      line_width=0.8,row=3,col=1)
    fig.add_hrect(y0=70,y1=100,fillcolor="rgba(255,82,82,0.05)",
                  line_width=0,row=3,col=1)
    fig.add_hrect(y0=0,y1=30,fillcolor="rgba(0,230,118,0.05)",
                  line_width=0,row=3,col=1)

    # MACD
    mh_clr=["#00E676" if v>=0 else "#FF5252" for v in I["macd_h"]]
    fig.add_trace(go.Bar(x=df.index,y=I["macd_h"],name="MACD H",
        marker_color=mh_clr,opacity=0.8,showlegend=False),row=4,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=I["macd"],name="MACD",
        line=dict(color="#00D4FF",width=1.2),showlegend=False),row=4,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=I["macd_sig"],name="Signal",
        line=dict(color="#FF9100",width=1.2),showlegend=False),row=4,col=1)

    # ADX
    fig.add_trace(go.Scatter(x=df.index,y=I["adx"],name="ADX",
        line=dict(color="#C9A84C",width=1.5),showlegend=False),row=5,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=I["pdi"],name="+DI",
        line=dict(color="#00E676",width=1),showlegend=False),row=5,col=1)
    fig.add_trace(go.Scatter(x=df.index,y=I["mdi"],name="-DI",
        line=dict(color="#FF5252",width=1),showlegend=False),row=5,col=1)
    fig.add_hline(y=25,line_dash="dash",line_color="#546E7A",
                  line_width=0.8,row=5,col=1)

    fig.update_layout(
        template="plotly_dark",paper_bgcolor="#0A0E1A",plot_bgcolor="#0D1B2A",
        height=820,showlegend=True,
        legend=dict(bgcolor="rgba(13,27,42,0.85)",bordercolor="#1E2A3A",
                    font=dict(color="#D0D6E0",size=9),x=0.01,y=0.99,
                    orientation="h"),
        margin=dict(l=0,r=10,t=20,b=0),
        xaxis_rangeslider_visible=False,
    )
    for i in range(1,6):
        fig.update_xaxes(gridcolor="#1E2A3A",showgrid=True,row=i,col=1)
        fig.update_yaxes(gridcolor="#1E2A3A",showgrid=True,
                         tickfont=dict(color="#8A94A8"),row=i,col=1)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 📊 Chart Controls [Stocks & Forex Only]")
    market   = st.selectbox("Market", ["Stocks","Forex"])
    symbol   = st.selectbox("Symbol", SYMBOL_LISTS[market])
    timeframe= st.selectbox("Timeframe",["1m","5m","15m","1h","4h","1d"],index=3)

    st.markdown("---")
    st.markdown("**Overlays**")
    o_ema      = st.toggle("EMAs",              value=True)
    o_ema_all  = st.toggle("EMA 50 & 200",      value=True)
    o_vwap     = st.toggle("VWAP",              value=True)
    o_bb       = st.toggle("Bollinger Bands",   value=False)
    o_vp       = st.toggle("Volume Profile",    value=True)
    o_patterns = st.toggle("Candle Patterns",   value=True)
    o_structure= st.toggle("Market Structure",  value=True)
    o_backtest = st.toggle("Backtest Signals",  value=True)

    st.markdown("---")
    auto_ref = st.toggle("Auto Refresh (60s)", value=False)
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.cache_data.clear(); st.rerun()

opts = dict(ema=o_ema,ema_all=o_ema_all,vwap=o_vwap,bb=o_bb,
            vp=o_vp,patterns=o_patterns,structure=o_structure,backtest=o_backtest)

# ══════════════════════════════════════════════════════════════════════════════
# LOAD & COMPUTE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"## 📈 {symbol}  ·  {timeframe}  ·  Live Analysis")

with st.spinner("Loading chart data..."):
    df = load_ohlcv(symbol, timeframe)

if df is None or len(df)<30:
    st.error("⚠️ INSTITUTIONAL DATA LOAD FAILED: Could not fetch data from Alpaca. Verify API keys in .env (ALPACA_API_KEY, ALPACA_SECRET_KEY) or check Alpaca connectivity.")
    st.stop()

I         = indicators(df)
vp_data   = volume_profile(df)
patterns  = detect_patterns(df) if o_patterns else []
structure = detect_structure(df) if o_structure else []
bt_signals= backtest_signals(df,I) if o_backtest else []
prob      = probability_score(df, I, vp_data)

# ══════════════════════════════════════════════════════════════════════════════
# TOP ROW — SIGNAL + METRICS
# ══════════════════════════════════════════════════════════════════════════════
lc  = float(df["close"].iloc[-1])
pc  = float(df["close"].iloc[-2])
chg = (lc-pc)/pc*100

# Signal banner
sig_clr_map = {
    "STRONG BUY":"#00E676","BUY":"#69F0AE",
    "STRONG SELL":"#FF5252","SELL":"#FF8A80","NEUTRAL / WAIT":"#C9A84C"
}
sig_bg_map = {
    "STRONG BUY":"#0D2A0D","BUY":"#0A1E0A",
    "STRONG SELL":"#2A0D0D","SELL":"#1E0A0A","NEUTRAL / WAIT":"#1A1A0D"
}
sig_clr = sig_clr_map.get(prob["signal"],"#C9A84C")
sig_bg  = sig_bg_map.get(prob["signal"],"#1A1A0D")

st.markdown(f"""
<div style="background:{sig_bg};border:2px solid {sig_clr};border-radius:10px;
            padding:14px 20px;margin:6px 0;display:flex;
            align-items:center;justify-content:space-between">
    <div>
        <span style="font-size:26px;font-weight:bold;color:{sig_clr}">
            {prob["icon"]}  {prob["signal"]}
        </span>
        <span style="color:#8A94A8;font-size:12px;margin-left:16px">
            Confluence: <b style="color:{sig_clr}">{prob["confluence"]}/7 signals aligned</b>
        </span>
    </div>
    <div style="text-align:right">
        <div style="color:#8A94A8;font-size:11px">Long Probability</div>
        <div style="background:#1E2A3A;border-radius:4px;height:10px;width:160px;margin-top:3px">
            <div style="background:{sig_clr};width:{prob['long_prob']}%;height:100%;
                        border-radius:4px"></div>
        </div>
        <div style="color:{sig_clr};font-size:13px;font-weight:bold;margin-top:2px">
            {prob['long_prob']}%
        </div>
    </div>
    <div style="text-align:right;font-size:12px;color:#8A94A8">
        Entry: <b style="color:#D0D6E0">{prob['entry']}</b><br>
        SL: <b style="color:#FF5252">{prob['sl']}</b><br>
        TP: <b style="color:#00E676">{prob['tp']}</b><br>
        R/R: <b style="color:#C9A84C">1:{prob['rr']}</b>
    </div>
</div>
""", unsafe_allow_html=True)

# Metrics row
c1,c2,c3,c4,c5,c6,c7,c8 = st.columns(8)
def metric_card(label, val, sub="", color="#C9A84C"):
    return f"""<div class="metric-dark">
        <div class="metric-val" style="color:{color}">{val}</div>
        <div class="metric-lbl">{label}</div>
        {"<div style='font-size:9px;color:#546E7A'>"+sub+"</div>" if sub else ""}
    </div>"""

c1.markdown(metric_card("PRICE", f"{lc:,.4f}",
    f"{'▲' if chg>=0 else '▼'}{chg:+.2f}%","#00E676" if chg>=0 else "#FF5252"),
    unsafe_allow_html=True)
c2.markdown(metric_card("RSI(14)", f"{float(I['rsi'].iloc[-1]):.1f}",
    "OB>70 OS<30","#FF5252" if float(I['rsi'].iloc[-1])>70
    else "#00E676" if float(I['rsi'].iloc[-1])<30 else "#C9A84C"),
    unsafe_allow_html=True)
c3.markdown(metric_card("ADX", f"{float(I['adx'].iloc[-1]):.1f}",
    "Trend>25","#00E676" if float(I['adx'].iloc[-1])>25 else "#8A94A8"),
    unsafe_allow_html=True)
c4.markdown(metric_card("ATR", f"{float(I['atr'].iloc[-1]):.4f}","Volatility"),
    unsafe_allow_html=True)
c5.markdown(metric_card("POC", f"{vp_data['poc']:,.4f}","Most traded price"),
    unsafe_allow_html=True)
c6.markdown(metric_card("VAH", f"{vp_data['vah']:,.4f}","Value Area High","#00E676"),
    unsafe_allow_html=True)
c7.markdown(metric_card("VAL", f"{vp_data['val']:,.4f}","Value Area Low","#FF5252"),
    unsafe_allow_html=True)
c8.markdown(metric_card("VWAP", f"{float(I['vwap'].iloc[-1]):,.4f}",
    "Institutional ref","#E040FB"), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MAIN CHART
# ══════════════════════════════════════════════════════════════════════════════
fig = build_main_chart(df,I,vp_data,patterns,structure,bt_signals,opts)
st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ANALYSIS PANELS BELOW CHART
# ══════════════════════════════════════════════════════════════════════════════
col_L, col_M, col_R = st.columns(3)

# ── Signal Component Breakdown ─────────────────────────────────────────────────
with col_L:
    st.markdown("#### 🎯 Signal Components")
    comp_names = {
        "ema":"EMA Alignment","rsi":"RSI Momentum","macd":"MACD Cross",
        "vp":"Volume Profile","adx":"ADX Trend","vwap":"VWAP Position","stoch":"Stochastic"
    }
    for key, score in prob["scores"].items():
        name = comp_names.get(key, key)
        clr  = "#00E676" if score > 0.2 else "#FF5252" if score < -0.2 else "#C9A84C"
        icon = "▲" if score>0.2 else "▼" if score<-0.2 else "—"
        bar_w = int(abs(score)*100)
        bar_clr = "#00E676" if score>0 else "#FF5252"
        st.markdown(f"""
        <div style="margin:3px 0">
            <div style="display:flex;justify-content:space-between;font-size:11px">
                <span style="color:#D0D6E0">{name}</span>
                <span style="color:{clr};font-weight:bold">{icon} {score:+.2f}</span>
            </div>
            <div style="background:#1E2A3A;border-radius:3px;height:5px">
                <div style="background:{bar_clr};width:{bar_w}%;height:100%;
                            border-radius:3px;opacity:0.8"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Patterns Panel ─────────────────────────────────────────────────────────────
with col_M:
    st.markdown("#### 🕯️ Detected Patterns")
    if patterns:
        shown = patterns[-10:][::-1]
        for p in shown:
            css = ("buy-signal" if p["d"]=="bullish"
                   else "sell-signal" if p["d"]=="bearish" else "neutral-signal")
            ts  = p["t"].strftime("%m/%d %H:%M") if hasattr(p["t"],"strftime") else ""
            st.markdown(
                f'<div class="signal-box {css}">'
                f'<b>{p["i"]} {p["n"]}</b>'
                f'<span style="float:right;font-size:10px;opacity:0.7">{ts}</span>'
                f'</div>', unsafe_allow_html=True)
    else:
        st.info("No significant patterns in recent candles.")

# ── Market Structure ───────────────────────────────────────────────────────────
with col_R:
    st.markdown("#### 🏗️ Market Structure & Levels")
    sh=[e for e in structure if e["type"]=="sh"]
    sl=[e for e in structure if e["type"]=="sl"]
    bos=[e for e in structure if "bos" in e["type"]]

    if sh:
        st.markdown(f'<div class="signal-box sell-signal">▼ Last Swing High: <b>{sh[-1]["price"]:,.4f}</b></div>',
                    unsafe_allow_html=True)
    if len(sh)>1:
        st.markdown(f'<div class="signal-box sell-signal" style="opacity:0.6">▼ Prev Swing High: <b>{sh[-2]["price"]:,.4f}</b></div>',
                    unsafe_allow_html=True)
    if sl:
        st.markdown(f'<div class="signal-box buy-signal">▲ Last Swing Low: <b>{sl[-1]["price"]:,.4f}</b></div>',
                    unsafe_allow_html=True)
    if len(sl)>1:
        st.markdown(f'<div class="signal-box buy-signal" style="opacity:0.6">▲ Prev Swing Low: <b>{sl[-2]["price"]:,.4f}</b></div>',
                    unsafe_allow_html=True)
    for b in bos:
        css = "buy-signal" if "bull" in b["type"] else "sell-signal"
        lbl = "✅ Bullish BoS" if "bull" in b["type"] else "❌ Bearish BoS"
        st.markdown(f'<div class="signal-box {css}"><b>{lbl}</b> @ {b["price"]:,.4f}</div>',
                    unsafe_allow_html=True)
    if not sh and not sl:
        st.info("Need more data for structure detection.")

# ══════════════════════════════════════════════════════════════════════════════
# BACKTEST STATS
# ══════════════════════════════════════════════════════════════════════════════
if o_backtest and bt_signals:
    st.markdown("---")
    st.markdown("#### 📊 Backtest Signal Summary (last 30 signals on this chart)")
    buys  = [s for s in bt_signals if s["type"]=="buy"]
    sells = [s for s in bt_signals if s["type"]=="sell"]
    bc1,bc2,bc3,bc4 = st.columns(4)
    bc1.metric("Total Signals", len(bt_signals))
    bc2.metric("Buy Signals",   len(buys), delta="bullish" if len(buys)>len(sells) else None)
    bc3.metric("Sell Signals",  len(sells))
    bc4.metric("Signal Ratio",  f"{len(buys)/(len(bt_signals)+1e-10)*100:.0f}% Long")

# Auto-refresh
if auto_ref:
    import time; time.sleep(60)
    st.cache_data.clear(); st.rerun()
import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import google.generativeai as genai
from datetime import datetime

# --- 初期設定 ---
st.set_page_config(page_title="アンチグラビティ・コア Pro+ 押し目特化", layout="wide")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ システム設定")
    gemini_key      = st.text_input("Gemini API Key", type="password")
    discord_webhook = st.text_input("Discord Webhook URL", type="password")

    st.markdown("---")
    st.subheader("🎯 スイング条件設定（押し目反発）")
    # BB位置の厳格化パラメータを追加（0%が-2σ、50%がセンター、100%が+2σ）
    bb_pos_max  = st.slider("BB位置 許容上限 (%) ※0=下限(-2σ)", 5, 50, 25, help="25%以下なら-2σ〜-1σの「怖い押し目」圏内")
    rsi_max     = st.slider("RSI 上限（過熱排除）",       50, 70,  50)
    rsi_min     = st.slider("RSI 下限（下落排除）",       25, 45,  35)
    ma200_range = st.slider("200日線乖離 上限 (%)",        1, 30,  20)
    vol_mult    = st.slider("出来高急増 除外倍率（5日比）", 1.5, 5.0, 2.0, step=0.1)
    stop_pct    = st.slider("損切りライン (%)",            1, 10,   4)
    target_pct  = st.slider("利確ライン (%)",              2, 20,   8)

    st.markdown("---")
    st.subheader("💡 押し目絶対条件")
    st.info(f"""
**【以下の条件を全て満たさないと買い候補になりません】**

1. 株価 > **200日線**（上昇トレンド維持）
2. **25日線が上向き**（中期トレンド維持）
3. BB位置 **{bb_pos_max}%以下**（下限タッチ必須）
4. RSI **{rsi_min}〜{rsi_max}**（売られすぎ圏）
5. 直近**陰線続き**（怖い押し目）
6. 出来高**急増なし**（{vol_mult}倍未満）
    """)
    st.markdown("---")
    if st.button("🔄 全データをリセット"):
        st.session_state.analysis_results = None
        st.rerun()

# ================================================================
# 指標計算
# ================================================================
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs    = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ema_fast = data['Close'].ewm(span=fast, adjust=False).mean()
    ema_slow = data['Close'].ewm(span=slow, adjust=False).mean()
    macd     = ema_fast - ema_slow
    sig      = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig

def calculate_bb(data, window=20, num_std=2):
    mid = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

def calculate_ma25(data):
    return data['Close'].rolling(25).mean()

# ================================================================
# 押し目判定ヘルパー
# ================================================================
def check_consecutive_bearish(hist, n=2):
    count = 0
    for i in range(1, min(6, len(hist))):
        row = hist.iloc[-i]
        if row['Close'] < row['Open']:
            count += 1
        else:
            break
    return count >= n, count

def check_reversal_sign(hist):
    signs = []
    latest = hist.iloc[-1]
    prev   = hist.iloc[-2]

    body   = abs(latest['Close'] - latest['Open'])
    lower_wick = min(latest['Close'], latest['Open']) - latest['Low']
    
    if latest['Close'] > latest['Open'] and body > 0 and lower_wick >= body * 1.5:
        signs.append("下ヒゲ陽線")
    if (prev['Close'] < prev['Open'] and latest['Close'] > latest['Open']
            and latest['Close'] > prev['Open'] and latest['Open'] < prev['Close']):
        signs.append("包み足")
    if prev['Close'] < prev['Open'] and latest['Close'] > latest['Open']:
        if "包み足" not in signs:
            signs.append("陽線転換")
    if body > 0 and lower_wick >= body * 2.0 and "下ヒゲ陽線" not in signs:
        signs.append("長い下ヒゲ")

    return signs

def check_ma25_slope(hist, window=5):
    ma25 = hist['Close'].rolling(25).mean()
    if len(ma25.dropna()) < window + 1:
        return False
    slope = ma25.iloc[-1] - ma25.iloc[-(window+1)]
    return slope > 0

# ================================================================
# バックテスト（押し目反発版・条件厳格化）
# ================================================================
def backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, vol_mult, bb_pos_max):
    hist = hist.copy().reset_index()
    trades   = []
    in_trade = False
    entry_price = 0.0

    for i in range(201, len(hist) - 1):
        r     = hist.iloc[i]
        price = r['Close']
        ma200 = r['MA200']
        ma25  = r['MA25']
        rsi   = r['RSI']
        bb_up = r['BB_upper']
        bb_mid= r['BB_mid']
        bb_lo = r['BB_lower']
        vol_ratio = r['VolRatio']

        if pd.isna(ma200) or pd.isna(rsi) or pd.isna(ma25) or pd.isna(bb_up):
            continue

        ma25_5ago = hist.iloc[i - 5]['MA25'] if i >= 5 else None
        ma25_slope_up = (ma25_5ago is not None) and (not pd.isna(ma25_5ago)) and (ma25 > ma25_5ago)

        bearish_count = 0
        for _j in range(1, 5):
            if i >= _j:
                _r = hist.iloc[i - _j + 1]
                if _r['Close'] < _r['Open']:
                    bearish_count += 1
                else:
                    break

        vol_cal

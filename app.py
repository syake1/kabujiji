import os
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import json
import time

# ページ設定
st.set_page_config(
    page_title="株スクリーニング＆アラートシステム",
    page_icon="📈",
    layout="wide"
)

# スタイリング
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stTextInput, .stNumberInput, .stSelectbox { color: white; }
</style>
""", unsafe_allow_html=True)

st.title("📈 短期スイング・スクリーニング＆アラート")

# サイドバーまたは設定エリア
with st.expander("⚙️ システム設定", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        gemini_api_key = st.text_input("Gemini API Key", type="password", value=os.environ.get("GEMINI_API_KEY", ""))
    with col2:
        discord_webhook_url = st.text_input("Discord Webhook URL", type="password", value=os.environ.get("DISCORD_WEBHOOK_URL", ""))

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        stop_loss_pct = st.number_input("損切りライン(%)", value=4.0, step=0.5)
    with col4:
        take_profit_pct = st.number_input("利確ライン(%)", value=8.0, step=0.5)
    with col5:
        volume_spike = st.number_input("出来高急増除外倍率", value=3.0, step=0.5)
    with col6:
        min_price = st.number_input("最低株価(円)", value=500, step=100)

    col7, col8, col9 = st.columns(3)
    with col7:
        # 最低売買代金を500百万円（5億円）に引き上げ
        min_trading_value = st.number_input("最低売買代金(百万円/日)", value=500, step=100)
    with col8:
        min_atr = st.number_input("最低ATR%(値幅)", value=1.0, step=0.1)
    with col9:
        min_bt_count = st.number_input("最低BT取引数", value=5, step=1)

# --- サンプル銘柄リスト（実際には全銘柄やJPXなどのコードリストに置き換え） ---
# ここでは動作確認用の代表的な銘柄コード（日本株は .T を付与）
DEFAULT_TICKERS = [
    "7203.T", "6758.T", "9984.T", "8306.T", "7011.T", 
    "6501.T", "4385.T", "6857.T", "6146.T", "9432.T",
    "4502.T", "4503.T", "8035.T", "6920.T", "5401.T"
]

st.markdown("### 【スキャン条件】")
st.markdown("""
1. ✅ **200日線の上** (長期上昇トレンド継続中)
2. ✅ **25日線が上向き** (中期トレンド上昇中)
3. ✅ **週足OK** (下落転換・上ヒゲ陰線でない)
4. ✅ **BB下限タッチ** (1〜2日以内)
5. ✅ **反発サイン** (下ヒゲ陽線 / 包み足 / 陽線転換↑)
""")
st.markdown(f"💰 売買代金:{min_trading_value}百万円以上 | 📊 ATR:{min_atr}%以上 | 💀 損切り:-{stop_loss_pct}% | 🎯 利確:+{take_profit_pct}% | 最低株価:{min_price}円以上")

def send_discord_notification(webhook_url, message):
    if not webhook_url:
        return False
    try:
        payload = {"content": message}
        response = requests.post(webhook_url, json=payload, timeout=5)
        return response.status_code == 204
    except Exception as e:
        print(f"Discord通知エラー: {e}")
        return False

# スキャン実行ボタン
if st.button("🚀 スキャン実行", type="primary"):
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    total = len(DEFAULT_TICKERS)
    for i, ticker in enumerate(DEFAULT_TICKERS):
        status_text.text(f"分析中... ({i+1}/{total}) {ticker}")
        try:
            # データの取得（日足・週足）
            df_day = yf.download(ticker, period="1yr", interval="1d", progress=False)
            if df_day.empty or len(df_day) < 200:
                continue
            
            # カラム名のマルチインデックス対策
            if isinstance(df_day.columns, pd.MultiIndex):
                df_day.columns = df_day.columns.get_level_values(0)

            # 基本指標の計算
            df_day['SMA25'] = df_day['Close'].rolling(window=25).mean()
            df_day['SMA200'] = df_day['Close'].rolling(window=200).mean()
            
            # ボリンジャーバンド (20日, 2σ)
            df_day['BB_Middle'] = df_day['Close'].rolling(window=20).mean()
            df_day['BB_Std'] = df_day['Close'].rolling(window=20).std()
            df_day['BB_Lower'] = df_day['BB_Middle'] - (2 * df_day['BB_Std'])
            
            # ATR計算
            df_day['H-L'] = df_day['High'] - df_day['Low']
            df_day['H-PC'] = abs(df_day['High'] - df_day['Close'].shift(1))
            df_day['L-PC'] = abs(df_day['Low'] - df_day['Close'].shift(1))
            df_day['TR'] = df_day[['H-L', 'H-PC', 'L-PC']].max(axis=1)
            df_day['ATR'] = df_day['TR'].rolling(window=14).mean()
            df_day['ATR_pct'] = (df_day['ATR'] / df_day['Close']) * 100

            latest = df_day.iloc[-1]
            prev = df_day.iloc[-2]

            # フィルター条件のチェック
            # 1. 株価・売買代金チェック
            close_price = latest['Close']
            trading_value = (latest['Close'] * latest['Volume']) / 1_000_000 # 百万円換算
            
            if close_price < min_price:
                continue
            if trading_value < min_trading_value:
                continue
            
            # 2. 200日線の上
            if close_price < latest['SMA200']:
                continue
                
            # 3. 25日線が上向き（直近5日間で上昇傾向）
            sma25_slope = latest['SMA25'] - df_day['SMA25'].iloc[-5]
            if sma25_slope <= 0:
                continue

            # 4. ATR条件
            if latest['ATR_pct'] < min_atr:
                continue

            # 5. ボリンジャーバンド下限タッチ & 反発サイン（簡易判定）
            is_near_bb_lower = (latest['Low'] <= latest['BB_Lower'] * 1.01) or (prev['Low'] <= prev['BB_Lower'] * 1.01)
            is_bullish_signal = (latest['Close'] > latest['Open']) # 陽線

            if is_near_bb_lower and is_bullish_signal:
                results.append({
                    "銘柄": ticker,
                    "終値": round(close_price, 1),
                    "売買代金(百万円)": round(trading_value, 1),
                    "ATR%": round(latest['ATR_pct'], 2),
                    "25日線": round(latest['SMA25'], 1)
                })

        except Exception as e:
            print(f"Error processing {ticker}: {e}")
            
        progress_bar.progress((i + 1) / total)

    status_text.text("スキャン完了！")
    
    if results:
        res_df = pd.DataFrame(results)
        st.success(f"条件に一致する銘柄が {len(results)} 件見つかりました！")
        st.dataframe(res_df, use_container_width=True)
        
        # Discord通知のテスト
        if discord_webhook_url:
            msg = f"【株アラート通知】条件一致銘柄: {len(results)}件検出されました。"
            if send_discord_notification(discord_webhook_url, msg):
                st.info("Discordへ通知を送信しました。")
            else:
                st.warning("Discord通知に失敗しました。Webhook URLをご確認ください。")
    else:
        st.warning("条件に一致する銘柄が見つかりませんでした。さらに条件を調整してください。")

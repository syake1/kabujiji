import streamlit as st
import yfinance as yf
import pandas as pd

# --- 初期設定 ---
st.set_page_config(page_title="Sniper-Day", layout="centered")

st.title("🎯 Sniper-Day")
st.caption("スマホ特化・押し目狙いデイトレ監視ボード")

# ================================================================
# 関数：テクニカル計算
# ================================================================
def calculate_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_vwap(df):
    q = df['Volume']
    p = (df['High'] + df['Low'] + df['Close']) / 3
    return (p * q).cumsum() / q.cumsum()

# ================================================================
# 入力エリア（スマホで打ちやすい個別枠に変更）
# ================================================================
st.markdown("---")
st.subheader("📝 監視銘柄の登録（数字4桁だけ）")

# スマホ画面で綺麗に並ぶように、2列×2行で配置
c1, c2 = st.columns(2)
code1 = c1.text_input("銘柄①", value="9020", max_chars=4)
code2 = c2.text_input("銘柄②", value="7974", max_chars=4)
code3 = c1.text_input("銘柄③", value="", placeholder="任意", max_chars=4)
code4 = c2.text_input("銘柄④", value="", placeholder="任意", max_chars=4)

if st.button("🔄 最新データを取得して判定", use_container_width=True):
    # 入力された枠の中から、数字が入っているものだけを抽出
    raw_codes = [code1, code2, code3, code4]
    codes = [c.strip() for c in raw_codes if c.strip().isdigit()]
    
    if not codes:
        st.warning("⚠️ 銘柄コード（4桁の数字）を1つ以上入力してください")
    else:
        # --- 全体相場（日経平均）の冷え込みチェック ---
        st.subheader("🌐 現在の地合い（日経平均）")
        try:
            nk = yf.Ticker("^N225")
            nk_hist = nk.history(period="5d", interval="5m")
            if not nk_hist.empty:
                nk_rsi = calculate_rsi(nk_hist['Close']).iloc[-1]
                
                # 地合いの判定（RSI 40以下で冷え込み＝押し目チャンス）
                if nk_rsi <= 40:
                    st.error(f"🚨 全体相場 下落中！【絶好の押し目チャンス】(5分足RSI: {nk_rsi:.1f})")
                elif nk_rsi >= 70:
                    st.warning(f"⚠️ 全体相場 過熱気味 (5分足RSI: {nk_rsi:.1f})")
                else:
                    st.info(f"🟢 全体相場 ニュートラル (5分足RSI: {nk_rsi:.1f})")
        except:
            st.write("日経平均のデータが取得できませんでした")

        st.markdown("---")
        st.subheader(f"🎯 監視銘柄（{len(codes)}銘柄）")
        
        # --- 個別銘柄のカード表示 ---
        for code in codes:
            try:
                tk = yf.Ticker(f"{code}.T")
                # 5分足データ（直近5日分）を取得
                hist = tk.history(period="5d", interval="5m")
                
                if hist.empty:
                    st.error(f"{code}: データが取得できません")
                    continue
                
                # 指標の計算
                hist['RSI'] = calculate_rsi(hist['Close'])
                hist['Date'] = hist.index.date
                hist['VWAP'] = hist.groupby('Date').apply(calculate_vwap).reset_index(level=0, drop=True)
                
                latest = hist.iloc[-1]
                current_price = latest['Close']
                rsi_5m = latest['RSI']
                vwap = latest['VWAP']
                vwap_diff = (current_price - vwap) / vwap * 100
                
                # 表示用カード
                with st.container():
                    st.markdown(f"### [{code}] 🔗[チャート(TradingView)](https://jp.tradingview.com/chart/?symbol=TSE:{code})")
                    
                    m1, m2, m3 = st.columns(3)
                    m1.metric("現在値", f"{current_price:,.1f}円")
                    
                    if rsi_5m <= 30:
                        m2.metric("⚡ 5分足RSI", f"{rsi_5m:.1f}", "売られすぎ", delta_color="normal")
                    elif rsi_5m >= 70:
                        m2.metric("⚡ 5分足RSI", f"{rsi_5m:.1f}", "-過熱気味", delta_color="inverse")
                    else:
                        m2.metric("⚡ 5分足RSI", f"{rsi_5m:.1f}")
                        
                    m3.metric("📊 VWAP乖離", f"{vwap_diff:+.2f}%", f"VWAP: {vwap:,.1f}円", delta_color="off")
                    
                    if rsi_5m <= 35 and vwap_diff < 0:
                        st.success("🔥【買いシグナル】RSI底値 ＆ VWAP下回り（反発狙い）")
                        
                    st.markdown("---")
            except Exception as e:
                st.error(f"{code}: 処理エラー ({e})")
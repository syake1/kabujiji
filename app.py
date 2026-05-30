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

        vol_calm = vol_ratio < vol_mult
        
        # BB位置計算（バックテスト内）
        bb_range_val = bb_up - bb_lo
        bb_pos_val = ((price - bb_lo) / bb_range_val * 100) if bb_range_val > 0 else 50.0

        if not in_trade:
            # BB下限付近を強制
            if (price > ma200
                    and ma25_slope_up
                    and bb_pos_val <= bb_pos_max   # <--- ここでBB下限を厳格にチェック
                    and rsi_min <= rsi <= rsi_max
                    and bearish_count >= 1         # 過去データでは1日以上の陰線で許容
                    and vol_calm):
                entry_price = hist.iloc[i + 1]['Open']
                in_trade    = True
        else:
            exit_price = None
            if r['Low'] <= entry_price * (1 - stop_pct / 100):
                exit_price = entry_price * (1 - stop_pct / 100)
            elif r['High'] >= max(entry_price * (1 + target_pct / 100), float(bb_mid)):
                exit_price = max(entry_price * (1 + target_pct / 100), float(bb_mid))
            
            if exit_price:
                trades.append((exit_price - entry_price) / entry_price * 100)
                in_trade = False

    if not trades:
        return None

    arr      = np.array(trades)
    wins     = arr[arr > 0]
    win_rate = len(wins) / len(arr) * 100
    avg_pnl  = arr.mean()
    cum      = np.cumsum(arr)
    max_dd   = (cum - np.maximum.accumulate(cum)).min()
    return {
        "取引回数": len(arr),
        "勝率":     f"{win_rate:.1f}%",
        "平均損益": f"{avg_pnl:.2f}%",
        "最大DD":   f"{max_dd:.2f}%",
    }

# ================================================================
# メイン解析（押し目反発スコアリング）
# ================================================================
def analyze_stock(ticker_code, company_name,
                  stop_pct, target_pct, rsi_min, rsi_max, ma200_range, vol_mult, bb_pos_max):
    import time
    last_err = ""
    try:
        hist = pd.DataFrame()
        for attempt in range(3):
            try:
                tk   = yf.Ticker(f"{ticker_code}.T")
                hist = tk.history(period="2y", timeout=10)
                if len(hist) > 0:
                    break
            except Exception as e:
                last_err = str(e)
                time.sleep(1)
        if len(hist) < 200:
            return {"_error": f"{ticker_code}: データ不足"}
        if len(hist) < 210:
            return None

        hist['MA200']    = hist['Close'].rolling(200).mean()
        hist['MA25']     = hist['Close'].rolling(25).mean()
        macd, sig        = calculate_macd(hist)
        hist['MACD']     = macd
        hist['Signal']   = sig
        bb_up, bb_mid, bb_lo = calculate_bb(hist)
        hist['BB_upper'] = bb_up
        hist['BB_mid']   = bb_mid
        hist['BB_lower'] = bb_lo
        hist['RSI']      = calculate_rsi(hist)
        hist['VolMA5']   = hist['Volume'].rolling(5).mean()
        hist['VolRatio'] = hist['Volume'] / hist['VolMA5']

        latest = hist.iloc[-1]
        prev   = hist.iloc[-2]

        current_price = float(latest['Close'])
        ma200         = float(latest['MA200'])
        ma25          = float(latest['MA25'])
        rsi           = float(latest['RSI'])
        macd_val      = float(latest['MACD'])
        sig_val       = float(latest['Signal'])
        vol_ratio     = float(latest['VolRatio'])
        bb_upper_val  = float(latest['BB_upper'])
        bb_mid_val    = float(latest['BB_mid'])
        bb_lo_val     = float(latest['BB_lower'])

        if pd.isna(ma200) or pd.isna(ma25):
            return None

        diff_pct_200 = (current_price - ma200) / ma200 * 100
        diff_pct_25  = (current_price - ma25)  / ma25  * 100

        # BB位置（0%=下限, 50%=中央, 100%=上限）
        bb_range = bb_upper_val - bb_lo_val
        bb_pos   = ((current_price - bb_lo_val) / bb_range * 100) if bb_range > 0 else 50.0

        ma25_slope = check_ma25_slope(hist)
        is_bearish_cont, bearish_count = check_consecutive_bearish(hist, n=2)
        reversal_signs = check_reversal_sign(hist)
        vol_calm = vol_ratio < vol_mult

        rsi_series = hist['RSI'].dropna()
        rsi_improving = False
        if len(rsi_series) >= 4:
            rsi_3ago = float(rsi_series.iloc[-4])
            rsi_2ago = float(rsi_series.iloc[-3])
            rsi_prev = float(rsi_series.iloc[-2])
            rsi_now  = float(rsi_series.iloc[-1])
            if (rsi_2ago <= rsi_3ago and rsi_prev <= rsi_2ago and rsi_now > rsi_prev) or (rsi_prev <= rsi_2ago and rsi_now > rsi_prev):
                rsi_improving = True

        macd_diff       = macd_val - sig_val
        macd_diff_prev  = float(hist.iloc[-2]['MACD']) - float(hist.iloc[-2]['Signal'])
        macd_narrowing  = abs(macd_diff) < abs(macd_diff_prev)
        macd_gc_recent  = False
        for _i in range(1, 4):
            if len(hist) > _i:
                _p = hist.iloc[-(_i+1)]
                _c = hist.iloc[-_i]
                if float(_p['MACD']) < float(_p['Signal']) and float(_c['MACD']) >= float(_c['Signal']):
                    macd_gc_recent = True
                    break

        # ================================================================
        # 絶対条件フィルター（ここですり抜けを防止）
        # ================================================================
        must_ok = (
            current_price > ma200 and       # 200日線上
            ma25_slope and                  # 25日線上向き
            bb_pos <= bb_pos_max and        # 指定したBB位置以下（絶対条件）
            rsi_min <= rsi <= rsi_max and   # RSI条件クリア
            vol_calm                        # 出来高が爆発していない
        )

        score    = 0
        reasons  = []
        warnings = []

        if not must_ok:
            if bb_pos > bb_pos_max:
                warnings.append(f"BB位置高すぎ({bb_pos:.0f}%)⛔")
            if current_price <= ma200:
                warnings.append("200日線下⛔")
            if not ma25_slope:
                warnings.append("25日線下向き⛔")
            if not (rsi_min <= rsi <= rsi_max):
                warnings.append(f"RSI範囲外({rsi:.0f})⛔")
            if not vol_calm:
                warnings.append("出来高急増⛔")
            
            status = "⛔ 除外（条件未達）"
        else:
            # 条件をクリアした銘柄のみスコアリング
            score += 4 # 基礎点
            reasons.append(f"BB下限到達({bb_pos:.0f}%)")
            
            if is_bearish_cont:
                score += 2
                reasons.append(f"陰線{bearish_count}日続き")
            
            if rsi_improving:
                score += 2
                reasons.append("RSI底打ち反転")
                
            if reversal_signs:
                score += 2
                reasons.append(" ".join(reversal_signs))
                
            if macd_gc_recent or macd_narrowing:
                score += 1
                reasons.append("MACD好転")
                
            if abs(diff_pct_25) <= 5.0:
                score += 1

            score = min(score, 10)

            if score >= 8:
                status = "🔥 買い候補"
            else:
                status = "👀 監視（押し目形成中）"

        stop_price   = round(current_price * (1 - stop_pct  / 100), 1)
        target_price = round(max(current_price * (1 + target_pct / 100), bb_mid_val), 1)
        rr_ratio     = round((target_price - current_price) / (current_price - stop_price), 1) if current_price > stop_price else 0.0

        if macd_gc_recent:
            macd_label = "🟢 GC直近"
        elif macd_narrowing and macd_val < sig_val:
            macd_label = "🟡 収束中"
        elif macd_val > sig_val:
            macd_label = "↑上"
        else:
            macd_label = "↓下"

        bt = backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, vol_mult, bb_pos_max)

        return {
            "コード":         ticker_code,
            "会社名":         company_name,
            "判定":           status,
            "スコア":         score,
            "現在値":         round(current_price, 1),
            "200日乖離":      f"{diff_pct_200:+.2f}%",
            "25日乖離":       f"{diff_pct_25:+.2f}%",
            "RSI(14)":        round(rsi, 1),
            "MACD":           macd_label,
            "BB位置":         f"{bb_pos:.0f}%",
            "出来高倍率":     f"{vol_ratio:.1f}x",
            "陰線日数":       bearish_count,
            "反発サイン":     " / ".join(reversal_signs) if reversal_signs else "-",
            "損切り価格":     stop_price,
            "利確目標":       target_price,
            "RRレシオ":       f"1:{rr_ratio}",
            "根拠":           " / ".join(reasons) if reasons else "-",
            "注意点":         " / ".join(warnings) if warnings else "-",
            "チャート":       f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "BT勝率":         bt["勝率"]     if bt else "-",
            "BT平均損益":     bt["平均損益"] if bt else "-",
            "BT取引数":       bt["取引回数"] if bt else 0,
            "BT最大DD":       bt["最大DD"]   if bt else "-",
        }
    except Exception as e:
        return {"_error": f"{ticker_code}: 予期せぬエラー {str(e)}"}

# ================================================================
# メイン画面
# ================================================================
st.title("🚀 アンチグラビティ・コア Pro+ v2")
st.caption("超・押し目反発特化版 ｜ BB下限到達銘柄のみを厳密に抽出します")

col_a, col_b = st.columns(2)

# AI ニュース分析
with col_a:
    st.subheader("📰 AI 投資判断")
    news_input = st.text_area("ニュースをペースト", height=150)
    if st.button("AI分析を実行"):
        if not gemini_key:
            st.warning("⚠️ サイドバーに Gemini API Key を入力してください")
        elif not news_input:
            st.warning("⚠️ ニュースを貼り付けてください")
        else:
            try:
                genai.configure(api_key=gemini_key)
                target_model_name = ""
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        target_model_name = m.name
                        if 'flash' in m.name or 'pro' in m.name:
                            break
                if target_model_name:
                    model = genai.GenerativeModel(target_model_name)
                    name  = target_model_name.replace('models/', '')
                    with st.spinner(f"AI ({name}) が分析中..."):
                        prompt = (
                            "あなたは日本株スイングトレードの専門家です。\n"
                            "戦略は「強い銘柄の押し目反発（平均回帰）」です。\n"
                            "以下のニュースを読み、\n"
                            "①相場全体への影響\n"
                            "②一時的売りで押し目が生じやすいセクター\n"
                            "③スイング押し目買いの観点で注目すべきポイント\n"
                            "を簡潔に解説してください。\n\nニュース:\n" + news_input
                        )
                        res = model.generate_content(prompt)
                        st.success("分析完了！")
                        st.info(res.text)
            except Exception as e:
                st.error(f"AI解析エラー: {e}")

# 銘柄スキャン
with col_b:
    st.subheader("📊 銘柄一括スキャン")
    files = st.file_uploader("SBIのCSVをアップロード（複数可）", type=['csv'], accept_multiple_files=True)
    if files:
        dfs = []
        for file in files:
            file.seek(0)
            try:
                dfs.append(pd.read_csv(file, encoding='shift

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import google.generativeai as genai
from datetime import datetime

# --- 初期設定 ---
st.set_page_config(page_title="アンチグラビティ・コア Pro+", layout="wide")

import json, os

SAVE_PATH = 'data/scan_result_full.json'

def save_results(df):
    os.makedirs('data', exist_ok=True)
    data = {
        'saved_at': datetime.now().strftime('%Y/%m/%d %H:%M'),
        'records': df.to_dict(orient='records')
    }
    with open(SAVE_PATH, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_results():
    if not os.path.exists(SAVE_PATH):
        return None, None
    try:
        with open(SAVE_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
        df = pd.DataFrame(data['records'])
        return df, data.get('saved_at', '')
    except:
        return None, None

if 'analysis_results' not in st.session_state:
    df_loaded, saved_at = load_results()
    if df_loaded is not None and not df_loaded.empty:
        st.session_state.analysis_results = df_loaded
        st.session_state.saved_at = saved_at
    else:
        st.session_state.analysis_results = None
        st.session_state.saved_at = ''

if 'short_results' not in st.session_state:
    st.session_state.short_results = None

# ================================================================
# タイトル
# ================================================================
st.title("🚀 アンチグラビティ・コア Pro+")
st.caption("押し目反発 & 空売り 両対応版")

# ================================================================
# システム設定
# ================================================================
with st.expander("⚙️ システム設定 / スイング条件設定", expanded=False):
    row0 = st.columns([2, 2, 1])
    with row0[0]:
        gemini_key = st.text_input("Gemini API Key", type="password")
    with row0[1]:
        discord_webhook = st.text_input("Discord Webhook URL", type="password")
    with row0[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.saved_at = ''
            st.session_state.short_results = None
            if os.path.exists(SAVE_PATH):
                os.remove(SAVE_PATH)
            st.rerun()

    st.markdown("---")
    st.markdown("**🎯 買いスキャン条件**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1:
        rsi_max = st.slider("RSI上限", 50, 70, 55)
    with c2:
        rsi_min = st.slider("RSI下限", 25, 45, 35)
    with c3:
        ma200_range = st.slider("200日線乖離上限(%)", 1, 30, 20)
    with c4:
        vol_mult = st.slider("出来高急増除外倍率", 1.5, 5.0, 2.5, step=0.1)
    with c5:
        stop_pct = st.slider("損切りライン(%)", 1, 10, 4)
    with c6:
        target_pct = st.slider("利確ライン(%)", 2, 20, 8)

    st.markdown("---")
    col_s, col_c = st.columns(2)
    with col_s:
        st.info(f"""
**【押し目条件】**
1. 株価 > 200日線 ／ 25日線上向き
2. BB下限タッチ後 反発サインあり
3. RSI {rsi_min}〜{rsi_max}
4. 出来高落ち着き
5. 下ヒゲ陽線🔥 / 包み足⚡ / 陽線転換↑

🎯 利確: BBセンター(+{target_pct}%)　💀 損切: -{stop_pct}%
        """)
    with col_c:
        st.warning("""
**【⏰ 10時半〜11時 反発チェック】**

✅ エントリー前に確認：
- 9〜10時に出来高が急減
- 前場安値を2〜3回試して割らない
- 日経先物が底打ち・戻し始め
- 小陽線 or 下ヒゲが出た

❌ 見送り条件：
- 個別の悪材料がある
- 出来高が増え続けている
- 節目サポートを割り込んだまま

💡 **損切りは前場安値割れ**
        """)

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
    body       = abs(latest['Close'] - latest['Open'])
    lower_wick = min(latest['Close'], latest['Open']) - latest['Low']
    upper_wick = latest['High'] - max(latest['Close'], latest['Open'])
    if latest['Close'] > latest['Open'] and body > 0 and lower_wick >= body * 1.5:
        signs.append("下ヒゲ陽線")
    if (prev['Close'] < prev['Open']
            and latest['Close'] > latest['Open']
            and latest['Close'] > prev['Open']
            and latest['Open'] < prev['Close']):
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
# バックテスト（押し目反発版）
# ================================================================
def backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, vol_mult):
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
        bb_mid = r['BB_mid']
        vol_ratio = r['VolRatio']
        if pd.isna(ma200) or pd.isna(rsi) or pd.isna(ma25):
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
        if not in_trade:
            if (price > ma200 and ma25_slope_up and price <= bb_mid
                    and rsi_min <= rsi <= rsi_max
                    and bearish_count >= 2 and vol_calm
                    and price <= ma25 * 1.03):
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
# 買いスキャン解析
# ================================================================
def analyze_stock(ticker_code, company_name,
                  stop_pct, target_pct, rsi_min, rsi_max, ma200_range, vol_mult):
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
            return {"_error": f"{ticker_code}: データ取得失敗 データ数={len(hist)} {last_err[:50]}"}
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

        bb_range = bb_upper_val - bb_lo_val
        bb_pos   = ((current_price - bb_lo_val) / bb_range * 100) if bb_range > 0 else 50.0

        bb_touched_lower  = False
        bb_touch_days_ago = 0
        for _bi in range(1, 6):
            if len(hist) > _bi:
                _row   = hist.iloc[-_bi]
                _bb_lo = float(_row['BB_lower'])
                _low   = float(_row['Low'])
                _close = float(_row['Close'])
                if _low <= _bb_lo * 1.02 or _close <= _bb_lo * 1.02:
                    bb_touched_lower  = True
                    bb_touch_days_ago = _bi
                    break

        bb_rebounding = bb_touched_lower and bb_pos >= 20
        ma25_slope    = check_ma25_slope(hist)
        is_bearish_cont, bearish_count = check_consecutive_bearish(hist, n=2)
        reversal_signs = check_reversal_sign(hist)
        vol_calm = vol_ratio < vol_mult

        rsi_series    = hist['RSI'].dropna()
        rsi_improving = False
        if len(rsi_series) >= 4:
            rsi_3ago = float(rsi_series.iloc[-4])
            rsi_2ago = float(rsi_series.iloc[-3])
            rsi_prev = float(rsi_series.iloc[-2])
            rsi_now  = float(rsi_series.iloc[-1])
            if rsi_2ago <= rsi_3ago and rsi_prev <= rsi_2ago and rsi_now > rsi_prev:
                rsi_improving = True
            elif rsi_prev <= rsi_2ago and rsi_now > rsi_prev:
                rsi_improving = True

        macd_diff      = macd_val - sig_val
        macd_diff_prev = float(hist.iloc[-2]['MACD']) - float(hist.iloc[-2]['Signal'])
        macd_narrowing = abs(macd_diff) < abs(macd_diff_prev)
        macd_gc_recent = False
        for _i in range(1, 4):
            if len(hist) > _i:
                _p = hist.iloc[-(_i+1)]
                _c = hist.iloc[-_i]
                if float(_p['MACD']) < float(_p['Signal']) and float(_c['MACD']) >= float(_c['Signal']):
                    macd_gc_recent = True
                    break

        score   = 0
        reasons = []
        warnings = []

        if current_price > ma200:
            score += 2
            reasons.append(f"200日線上(+{diff_pct_200:.1f}%)")
        else:
            warnings.append("200日線下⚠️")

        if ma25_slope:
            score += 2
            reasons.append("25日線上向き")
        else:
            warnings.append("25日線下向き")

        if bb_touched_lower and bb_pos <= 80:
            if bb_touch_days_ago <= 2:
                score += 3
                reasons.append(f"BB下限タッチ翌{bb_touch_days_ago}日目({bb_pos:.0f}%)")
            elif bb_touch_days_ago == 3:
                score += 2
                reasons.append(f"BB下限タッチ{bb_touch_days_ago}日後({bb_pos:.0f}%)")
            elif bb_touch_days_ago <= 5:
                score += 1
                reasons.append(f"BB下限タッチ{bb_touch_days_ago}日後({bb_pos:.0f}%) 乗り遅れ気味")
        elif bb_pos <= 25:
            score += 2
            reasons.append(f"BB下限付近({bb_pos:.0f}%)")
        elif bb_pos <= 50:
            score += 1
            reasons.append(f"BBセンター以下({bb_pos:.0f}%)")
        else:
            warnings.append(f"BB上部({bb_pos:.0f}%)")

        if rsi_min <= rsi <= rsi_max:
            score += 1
            reasons.append(f"RSI適正({rsi:.0f})")
            if rsi_improving:
                score += 1
                reasons.append("RSI底打ち反転")
        elif rsi < rsi_min:
            warnings.append(f"RSI過売({rsi:.0f})")
        else:
            warnings.append(f"RSI過熱({rsi:.0f})")

        if is_bearish_cont:
            score += 1
            reasons.append(f"陰線{bearish_count}日続き")

        if vol_calm:
            score += 1
            reasons.append(f"出来高落ち着き({vol_ratio:.1f}x)")
        else:
            warnings.append(f"出来高急増({vol_ratio:.1f}x)⚠️")

        if abs(diff_pct_25) <= 5.0:
            score += 1
            reasons.append(f"25日線付近({diff_pct_25:+.1f}%)")

        reversal_score = 0
        if "下ヒゲ陽線" in reversal_signs:
            reversal_score = 3
        elif "包み足" in reversal_signs:
            reversal_score = 2
        elif "陽線転換" in reversal_signs:
            reversal_score = 1
        elif "長い下ヒゲ" in reversal_signs:
            reversal_score = 1
        if reversal_score > 0:
            score += reversal_score
            sign_label = reversal_signs[0] if reversal_signs else ""
            if reversal_score == 3:
                reasons.append(f"🔥 {sign_label}（最強反発）")
            elif reversal_score == 2:
                reasons.append(f"⚡ {sign_label}（強い反発）")
            else:
                reasons.append(f"↑ {sign_label}（反発予兆）")

        if macd_gc_recent:
            score = min(score + 1, 13)
            reasons.append("MACD-GC直近")
        elif macd_narrowing and macd_val < sig_val:
            score = min(score + 1, 13)
            reasons.append("MACD収束中")

        if bb_pos > 80:
            score = max(score - 3, 0)
            warnings.append(f"BB上部({bb_pos:.0f}%) 押し目未形成⛔")
        if bb_touched_lower and bb_pos > 80:
            score = max(score - 2, 0)
            warnings.append(f"BB下限タッチ後に急騰⛔ 乗り遅れ(BB{bb_pos:.0f}%)")
        if rsi > rsi_max:
            warnings.append(f"RSI過熱({rsi:.0f}) 押し目未形成⛔")

        must_ok     = (current_price > ma200) and ma25_slope
        has_reversal = len(reversal_signs) > 0
        bb_must_ok   = (bb_touched_lower and bb_pos <= 80 and rsi <= rsi_max
                        and bb_touch_days_ago <= 3 and has_reversal)

        if not must_ok:
            status = "⛔ 除外（弱い銘柄）"
        elif bb_pos > 80:
            status = "⛔ 除外（BB上部/過熱）"
        elif not bb_touched_lower:
            status = "👀 監視（BB下限未タッチ）"
        elif bb_touched_lower and not has_reversal:
            status = "⏳ 様子見（反発サイン待ち）"
        elif bb_touch_days_ago > 3 and bb_pos > 50:
            status = "⏳ 様子見（反発乗り遅れ）"
        elif rsi > rsi_max:
            status = "⏳ 様子見（RSI過熱冷め待ち）"
        elif score >= 8:
            status = "🔥 買い候補"
        elif score >= 6:
            status = "👀 監視（押し目形成中）"
        elif score >= 4:
            status = "⏳ 様子見"
        else:
            status = "➖ 対象外"

        stop_price   = round(current_price * (1 - stop_pct  / 100), 1)
        target_price = round(max(current_price * (1 + target_pct / 100), bb_mid_val), 1)
        rr_ratio     = round((target_price - current_price) / (current_price - stop_price), 1) \
                       if current_price > stop_price else 0.0

        if macd_gc_recent:
            macd_label = "🟢 GC直近"
        elif macd_narrowing and macd_val < sig_val:
            macd_label = "🟡 収束中"
        elif macd_val > sig_val:
            macd_label = "↑上"
        else:
            macd_label = "↓下"

        bt = backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, vol_mult)

        try:
            info      = tk.info
            div_yield = info.get('dividendYield') or 0
            per       = info.get('forwardPE', '-')
        except:
            div_yield = 0
            per       = '-'

        earnings_date = "未発表"
        try:
            cal = tk.calendar
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                d = cal.loc['Earnings Date'].iloc[0] if 'Earnings Date' in cal.index else cal.iloc[0, 0]
                earnings_date = d.strftime('%Y/%m/%d')
        except:
            pass

        return {
            "コード":        ticker_code,
            "会社名":        company_name,
            "判定":          status,
            "スコア":        score,
            "現在値":        round(current_price, 1),
            "200日乖離":     f"{diff_pct_200:+.2f}%",
            "25日乖離":      f"{diff_pct_25:+.2f}%",
            "RSI(14)":       round(rsi, 1),
            "MACD":          macd_label,
            "BB位置":        f"{bb_pos:.0f}%" + (f"(↑{bb_touch_days_ago}日前下限タッチ)" if bb_touched_lower else ""),
            "出来高倍率":    f"{vol_ratio:.1f}x",
            "陰線日数":      bearish_count,
            "反発サイン":    " / ".join(reversal_signs) if reversal_signs else "-",
            "損切り価格":    stop_price,
            "利確目標":      target_price,
            "RRレシオ":      f"1:{rr_ratio}",
            "配当利回り":    f"{div_yield * 100:.2f}%",
            "次期決算":      earnings_date,
            "PER":           per,
            "根拠":          " / ".join(reasons) if reasons else "-",
            "注意点":        " / ".join(warnings) if warnings else "-",
            "チャート":      f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "BT勝率":        bt["勝率"]     if bt else "-",
            "BT平均損益":    bt["平均損益"] if bt else "-",
            "BT取引数":      bt["取引回数"] if bt else 0,
            "BT最大DD":      bt["最大DD"]   if bt else "-",
        }
    except:
        return None

# ================================================================
# 空売り判定関数
# ================================================================
def analyze_short(ticker_code, company_name,
                  credit_ratio, credit_sell_change,
                  credit_sell_buy_ratio,
                  stop_pct=4, target_pct=8,
                  rsi_short_min=40, rsi_short_max=60):
    """
    空売りロジック（下落トレンド継続銘柄の戻り売り）
    返り値の判定ステータス:
      "🔻 空売り候補"
      "👀 監視（下落トレンド継続）"
      "⏳ 様子見"
      "➖ 対象外"
      "⛔ 除外（200日線上・買いスキャン対象）"
    """
    import time
    try:
        hist = pd.DataFrame()
        for attempt in range(3):
            try:
                tk   = yf.Ticker(f"{ticker_code}.T")
                hist = tk.history(period="2y", timeout=10)
                if len(hist) > 0:
                    break
            except:
                time.sleep(1)
        if len(hist) < 200:
            return None

        hist['MA200']    = hist['Close'].rolling(200).mean()
        hist['MA25']     = hist['Close'].rolling(25).mean()
        hist['MA75']     = hist['Close'].rolling(75).mean()
        bb_up, bb_mid, bb_lo = calculate_bb(hist)
        hist['BB_upper'] = bb_up
        hist['BB_mid']   = bb_mid
        hist['BB_lower'] = bb_lo
        hist['RSI']      = calculate_rsi(hist)
        macd, sig        = calculate_macd(hist)
        hist['MACD']     = macd
        hist['Signal']   = sig
        hist['VolMA5']   = hist['Volume'].rolling(5).mean()
        hist['VolRatio'] = hist['Volume'] / hist['VolMA5']

        latest = hist.iloc[-1]
        prev   = hist.iloc[-2]

        current_price = float(latest['Close'])
        ma200         = float(latest['MA200'])
        ma25          = float(latest['MA25'])
        ma75          = float(latest['MA75']) if not pd.isna(latest['MA75']) else ma200
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

        bb_range = bb_upper_val - bb_lo_val
        bb_pos   = ((current_price - bb_lo_val) / bb_range * 100) if bb_range > 0 else 50.0

        # 200日線・25日線の傾き（下向き判定）
        ma200_5ago       = float(hist['MA200'].dropna().iloc[-6]) if len(hist['MA200'].dropna()) >= 6 else ma200
        ma200_slope_down = ma200 < ma200_5ago
        ma25_5ago        = float(hist['MA25'].dropna().iloc[-6]) if len(hist['MA25'].dropna()) >= 6 else ma25
        ma25_slope_down  = ma25 < ma25_5ago

        # 直近の反発（戻り）を検出
        recent_low  = min([float(hist.iloc[-_i]['Low']) for _i in range(1, 6) if len(hist) > _i])
        bounce_pct  = (current_price - recent_low) / recent_low * 100 if recent_low > 0 else 0

        # 陽線続き（戻り局面の確認）
        bullish_count = 0
        for _j in range(1, 6):
            if len(hist) > _j:
                _r = hist.iloc[-_j]
                if _r['Close'] >= _r['Open']:
                    bullish_count += 1
                else:
                    break

        # 失速サイン（戻り売りエントリーの根拠）
        top_signs  = []
        body       = abs(latest['Close'] - latest['Open'])
        upper_wick = latest['High'] - max(latest['Close'], latest['Open'])
        lower_wick = min(latest['Close'], latest['Open']) - latest['Low']

        # 上ヒゲ陰線（戻り失速）
        if latest['Close'] < latest['Open'] and body > 0 and upper_wick >= body * 1.5:
            top_signs.append("上ヒゲ陰線")
        # 被せ線
        if (prev['Close'] >= prev['Open']
                and latest['Open'] > prev['Close']
                and latest['Close'] < prev['Open']):
            top_signs.append("被せ線")
        # 陰線転換
        elif prev['Close'] >= prev['Open'] and latest['Close'] < latest['Open']:
            if "上ヒゲ陰線" not in top_signs:
                top_signs.append("陰線転換")
        # 長い上ヒゲ（陽線でも上ヒゲが長い）
        if body > 0 and upper_wick >= body * 2.0 and "上ヒゲ陰線" not in top_signs:
            top_signs.append("長い上ヒゲ")

        # MACD デッドクロス・下方向
        macd_diff      = macd_val - sig_val
        macd_diff_prev = float(hist.iloc[-2]['MACD']) - float(hist.iloc[-2]['Signal'])
        macd_dc_recent = False
        for _i in range(1, 4):
            if len(hist) > _i:
                _p = hist.iloc[-(_i+1)]
                _c = hist.iloc[-_i]
                if float(_p['MACD']) > float(_p['Signal']) and float(_c['MACD']) <= float(_c['Signal']):
                    macd_dc_recent = True
                    break
        macd_below_sig = macd_val < sig_val

        # ================================================================
        # 空売りスコアリング（最大15点）
        # ================================================================
        score    = 0
        reasons  = []
        warnings = []

        def safe_float(val):
            try:
                return float(str(val).replace(',', ''))
            except:
                return np.nan

        # 1. 200日線乖離
        if diff_pct_200 <= -20:
            score += 4
            reasons.append(f"200日線大幅下乖離({diff_pct_200:.1f}%)")
        elif diff_pct_200 <= -15:
            score += 3
            reasons.append(f"200日線下乖離({diff_pct_200:.1f}%)")
        elif diff_pct_200 <= -10:
            score += 2
            reasons.append(f"200日線下({diff_pct_200:.1f}%)")
        elif diff_pct_200 <= -5:
            score += 1
            reasons.append(f"200日線やや下({diff_pct_200:.1f}%)")
        elif diff_pct_200 > 0:
            warnings.append(f"200日線上⚠️ (+{diff_pct_200:.1f}%)")

        # 2. 200日線が下向き
        if ma200_slope_down:
            score += 1
            reasons.append("200日線下向き継続")

        # 3. 25日線下向き
        if ma25_slope_down:
            score += 1
            reasons.append("25日線下向き")

        # 4. MA配列（完全下落配列）
        if ma25 < ma75 < ma200:
            score += 2
            reasons.append("完全下落配列(25<75<200)")
        elif ma25 < ma200:
            score += 1
            reasons.append("25日線<200日線")

        # 5. RSI（戻り一服ゾーン）
        if rsi_short_min <= rsi <= rsi_short_max:
            score += 2
            reasons.append(f"RSI戻り一服({rsi:.0f})")
        elif rsi > rsi_short_max:
            score += 1
            reasons.append(f"RSI高め({rsi:.0f}) 戻り過ぎ注意")
            warnings.append(f"RSI過熱({rsi:.0f}) 反発リスク")
        elif rsi < 30:
            warnings.append(f"RSI売られすぎ({rsi:.0f}) 一時反発注意")
        else:
            score += 1
            reasons.append(f"RSI({rsi:.0f})")

        # 6. 信用倍率
        cr = safe_float(credit_ratio)
        if not np.isnan(cr):
            if cr >= 10:
                score += 3
                reasons.append(f"信用倍率{cr:.1f}(買い残過多・空売り超有利)")
            elif cr >= 5:
                score += 2
                reasons.append(f"信用倍率{cr:.1f}(買い残多め・空売り有利)")
            elif cr >= 2:
                score += 1
                reasons.append(f"信用倍率{cr:.1f}")
            elif cr <= 1.0:
                warnings.append(f"信用倍率{cr:.2f}(売り残多め・踏み上げ注意⚠️)")

        # 7. 売り残前週比プラス
        sc_val = safe_float(credit_sell_change)
        if not np.isnan(sc_val):
            if sc_val > 0:
                score += 1
                reasons.append(f"売り残増加(+{sc_val:,.0f}株)")
            elif sc_val < -10000:
                warnings.append(f"売り残大幅減少({sc_val:,.0f}株)⚠️")

        # 8. 失速サイン
        top_score = 0
        if "被せ線" in top_signs:
            top_score = 3
        elif "上ヒゲ陰線" in top_signs:
            top_score = 2
        elif "陰線転換" in top_signs or "長い上ヒゲ" in top_signs:
            top_score = 1
        if top_score > 0:
            score += top_score
            label = top_signs[0]
            if top_score == 3:
                reasons.append(f"🔻{label}（最強失速）")
            elif top_score == 2:
                reasons.append(f"⬇️{label}（失速サイン）")
            else:
                reasons.append(f"↓{label}（失速予兆）")

        # 9. MACD
        if macd_dc_recent:
            score = min(score + 1, 15)
            reasons.append("MACD-DC直近")
        elif macd_below_sig:
            score = min(score + 1, 15)
            reasons.append("MACD下方向")

        # --- 必須条件チェック ---
        trend_down = diff_pct_200 < 0  # 200日線を下回っていること

        if not trend_down:
            status = "⛔ 除外（200日線上・買いスキャン対象）"
        elif score >= 9:
            status = "🔻 空売り候補"
        elif score >= 7:
            status = "👀 監視（下落トレンド継続）"
        elif score >= 5:
            status = "⏳ 様子見"
        else:
            status = "➖ 対象外"

        stop_price   = round(current_price * (1 + stop_pct  / 100), 1)
        target_price = round(current_price * (1 - target_pct / 100), 1)
        rr_ratio     = round((current_price - target_price) / (stop_price - current_price), 1) \
                       if stop_price > current_price else 0.0

        if macd_dc_recent:
            macd_label = "🔴 DC直近"
        elif macd_below_sig:
            macd_label = "↓下"
        else:
            macd_label = "↑上"

        return {
            "コード":        ticker_code,
            "会社名":        company_name,
            "判定":          status,
            "スコア":        score,
            "現在値":        round(current_price, 1),
            "200日乖離":     f"{diff_pct_200:+.2f}%",
            "25日乖離":      f"{diff_pct_25:+.2f}%",
            "MA配列":        f"25:{ma25:.0f} / 75:{ma75:.0f} / 200:{ma200:.0f}",
            "RSI(14)":       round(rsi, 1),
            "MACD":          macd_label,
            "BB位置":        f"{bb_pos:.0f}%",
            "出来高倍率":    f"{vol_ratio:.1f}x",
            "陽線日数":      bullish_count,
            "失速サイン":    " / ".join(top_signs) if top_signs else "-",  # ← 正しいキー名
            "信用倍率":      f"{cr:.2f}" if not np.isnan(cr) else "-",
            "売り残前週比":  f"{sc_val:+,.0f}株" if not np.isnan(sc_val) else "-",
            "損切り価格":    stop_price,
            "利確目標":      target_price,
            "RRレシオ":      f"1:{rr_ratio}",
            "チャート":      f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "根拠":          " / ".join(reasons) if reasons else "-",
            "注意点":        " / ".join(warnings) if warnings else "-",
        }
    except:
        return None

# ================================================================
# タブ構成: 買いスキャン / 空売りスキャン / AIニュース分析
# ================================================================
st.markdown("---")
tab_buy, tab_short, tab_ai = st.tabs(["📈 買いスキャン", "🔻 空売りスキャン", "📰 AIニュース分析"])

# ================================================================
# タブ1: 買いスキャン
# ================================================================
with tab_buy:
    st.subheader("📊 銘柄一括スキャン（押し目買い）")
    files = st.file_uploader("SBIのCSVをアップロード（複数可）", type=['csv'],
                             accept_multiple_files=True, key="buy_csv")
    if files:
        dfs = []
        for file in files:
            file.seek(0)
            try:
                dfs.append(pd.read_csv(file, encoding='shift-jis'))
            except:
                file.seek(0)
                dfs.append(pd.read_csv(file, encoding='utf-8'))
        df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        st.caption(f"📂 {len(files)}ファイル合計 {len(df)}銘柄を読み込みました")

        c_col = [c for c in df.columns if 'コード' in c]
        if not c_col:
            c_col = [c for c in df.columns if c.strip() == '銘柄']
        n_col = [c for c in df.columns if '銘柄名' in c]
        if not n_col:
            n_col = [c for c in df.columns if '会社名' in c]
        if not n_col:
            n_col = [c for c in df.columns if c.strip() == '銘柄.1']

        if not c_col or not n_col:
            st.error(f"⚠️ CSVに「コード」「銘柄名」列が見つかりません。検出された列: {list(df.columns)}")
        else:
            t_list = df[[c_col[0], n_col[0]]].dropna()
            if st.button(f"🚀 {len(t_list)}銘柄を一括解析（バックテスト込み）"):
                results    = []
                errors     = []
                bar        = st.progress(0)
                status_txt = st.empty()
                err_txt    = st.empty()
                for i, (idx, row) in enumerate(t_list.iterrows()):
                    code = str(row[c_col[0]])
                    name = str(row[n_col[0]])
                    status_txt.text(f"解析中... {code} {name} ({i+1}/{len(t_list)}) ✅{len(results)}件 ❌{len(errors)}件")
                    res = analyze_stock(code, name, stop_pct, target_pct,
                                        rsi_min, rsi_max, ma200_range, vol_mult)
                    if res and '_error' in res:
                        errors.append(res['_error'])
                        if len(errors) <= 3:
                            err_txt.warning(f"⚠️ 直近エラー: {res['_error']}")
                    elif res:
                        results.append(res)
                    bar.progress((i + 1) / len(t_list))
                status_txt.text(f"✅ 完了！ 成功:{len(results)}件 / 失敗:{len(errors)}件")

                if errors:
                    with st.expander(f"❌ 取得失敗 {len(errors)}件"):
                        for e in errors[:20]:
                            st.text(e)

                if results:
                    good    = [r for r in results if '判定' in r]
                    df_good = pd.DataFrame(good) if good else None
                    st.session_state.analysis_results = df_good
                    if df_good is not None:
                        save_results(df_good)
                        st.session_state.saved_at = datetime.now().strftime('%Y/%m/%d %H:%M')
                        buy_list   = [r for r in good if r.get('判定') == '🔥 買い候補']
                        watch_list = [r for r in good if r.get('判定') == '👀 監視（押し目形成中）']
                        mobile_data = {
                            'updated': st.session_state.saved_at,
                            'buy':   buy_list,
                            'watch': watch_list,
                            'total': len(good),
                        }
                        os.makedirs('data', exist_ok=True)
                        with open('data/scan_result.json', 'w', encoding='utf-8') as f:
                            json.dump(mobile_data, f, ensure_ascii=False, indent=2)
                        st.success(f"✅ {len(good)}銘柄を保存しました")

    # 買いスキャン結果表示
    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        if res_df.empty or '判定' not in res_df.columns:
            st.warning("⚠️ 表示できる解析結果がありません。")
        else:
            saved_at = st.session_state.get('saved_at', '')
            if saved_at:
                st.info(f"📂 {saved_at} スキャン結果")

            st.markdown("---")
            st.header("🔥 【厳選】押し目買い候補")
            buy_only = res_df[res_df['判定'] == "🔥 買い候補"].copy()

            def parse_pct(x):
                try:
                    return float(str(x).replace('%', ''))
                except:
                    return 0.0

            if not buy_only.empty:
                buy_only['_win'] = buy_only['BT勝率'].apply(parse_pct)
                buy_only = buy_only.sort_values(['_win', 'スコア'], ascending=False).drop(columns=['_win'])

                def pct_to_float(col):
                    return col.apply(lambda x: float(str(x).replace('%','').replace('+','')) if x != '-' else 0.0)

                buy_display = buy_only.copy()
                for c in ['200日乖離', '25日乖離', 'BT勝率', 'BT平均損益', 'BT最大DD']:
                    if c in buy_display.columns:
                        buy_display[c] = pct_to_float(buy_display[c])

                if '反発サイン' in buy_display.columns and '陰線日数' in buy_display.columns:
                    buy_display['根拠'] = buy_display.apply(
                        lambda r: (
                            (f"{'🔥' if '下ヒゲ陽線' in str(r['反発サイン']) else '⚡' if '包み足' in str(r['反発サイン']) else '↑'}"
                             f"{r['反発サイン']} " if str(r['反発サイン']) not in ['-',''] else '') +
                            (f"陰{r['陰線日数']}日 " if str(r['陰線日数']) not in ['-','0',''] else '') +
                            str(r.get('根拠',''))
                        )[:40], axis=1
                    )
                if 'チャート' in buy_display.columns:
                    buy_display['📊'] = buy_display['チャート']

                display_cols = [
                    "コード", "会社名", "スコア", "現在値",
                    "200日乖離", "25日乖離", "RSI(14)", "MACD", "BB位置",
                    "出来高倍率", "損切り価格", "利確目標", "RRレシオ",
                    "根拠", "注意点", "BT勝率", "BT平均損益", "BT最大DD", "📊"
                ]
                disp = [c for c in display_cols if c in buy_display.columns]
                st.dataframe(
                    buy_display[disp],
                    use_container_width=True,
                    column_config={
                        "📊":         st.column_config.LinkColumn("📊", display_text="📊"),
                        "損切り価格": st.column_config.NumberColumn("損切💀", format="%.0f"),
                        "利確目標":   st.column_config.NumberColumn("利確🎯", format="%.0f"),
                        "スコア":     st.column_config.ProgressColumn("スコア", min_value=0, max_value=13),
                        "200日乖離":  st.column_config.NumberColumn("200日%", format="%.1f"),
                        "25日乖離":   st.column_config.NumberColumn("25日%",  format="%.1f"),
                        "BT勝率":     st.column_config.NumberColumn("BT勝率", format="%.0f"),
                        "BT平均損益": st.column_config.NumberColumn("BT損益", format="%.1f"),
                        "BT最大DD":   st.column_config.NumberColumn("最大DD", format="%.1f"),
                    },
                    height=400,
                )

                dl_cols  = [c for c in display_cols if c in buy_only.columns and c != "チャート"]
                csv_data = buy_only[dl_cols].to_csv(index=False, encoding='utf-8-sig')
                now_str  = datetime.now().strftime('%Y%m%d_%H%M')
                st.download_button(
                    label     = f"📥 買い候補CSVをダウンロード（{len(buy_only)}銘柄）",
                    data      = csv_data,
                    file_name = f"買い候補_{now_str}.csv",
                    mime      = "text/csv",
                    use_container_width=True,
                )

                st.subheader("📈 バックテスト結果")
                bt_cols = ['コード', '会社名', 'BT勝率', 'BT平均損益', 'BT取引数', 'BT最大DD', 'RRレシオ']
                st.dataframe(buy_only[[c for c in bt_cols if c in buy_only.columns]], use_container_width=True)

                # チャート表示
                st.markdown("---")
                st.subheader("📊 買い候補チャート")
                try:
                    import matplotlib
                    matplotlib.use('Agg')
                    import matplotlib.pyplot as plt
                    HAS_MPL = True
                except ImportError:
                    HAS_MPL = False

                def plot_stock_chart(code, name, stop_price, target_price):
                    try:
                        tk2  = yf.Ticker(f"{code}.T")
                        hist2 = tk2.history(period="6mo")
                        if len(hist2) < 30:
                            return None
                        hist2 = hist2.reset_index()
                        hist2['MA25']  = hist2['Close'].rolling(25).mean()
                        hist2['MA200'] = hist2['Close'].rolling(200).mean()
                        bb_up2, bb_mid2, bb_lo2 = calculate_bb(hist2)
                        hist2['BB_upper'] = bb_up2
                        hist2['BB_mid']   = bb_mid2
                        hist2['BB_lower'] = bb_lo2
                        hist2['VolMA5']   = hist2['Volume'].rolling(5).mean()
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                            gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                        fig.patch.set_facecolor('#0E1117')
                        ax1.set_facecolor('#0E1117')
                        ax2.set_facecolor('#0E1117')
                        for i, row in hist2.iterrows():
                            color = '#FF4B4B' if row['Close'] >= row['Open'] else '#1F77B4'
                            ax1.plot([i, i], [row['Low'], row['High']], color=color, linewidth=0.8)
                            ax1.bar(i, abs(row['Close'] - row['Open']),
                                    bottom=min(row['Open'], row['Close']),
                                    color=color, width=0.6, alpha=0.9)
                        ax1.plot(range(len(hist2)), hist2['MA25'],  color='orange',  linewidth=1.2, label='MA25')
                        ax1.plot(range(len(hist2)), hist2['MA200'], color='#00BFFF', linewidth=1.0, label='MA200', linestyle='--')
                        ax1.fill_between(range(len(hist2)), hist2['BB_upper'], hist2['BB_lower'], alpha=0.08, color='gray')
                        ax1.plot(range(len(hist2)), hist2['BB_mid'],   color='#AAAAAA', linewidth=0.8, linestyle=':')
                        ax1.plot(range(len(hist2)), hist2['BB_upper'], color='#888888', linewidth=0.6)
                        ax1.plot(range(len(hist2)), hist2['BB_lower'], color='#888888', linewidth=0.6)
                        ax1.axhline(y=stop_price,   color='red',    linestyle='--', linewidth=1.2, label=f'損切 {stop_price}')
                        ax1.axhline(y=target_price, color='#00FF7F', linestyle='--', linewidth=1.2, label=f'利確 {target_price}')
                        ax1.text(len(hist2)-1, stop_price,   f' 損切 {stop_price}', color='red',     fontsize=8, va='center')
                        ax1.text(len(hist2)-1, target_price, f' 利確 {target_price}', color='#00FF7F', fontsize=8, va='center')
                        ax1.set_title(f'{code}  {name}', color='white', fontsize=13)
                        ax1.tick_params(colors='#AAAAAA')
                        ax1.set_ylabel('株価 (円)', color='#AAAAAA')
                        for spine in ax1.spines.values():
                            spine.set_edgecolor('#333333')
                        ax1.legend(loc='upper left', fontsize=8, facecolor='#1E1E1E', labelcolor='white', framealpha=0.7)
                        ax1.grid(axis='y', color='#222222', linewidth=0.5)
                        vol_colors = ['#FF4B4B' if c >= o else '#1F77B4'
                                      for c, o in zip(hist2['Close'], hist2['Open'])]
                        ax2.bar(range(len(hist2)), hist2['Volume'], color=vol_colors, alpha=0.7, width=0.6)
                        ax2.plot(range(len(hist2)), hist2['VolMA5'], color='yellow', linewidth=1)
                        ax2.set_ylabel('出来高', color='#AAAAAA')
                        ax2.tick_params(colors='#AAAAAA')
                        for spine in ax2.spines.values():
                            spine.set_edgecolor('#333333')
                        ax2.grid(axis='y', color='#222222', linewidth=0.5)
                        tick_step = max(1, len(hist2) // 8)
                        ticks = list(range(0, len(hist2), tick_step))
                        ax2.set_xticks(ticks)
                        ax2.set_xticklabels(
                            [str(hist2['Date'].iloc[t])[:10] for t in ticks],
                            rotation=30, ha='right', color='#AAAAAA', fontsize=7)
                        plt.tight_layout(h_pad=0.5)
                        return fig
                    except:
                        return None

                chart_codes = buy_only[['コード', '会社名', '損切り価格', '利確目標']].values.tolist()
                for i, (code, name, stop_p, tgt_p) in enumerate(chart_codes):
                    with st.expander(f"📈 {code} {name}　損切:{stop_p}円 / 利確:{tgt_p}円", expanded=(i == 0)):
                        col_left, col_right = st.columns([4, 1])
                        with col_left:
                            with st.spinner(f"{code} チャート読み込み中..."):
                                fig = plot_stock_chart(code, name, float(stop_p), float(tgt_p)) if HAS_MPL else None
                            if fig:
                                st.pyplot(fig, use_container_width=True)
                                plt.close(fig)
                            else:
                                try:
                                    tk3 = yf.Ticker(f"{code}.T")
                                    h3  = tk3.history(period="6mo").reset_index()
                                    if len(h3) > 0:
                                        h3['MA25']  = h3['Close'].rolling(25).mean()
                                        h3['MA200'] = h3['Close'].rolling(200).mean()
                                        h3 = h3.set_index('Date')
                                        st.line_chart(h3[['Close','MA25','MA200']], use_container_width=True)
                                except Exception as e:
                                    st.warning(f"チャート表示失敗: {e}")
                        with col_right:
                            row_data = buy_only[buy_only['コード'] == code].iloc[0]
                            st.markdown(f"**{name}**")
                            st.markdown(f"🔴 損切: **{stop_p}円**")
                            st.markdown(f"🟢 利確: **{tgt_p}円**")
                            st.markdown(f"📊 スコア: **{row_data['スコア']}**")
                            st.markdown(f"RSI: {row_data['RSI(14)']}")
                            st.markdown(f"BB位置: {row_data['BB位置']}")
                            st.markdown(f"反発サイン: {row_data['反発サイン']}")
                            st.markdown(f"BT勝率: {row_data['BT勝率']}")
                            st.link_button("🔗 TradingView", row_data['チャート'])

                if discord_webhook:
                    msg = "【🔥押し目買いサイン点灯】\n" + "\n".join(
                        [f"・{r['コード']} {r['会社名']} スコア{r['スコア']} BT勝率{r['BT勝率']} 反発:{r['反発サイン']}"
                         for _, r in buy_only.iterrows()])
                    requests.post(discord_webhook, json={"content": msg})
                    st.balloons()
            else:
                st.info("現在、押し目買い条件を満たす銘柄はありません。")

            st.markdown("---")
            st.subheader("👀 監視銘柄（押し目形成中）")
            watch = res_df[res_df['判定'] == "👀 監視（押し目形成中）"].sort_values('スコア', ascending=False)
            if not watch.empty:
                st.dataframe(watch, use_container_width=True,
                             column_config={"チャート": st.column_config.LinkColumn("チャート")})
            else:
                st.write("監視銘柄はありません。")

            st.markdown("---")
            st.subheader("⏳ BB下限タッチ済み・反発サイン待ち")
            sign_wait = res_df[res_df['判定'] == "⏳ 様子見（反発サイン待ち）"].sort_values('スコア', ascending=False)
            if not sign_wait.empty:
                st.dataframe(sign_wait, use_container_width=True,
                             column_config={"チャート": st.column_config.LinkColumn("チャート")})
            else:
                st.write("該当銘柄はありません。")

            st.markdown("---")
            st.subheader("⏳ BB下限未タッチ（押し目待ち）")
            bb_not_touched = res_df[res_df['判定'] == "👀 監視（BB下限未タッチ）"].sort_values('スコア', ascending=False)
            if not bb_not_touched.empty:
                st.dataframe(bb_not_touched, use_container_width=True,
                             column_config={"チャート": st.column_config.LinkColumn("チャート")})
            else:
                st.write("該当銘柄はありません。")

            st.markdown("---")
            st.subheader("📊 スキャン統計")
            c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
            c1.metric("解析銘柄数",         len(res_df))
            c2.metric("🔥 買い候補",        len(res_df[res_df['判定'] == "🔥 買い候補"]))
            c3.metric("👀 押し目形成中",     len(res_df[res_df['判定'] == "👀 監視（押し目形成中）"]))
            c4.metric("⏳ 反発サイン待ち",   len(res_df[res_df['判定'] == "⏳ 様子見（反発サイン待ち）"]))
            c5.metric("👀 BB下限未タッチ",   len(res_df[res_df['判定'] == "👀 監視（BB下限未タッチ）"]))
            c6.metric("⛔ BB上部/過熱",      len(res_df[res_df['判定'] == "⛔ 除外（BB上部/過熱）"]))
            c7.metric("⛔ 弱い銘柄",         len(res_df[res_df['判定'] == "⛔ 除外（弱い銘柄）"]))

            st.markdown("---")
            st.subheader("📋 全銘柄一覧")
            st.dataframe(res_df, use_container_width=True,
                         column_config={"チャート": st.column_config.LinkColumn("チャート")})

# ================================================================
# タブ2: 空売りスキャン
# ================================================================
with tab_short:
    st.subheader("🔻 空売りスキャン（信用データ＋テクニカル）")

    with st.expander("⚙️ 空売り条件設定", expanded=False):
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            rsi_short_min = st.slider("RSI下限（空売り）", 30, 60, 40, key="s_rsi_min")
        with sc2:
            rsi_short_max = st.slider("RSI上限（空売り）", 45, 75, 60, key="s_rsi_max")
        with sc3:
            short_stop_pct   = st.slider("損切ライン(%)", 2, 10, 4, key="s_stop")
        with sc4:
            short_target_pct = st.slider("利確ライン(%)", 4, 20, 8, key="s_target")

        st.info(f"""
**【空売り条件】**
1. 株価 < 200日線（下落トレンド確認）
2. RSI **戻り一服ゾーン**（{rsi_short_min}〜{rsi_short_max}）
3. **信用倍率 高い**（2倍以上 → 買い残過多）
4. **売り残 前週比プラス**（売り圧力増加中）
5. 上ヒゲ陰線🔻 / 被せ線⬇️ / 陰線転換↓

🎯 利確: -{short_target_pct}%　💀 損切: +{short_stop_pct}%（上抜け）
        """)

    col_up1, col_up2 = st.columns(2)
    with col_up1:
        short_credit_file = st.file_uploader(
            "📂 ① 信用CSVをアップロード（信用倍率・売り残含む）",
            type=['csv'], key="short_credit_csv"
        )
    with col_up2:
        short_tech_files = st.file_uploader(
            "📂 ② テクニカルCSVをアップロード（銘柄コード・会社名含む）",
            type=['csv'], accept_multiple_files=True, key="short_tech_csv"
        )

    st.caption("💡 信用CSVのみでもスキャン可能です（テクニカルCSVは省略できます）")

    # 信用CSVのみでもスキャンできるよう対応
    if short_credit_file:
        short_credit_file.seek(0)
        try:
            df_credit = pd.read_csv(short_credit_file, encoding='utf-8-sig')
        except:
            short_credit_file.seek(0)
            df_credit = pd.read_csv(short_credit_file, encoding='shift-jis')

        if short_tech_files:
            dfs_tech = []
            for f in short_tech_files:
                f.seek(0)
                try:
                    dfs_tech.append(pd.read_csv(f, encoding='shift-jis'))
                except:
                    f.seek(0)
                    dfs_tech.append(pd.read_csv(f, encoding='utf-8'))
            df_tech = pd.concat(dfs_tech, ignore_index=True).drop_duplicates()
            df_credit['_code'] = df_credit[[c for c in df_credit.columns if 'コード' in c][0]].astype(str).str.strip()
            tech_code_col = [c for c in df_tech.columns if 'コード' in c]
            tech_name_col = [c for c in df_tech.columns if '銘柄名' in c or '会社名' in c]
            if tech_code_col and tech_name_col:
                df_tech['_code'] = df_tech[tech_code_col[0]].astype(str).str.strip()
                df_merged = df_credit.merge(df_tech[['_code', tech_name_col[0]]], on='_code', how='left')
                df_merged['_name'] = df_merged[tech_name_col[0]].fillna(df_merged.get('銘柄名', ''))
            else:
                df_merged = df_credit.copy()
                df_merged['_code'] = df_merged[[c for c in df_credit.columns if 'コード' in c][0]].astype(str).str.strip()
                df_merged['_name'] = df_merged.get('銘柄名', df_merged['_code'])
        else:
            df_credit['_code'] = df_credit[[c for c in df_credit.columns if 'コード' in c][0]].astype(str).str.strip()
            name_col = [c for c in df_credit.columns if '銘柄名' in c or '会社名' in c or '銘柄' in c]
            df_merged = df_credit.copy()
            df_merged['_name'] = df_merged[name_col[0]] if name_col else df_merged['_code']

        st.caption(f"📊 スキャン対象: {len(df_merged)}銘柄")

        # ← スキャンボタン（修正済み）
        if st.button(f"🔻 {len(df_merged)}銘柄を空売りスキャン実行", type="primary"):
            short_results = []
            short_errors  = []
            s_bar         = st.progress(0)
            s_status      = st.empty()

            def safe_float_local(val):
                try:
                    return float(str(val).replace(',', ''))
                except:
                    return np.nan

            def get_credit_col(row, keywords):
                for k in keywords:
                    for c in row.index:
                        if k in str(c):
                            v = row[c]
                            if pd.notna(v):
                                return safe_float_local(v)
                return np.nan

            for i, (_, row) in enumerate(df_merged.iterrows()):
                code = str(row['_code'])
                name = str(row['_name'])
                cr   = get_credit_col(row, ['信用倍率'])
                sc   = get_credit_col(row, ['前週比(売)', '前週比（売）'])
                sblr = get_credit_col(row, ['売買高レシオ'])

                s_status.text(f"空売りスキャン中... {code} {name} ({i+1}/{len(df_merged)}) ✅{len(short_results)}件")

                res = analyze_short(
                    code, name, cr, sc, sblr,
                    stop_pct=short_stop_pct,
                    target_pct=short_target_pct,
                    rsi_short_min=rsi_short_min,
                    rsi_short_max=rsi_short_max,
                )
                if res:
                    short_results.append(res)
                else:
                    short_errors.append(code)
                s_bar.progress((i + 1) / len(df_merged))

            s_status.text(f"✅ 完了！ 成功:{len(short_results)}件 / 失敗:{len(short_errors)}件")
            st.session_state.short_results = short_results

            if discord_webhook and short_results:
                short_df_tmp = pd.DataFrame(short_results)
                cands = short_df_tmp[short_df_tmp['判定'] == "🔻 空売り候補"]
                if not cands.empty:
                    msg = "【🔻空売りサイン点灯】\n" + "\n".join(
                        [f"・{r['コード']} {r['会社名']} スコア{r['スコア']} 信用倍率{r['信用倍率']} 失速:{r['失速サイン']}"
                         for _, r in cands.iterrows()])
                    requests.post(discord_webhook, json={"content": msg})

    # ================================================================
    # 空売り結果表示
    # ================================================================
    if st.session_state.short_results:
        short_df = pd.DataFrame(st.session_state.short_results)

        st.markdown("---")
        st.header("🔻 【厳選】空売り候補")
        short_cands = short_df[short_df['判定'] == "🔻 空売り候補"].sort_values('スコア', ascending=False)

        if not short_cands.empty:
            sc_display = short_cands.copy()
            if 'チャート' in sc_display.columns:
                sc_display['📊'] = sc_display['チャート']

            # ← 列名を analyze_short() の返り値と完全一致させる
            disp_cols = [
                "コード", "会社名", "スコア", "現在値",
                "RSI(14)", "MACD", "BB位置",
                "信用倍率", "売り残前週比",
                "失速サイン",   # 修正: "天井サイン" → "失速サイン"
                "陽線日数",
                "損切り価格", "利確目標", "RRレシオ",
                "200日乖離", "25日乖離", "MA配列",
                "根拠", "注意点", "📊"
            ]
            disp = [c for c in disp_cols if c in sc_display.columns]
            st.dataframe(
                sc_display[disp],
                use_container_width=True,
                column_config={
                    "📊":         st.column_config.LinkColumn("📊", display_text="📊"),
                    "損切り価格": st.column_config.NumberColumn("損切💀", format="%.0f"),
                    "利確目標":   st.column_config.NumberColumn("利確🎯", format="%.0f"),
                    "スコア":     st.column_config.ProgressColumn("スコア", min_value=0, max_value=15),
                },
                height=400,
            )
            csv_short = short_cands[[c for c in disp_cols if c in short_cands.columns and c != '📊']].to_csv(
                index=False, encoding='utf-8-sig')
            st.download_button(
                label     = f"📥 空売り候補CSVダウンロード（{len(short_cands)}銘柄）",
                data      = csv_short,
                file_name = f"空売り候補_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime      = "text/csv",
                use_container_width=True,
            )
        else:
            st.info("空売り候補は現在ありません。条件を調整してみてください。")

        st.markdown("---")
        st.subheader("👀 監視（下落トレンド継続）")
        # ← 判定ステータスを analyze_short() の返り値と完全一致させる
        watch_short = short_df[short_df['判定'] == "👀 監視（下落トレンド継続）"].sort_values('スコア', ascending=False)
        if not watch_short.empty:
            st.dataframe(watch_short, use_container_width=True,
                         column_config={"チャート": st.column_config.LinkColumn("📊")})
        else:
            st.write("該当なし")

        st.markdown("---")
        st.subheader("⏳ 様子見")
        # ← 判定ステータスを analyze_short() の返り値と完全一致させる
        watch_short2 = short_df[short_df['判定'] == "⏳ 様子見"].sort_values('スコア', ascending=False)
        if not watch_short2.empty:
            st.dataframe(watch_short2, use_container_width=True,
                         column_config={"チャート": st.column_config.LinkColumn("📊")})
        else:
            st.write("該当なし")

        st.markdown("---")
        st.subheader("📊 空売りスキャン統計")
        sc1, sc2, sc3, sc4, sc5 = st.columns(5)
        sc1.metric("解析銘柄数",           len(short_df))
        sc2.metric("🔻 空売り候補",        len(short_df[short_df['判定'] == "🔻 空売り候補"]))
        sc3.metric("👀 監視",              len(short_df[short_df['判定'] == "👀 監視（下落トレンド継続）"]))
        sc4.metric("⏳ 様子見",            len(short_df[short_df['判定'] == "⏳ 様子見"]))
        sc5.metric("⛔ 除外/対象外",       len(short_df[short_df['判定'].isin(["⛔ 除外（200日線上・買いスキャン対象）", "➖ 対象外"])]))

        st.markdown("---")
        st.subheader("📋 全銘柄一覧（空売りスキャン）")
        st.dataframe(short_df, use_container_width=True,
                     column_config={"チャート": st.column_config.LinkColumn("📊")})

# ================================================================
# タブ3: AIニュース分析
# ================================================================
with tab_ai:
    st.subheader("📰 AI 投資判断（Gemini）")
    news_input = st.text_area("ニュースをペースト", height=150)
    if st.button("AI分析を実行"):
        if not gemini_key:
            st.warning("⚠️ システム設定欄に Gemini API Key を入力してください")
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
                if not target_model_name:
                    st.error("使えるモデルが見つかりません")
                else:
                    model = genai.GenerativeModel(target_model_name)
                    name  = target_model_name.replace('models/', '')
                    with st.spinner(f"AI ({name}) が分析中..."):
                        prompt = (
                            "あなたは日本株スイングトレードの専門家です。\n"
                            "戦略は「強い銘柄の押し目反発（平均回帰）」と「弱い銘柄の戻り売り（空売り）」の両方です。\n"
                            "以下のニュースを読み、\n"
                            "①相場全体への影響（強気/中立/弱気）\n"
                            "②押し目買いが狙えるセクター・銘柄\n"
                            "③空売りが有効なセクター・銘柄\n"
                            "④スイングトレードの観点で注目すべきポイント\n"
                            "を簡潔に解説してください。\n\nニュース:\n" + news_input
                        )
                        res = model.generate_content(prompt)
                        st.success("分析完了！")
                        st.info(res.text)
            except Exception as e:
                st.error(f"AI解析エラー: {e}")

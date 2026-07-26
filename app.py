import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import google.generativeai as genai
from datetime import datetime
import json, os, base64

st.set_page_config(page_title="アンチグラビティ・コア Pro+", layout="wide")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
    st.session_state.saved_at = ''
if 'short_results' not in st.session_state:
    st.session_state.short_results = None
if 'df_merged' not in st.session_state:
    st.session_state.df_merged = None
if 'discord_webhook' not in st.session_state:
    st.session_state.discord_webhook = ''

st.title("🚀 アンチグラビティ・コア Pro+")
st.caption("シンプル版：200日線✅ BB下限タッチ✅ 反発サイン✅")

with st.expander("⚙️ システム設定", expanded=False):
    row0 = st.columns([2, 2, 1])
    with row0[0]:
        gemini_key = st.text_input("Gemini API Key", type="password")
    with row0[1]:
        _wh = st.text_input("Discord Webhook URL", value=st.session_state.get("discord_webhook",""), type="password")
        if _wh:
            st.session_state.discord_webhook = _wh
        discord_webhook = st.session_state.get("discord_webhook","")
    with row0[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.saved_at = ''
            st.session_state.short_results = None
            st.rerun()

    st.markdown("---")
    c1, c2, c3, c4 = st.columns(4)
    with c1: stop_pct    = st.slider("損切りライン(%)", 1, 10, 4)
    with c2: target_pct  = st.slider("利確ライン(%)", 2, 20, 8)
    with c3: vol_mult    = st.slider("出来高急増除外倍率", 1.5, 5.0, 3.0, step=0.5)
    with c4: min_price   = st.slider("最低株価(円)", 0, 2000, 500, step=100)

    c5, c6, c7 = st.columns(3)
    with c5: min_turnover  = st.slider("最低売買代金(百万円/日)", 0, 2000, 500, step=100)
    with c6: min_atr_pct   = st.slider("最低ATR%（値幅）", 0.0, 5.0, 1.0, step=0.1)
    with c7: min_bt_trades = st.slider("最低BT取引数", 0, 30, 5, step=1)

    st.info(f"""
**【スキャン条件】**
1. ✅ 200日線の上（長期上昇トレンド継続中）
2. ✅ BB下限タッチあり（直近3日以内）
3. ✅ 最低売買代金・ATR・BT条件クリア

💹 売買代金:{min_turnover}百万円以上 📊 ATR:{min_atr_pct}%以上 🔢 BT取引数:{min_bt_trades}回以上
💀 損切:-{stop_pct}% 🎯 利確:+{target_pct}% 最低株価:{min_price}円以上
    """)

    st.markdown("---")
    st.markdown("**📖 用語説明**")
    st.markdown("""
| 用語 | 説明 |
|---|---|
| **200日線** | 過去200日間の終値の平均。これより上にある銘柄は長期上昇トレンド中 |
| **25日線** | 過去25日間の終値の平均。上向きなら短期トレンドも上昇中 |
| **BB（ボリンジャーバンド）** | 株価の振れ幅を示すバンド。下限に触れると「売られすぎ」のサイン |
| **ATR%** | 1日の平均的な値幅（株価に対する%）。高いほど動きやすい銘柄 |
| **売買代金** | 1日の取引金額（百万円）。少ないと売買しにくい薄商い銘柄 |
| **反発サイン** | 詳細表示・一覧で参考確認するサイン（下ヒゲ陽線🔥 / 包み足⚡ など） |
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

# ================================================================
# 反発サイン（情報表示用）
# ================================================================
def check_reversal_sign(hist):
    signs  = []
    latest = hist.iloc[-1]
    prev   = hist.iloc[-2]
    body         = abs(latest['Close'] - latest['Open'])
    lower_wick = min(latest['Close'], latest['Open']) - latest['Low']
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

# ================================================================
# バックテスト
# ================================================================
def backtest(hist, stop_pct, target_pct):
    hist = hist.copy().reset_index()
    trades   = []
    in_trade = False
    entry_price = 0.0
    for i in range(201, len(hist) - 1):
        r      = hist.iloc[i]
        price  = r['Close']
        ma200  = r['MA200']
        bb_lo  = r['BB_lower']
        bb_mid = r['BB_mid']
        if pd.isna(ma200) or pd.isna(bb_lo):
            continue
        if not in_trade:
            if price > ma200 and (r['Low'] <= bb_lo * 1.005 or price <= bb_lo * 1.005):
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
    return {"取引回数": len(arr), "勝率": f"{win_rate:.1f}%",
            "平均損益": f"{avg_pnl:.2f}%", "最大DD": f"{max_dd:.2f}%"}

# ================================================================
# 買いスキャン（トレンド条件のみでシンプル抽出）
# ================================================================
def analyze_stock(ticker_code, company_name, stop_pct, target_pct, vol_mult, min_price, min_turnover, min_atr_pct, min_bt_trades):
    import time
    try:
        hist = pd.DataFrame()
        tk   = None
        for attempt in range(3):
            try:
                tk   = yf.Ticker(f"{ticker_code}.T")
                hist = tk.history(period="2y", timeout=10)
                if len(hist) > 0:
                    break
            except:
                time.sleep(1)
        if len(hist) < 210:
            return None

        hist['MA200'] = hist['Close'].rolling(200).mean()
        hist['MA25']  = hist['Close'].rolling(25).mean()
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

        latest        = hist.iloc[-1]
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
        bb_range     = bb_upper_val - bb_lo_val
        bb_pos       = ((current_price - bb_lo_val) / bb_range * 100) if bb_range > 0 else 50.0

        # 基本フィルター：200日線の上のみ
        if current_price <= ma200:
            return None

        if current_price < min_price:
            return None

        avg_volume   = float(hist['Volume'].rolling(5).mean().iloc[-1])
        avg_turnover = current_price * avg_volume / 1_000_000
        if avg_turnover < min_turnover:
            return None

        hl   = hist['High'] - hist['Low']
        hc   = abs(hist['High'] - hist['Close'].shift())
        lc   = abs(hist['Low']  - hist['Close'].shift())
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_val     = tr.rolling(14).mean().iloc[-1]
        atr_pct_val = atr_val / current_price * 100
        if atr_pct_val < min_atr_pct:
            return None

        # サインやタッチ状態は判定ではなく「情報」として取得
        reversal_signs = check_reversal_sign(hist)
        
        bb_touched = False
        bb_touch_days_ago = 0
        for _bi in range(1, 4):
            if len(hist) > _bi:
                _row   = hist.iloc[-_bi]
                _bb_lo = float(_row['BB_lower'])
                _low   = float(_row['Low'])
                _close = float(_row['Close'])
                if _low <= _bb_lo * 1.005 or _close <= _bb_lo * 1.005:
                    bb_touched        = True
                    bb_touch_days_ago = _bi
                    break

        # BB下限タッチ必須チェック
        if not bb_touched:
            return None

        status = "🔥 買い候補"

        vol_warn = f"⚠️出来高急増({vol_ratio:.1f}x)" if vol_ratio >= vol_mult else ""

        stop_price   = round(current_price * (1 - stop_pct / 100), 1)
        target_price = round(max(current_price * (1 + target_pct / 100), bb_mid_val), 1)
        rr_ratio     = round((target_price - current_price) / (current_price - stop_price), 1) \
                       if current_price > stop_price else 0.0

        atr_pct = round(atr_pct_val, 1)

        if macd_val > sig_val:
            macd_label = "↑上"
        elif abs(macd_val - sig_val) < abs(float(hist.iloc[-2]['MACD']) - float(hist.iloc[-2]['Signal'])):
            macd_label = "🟡 収束中"
        else:
            macd_label = "↓下"

        bt = backtest(hist, stop_pct, target_pct)

        if bt and bt["取引回数"] < min_bt_trades:
            return None

        sign_label = reversal_signs[0] if reversal_signs else "-"
        if "下ヒゲ陽線" in reversal_signs:
            sign_emoji = f"🔥 {sign_label}"
        elif "包み足" in reversal_signs:
            sign_emoji = f"⚡ {sign_label}"
        elif reversal_signs:
            sign_emoji = f"↑ {sign_label}"
        else:
            sign_emoji = "-"

        return {
            "コード":        ticker_code,
            "会社名":        company_name,
            "判定":            status,
            "現在値":        round(current_price, 1),
            "BB位置":        f"{bb_pos:.0f}%({bb_touch_days_ago}日前タッチ)" if bb_touched else f"{bb_pos:.0f}%",
            "RSI(14)":        round(rsi, 1),
            "MACD":            macd_label,
            "ATR%":            f"{atr_pct}%",
            "売買代金(百万)": f"{avg_turnover:.0f}M",
            "反発サイン":     sign_emoji,
            "出来高注意":     vol_warn,
            "損切り価格":     stop_price,
            "利確目標":       target_price,
            "RRレシオ":       f"1:{rr_ratio}",
            "200日乖離":      f"{diff_pct_200:+.1f}%",
            "25日乖離":       f"{diff_pct_25:+.1f}%",
            "チャート":       f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "BT勝率":         bt["勝率"]     if bt else "-",
            "BT平均損益":     bt["平均損益"] if bt else "-",
            "BT取引数":       bt["取引回数"] if bt else 0,
            "BT最大DD":       bt["最大DD"]   if bt else "-",
        }
    except:
        return None

# ================================================================
# 空売り判定（そのまま維持）
# ================================================================
def analyze_short(ticker_code, company_name, credit_ratio, credit_sell_change,
                  credit_sell_buy_ratio, stop_pct=4, target_pct=8,
                  rsi_short_min=40, rsi_short_max=60):
    import time
    try:
        hist = pd.DataFrame()
        for attempt in range(3):
            try:
                tk   = yf.Ticker(f"{ticker_code}.T")
                hist = tk.history(period="2y", timeout=10)
                if len(hist) > 0: break
            except: time.sleep(1)
        if len(hist) < 200: return None

        hist['MA200'] = hist['Close'].rolling(200).mean()
        hist['MA25']  = hist['Close'].rolling(25).mean()
        hist['MA75']  = hist['Close'].rolling(75).mean()
        bb_up, bb_mid, bb_lo = calculate_bb(hist)
        hist['BB_upper'] = bb_up; hist['BB_mid'] = bb_mid; hist['BB_lower'] = bb_lo
        hist['RSI']    = calculate_rsi(hist)
        macd, sig      = calculate_macd(hist)
        hist['MACD']   = macd; hist['Signal'] = sig
        hist['VolMA5'] = hist['Volume'].rolling(5).mean()

        latest = hist.iloc[-1]; prev = hist.iloc[-2]
        current_price = float(latest['Close'])
        ma200 = float(latest['MA200']); ma25 = float(latest['MA25'])
        ma75  = float(latest['MA75']) if not pd.isna(latest['MA75']) else ma200
        rsi   = float(latest['RSI'])
        macd_val = float(latest['MACD']); sig_val = float(latest['Signal'])
        bb_upper_val = float(latest['BB_upper']); bb_lo_val = float(latest['BB_lower'])

        if pd.isna(ma200) or pd.isna(ma25): return None

        diff_pct_200 = (current_price - ma200) / ma200 * 100
        diff_pct_25  = (current_price - ma25)  / ma25  * 100
        bb_range = bb_upper_val - bb_lo_val
        bb_pos   = ((current_price - bb_lo_val) / bb_range * 100) if bb_range > 0 else 50.0

        ma200_5ago = float(hist['MA200'].dropna().iloc[-6]) if len(hist['MA200'].dropna()) >= 6 else ma200
        ma25_5ago  = float(hist['MA25'].dropna().iloc[-6])  if len(hist['MA25'].dropna())  >= 6 else ma25
        ma200_slope_down = ma200 < ma200_5ago
        ma25_slope_down  = ma25  < ma25_5ago

        bullish_count = 0
        for _j in range(1, 6):
            if len(hist) > _j:
                _r = hist.iloc[-_j]
                if _r['Close'] >= _r['Open']: bullish_count += 1
                else: break

        top_signs  = []
        body       = abs(latest['Close'] - latest['Open'])
        upper_wick = latest['High'] - max(latest['Close'], latest['Open'])
        if latest['Close'] < latest['Open'] and body > 0 and upper_wick >= body * 1.5:
            top_signs.append("上ヒゲ陰線")
        if (prev['Close'] >= prev['Open'] and latest['Open'] > prev['Close'] and latest['Close'] < prev['Open']):
            top_signs.append("被せ線")
        elif prev['Close'] >= prev['Open'] and latest['Close'] < latest['Open']:
            if "上ヒゲ陰線" not in top_signs: top_signs.append("陰線転換")
        if body > 0 and upper_wick >= body * 2.0 and "上ヒゲ陰線" not in top_signs:
            top_signs.append("長い上ヒゲ")

        macd_dc_recent = False
        for _i in range(1, 4):
            if len(hist) > _i:
                _p = hist.iloc[-(_i+1)]; _c = hist.iloc[-_i]
                if float(_p['MACD']) > float(_p['Signal']) and float(_c['MACD']) <= float(_c['Signal']):
                    macd_dc_recent = True; break
        macd_below_sig = macd_val < sig_val

        score = 0; reasons = []; warnings = []

        def safe_float(val):
            try: return float(str(val).replace(',', ''))
            except: return np.nan

        # ── 200日線との乖離（上に乖離しているほど空売り好機）──
        if diff_pct_200 >= 30:    score += 4; reasons.append(f"200日線大幅上乖離(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 >= 20:  score += 3; reasons.append(f"200日線上乖離(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 >= 10:  score += 2; reasons.append(f"200日線やや上乖離(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 >= 5:   score += 1; reasons.append(f"200日線上(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 < 0:    warnings.append(f"200日線割れ⚠️ 空売り不適({diff_pct_200:.1f}%)")

        # ── MA配列（上昇配列＝過熱感あり＝空売り好機）──
        if ma25 > ma75 > ma200:   score += 2; reasons.append("完全上昇配列（過熱）")
        elif ma25 > ma200:        score += 1; reasons.append("25日線>200日線")

        # ── RSI（高RSIほど過熱＝空売り好機）──
        if rsi >= 75:             score += 3; reasons.append(f"RSI過熱({rsi:.0f})")
        elif rsi >= 65:           score += 2; reasons.append(f"RSIやや過熱({rsi:.0f})")
        elif rsi >= 55:           score += 1; reasons.append(f"RSI高め({rsi:.0f})")
        elif rsi < 40:            warnings.append(f"RSI低すぎ({rsi:.0f})⚠️ 空売り不適")

        cr = safe_float(credit_ratio)
        if not np.isnan(cr):
            if cr >= 10:   score += 3; reasons.append(f"信用倍率{cr:.1f}(超有利)")
            elif cr >= 5:  score += 2; reasons.append(f"信用倍率{cr:.1f}(有利)")
            elif cr >= 2:  score += 1; reasons.append(f"信用倍率{cr:.1f}")
            elif cr <= 1.0: warnings.append(f"信用倍率{cr:.2f}(踏み上げ注意⚠️)")

        sc_val = safe_float(credit_sell_change)
        if not np.isnan(sc_val):
            if sc_val > 0:        score += 1; reasons.append(f"売り残増加(+{sc_val:,.0f}株)")
            elif sc_val < -10000: warnings.append(f"売り残大幅減少⚠️")

        top_score = 0
        if "被せ線" in top_signs:       top_score = 3
        elif "上ヒゲ陰線" in top_signs: top_score = 2
        elif top_signs:                  top_score = 1
        if top_score > 0:
            score += top_score; label = top_signs[0]
            reasons.append(f"{'🔻' if top_score==3 else '⬇️' if top_score==2 else '↓'}{label}")

        if macd_dc_recent:    score = min(score+1,15); reasons.append("MACD-DC直近")
        elif macd_below_sig:  score = min(score+1,15); reasons.append("MACD下方向")

        # 200日線を割れている銘柄は空売り不適（すでに下がりきり）
        overheated = diff_pct_200 >= 5
        if not overheated:   status = "⛔ 除外（上乖離不足）"
        elif score >= 9:     status = "🔻 空売り候補"
        elif score >= 7:     status = "👀 監視"
        elif score >= 5:     status = "⏳ 様子見"
        else:                status = "➖ 対象外"

        stop_price   = round(current_price * (1 + stop_pct  / 100), 1)
        target_price = round(current_price * (1 - target_pct / 100), 1)
        rr_ratio     = round((current_price - target_price) / (stop_price - current_price), 1) \
                       if stop_price > current_price else 0.0

        if macd_dc_recent:   macd_label = "🔴 DC直近"
        elif macd_below_sig: macd_label = "↓下"
        else:                macd_label = "↑上"

        return {
            "コード": ticker_code, "会社名": company_name, "判定": status, "スコア": score,
            "現在値": round(current_price, 1), "200日乖離": f"{diff_pct_200:+.2f}%",
            "25日乖離": f"{diff_pct_25:+.2f}%", "MA配列": f"25:{ma25:.0f}/75:{ma75:.0f}/200:{ma200:.0f}",
            "RSI(14)": round(rsi, 1), "MACD": macd_label, "BB位置": f"{bb_pos:.0f}%",
            "陽線日数": bullish_count, "失速サイン": " / ".join(top_signs) if top_signs else "-",
            "信用倍率": f"{cr:.2f}" if not np.isnan(cr) else "-",
            "売り残前週比": f"{sc_val:+,.0f}株" if not np.isnan(sc_val) else "-",
            "損切り価格": stop_price, "利確目標": target_price, "RRレシオ": f"1:{rr_ratio}",
            "チャート": f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "根拠": " / ".join(reasons) if reasons else "-",
            "注意点": " / ".join(warnings) if warnings else "-",
        }
    except: return None

# ================================================================
# GitHub watchlist操作
# ================================================================
def get_watchlist_from_github(token, repo):
    try:
        url = f"https://api.github.com/repos/{repo}/contents/watchlist.json"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data    = res.json()
            content = base64.b64decode(data['content']).decode('utf-8')
            return json.loads(content), data['sha']
        return None, None
    except:
        return None, None

def update_watchlist_to_github(token, repo, watchlist_data, sha):
    try:
        url     = f"https://api.github.com/repos/{repo}/contents/watchlist.json"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        content = json.dumps(watchlist_data, ensure_ascii=False, indent=2)
        encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        payload = {
            "message": f"監視銘柄更新 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
            "content": encoded,
            "sha": sha
        }
        res = requests.put(url, headers=headers, json=payload)
        return res.status_code == 200
    except:
        return False

# ================================================================
# タブ構成
# ================================================================
st.markdown("---")
tab_buy, tab_short, tab_ai, tab_watch = st.tabs([
    "📈 買いスキャン", "🔻 空売りスキャン", "📰 AIニュース分析", "👁 監視銘柄登録"
])

# ================================================================
# タブ1: 買いスキャン
# ================================================================
with tab_buy:
    st.subheader("📊 銘柄一括スキャン")
    if st.session_state.analysis_results is not None:
        st.info(f"📂 {st.session_state.get('saved_at','')} スキャン結果（ページを閉じると消えます→CSVで保存）")

    files = st.file_uploader("SBIのCSVをアップロード（複数可）", type=['csv'],
                             accept_multiple_files=True, key="buy_csv")
    if files:
        dfs = []
        for file in files:
            file.seek(0)
            try: dfs.append(pd.read_csv(file, encoding='shift-jis'))
            except:
                file.seek(0); dfs.append(pd.read_csv(file, encoding='utf-8'))
        df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        st.caption(f"📂 {len(files)}ファイル合計 {len(df)}銘柄")

        c_col = [c for c in df.columns if 'コード' in c]
        if not c_col: c_col = [c for c in df.columns if c.strip() == '銘柄']
        n_col = [c for c in df.columns if '銘柄名' in c]
        if not n_col: n_col = [c for c in df.columns if '会社名' in c]
        if not n_col: n_col = [c for c in df.columns if c.strip() == '銘柄.1']

        if not c_col or not n_col:
            st.error(f"⚠️ CSVに「コード」「銘柄名」列が見つかりません: {list(df.columns)}")
        else:
            t_list = df[[c_col[0], n_col[0]]].dropna()
            if st.button(f"🚀 {len(t_list)}銘柄をスキャン"):
                results = []; bar = st.progress(0); status_txt = st.empty()
                for i, (idx, row) in enumerate(t_list.iterrows()):
                    code = str(row[c_col[0]]); name = str(row[n_col[0]])
                    status_txt.text(f"スキャン中... {code} {name} ({i+1}/{len(t_list)}) ✅{len(results)}件")
                    res = analyze_stock(code, name, stop_pct, target_pct, vol_mult, min_price,
                                         min_turnover, min_atr_pct, min_bt_trades)
                    if res: results.append(res)
                    bar.progress((i + 1) / len(t_list))
                status_txt.text(f"✅ 完了！ {len(results)}件")
                if results:
                    st.session_state.analysis_results = pd.DataFrame(results)
                    st.session_state.saved_at = datetime.now().strftime('%Y/%m/%d %H:%M')
                    buy_list = [r for r in results if '買い候補' in r.get('判定','')]
                    os.makedirs('data', exist_ok=True)
                    with open('data/scan_result.json', 'w', encoding='utf-8') as f:
                        json.dump({'updated': st.session_state.saved_at, 'buy': buy_list,
                                   'total': len(results)}, f, ensure_ascii=False, indent=2)
                    st.success(f"✅ {len(results)}銘柄をスキャンしました")
                    if discord_webhook:
                        buys = [r for r in results if '買い候補' in r.get('判定','')]
                        if buys:
                            msg = "【🔥買いシグナル】\n" + "\n".join(
                                [f"・{r['コード']} {r['会社名']} {r['反発サイン']} BT:{r['BT勝率']}"
                                 for r in buys])
                        else:
                            msg = f"【📊スキャン完了】{st.session_state.saved_at}\n本日の買い候補: 0件\nスキャン対象: {len(results)}銘柄"
                        try:
                            res = requests.post(discord_webhook, json={"content": msg}, timeout=10)
                            if res.status_code == 204:
                                if buys:
                                    st.success(f"✅ Discord通知しました（買い候補{len(buys)}件）")
                                else:
                                    st.info("📨 Discord通知しました（本日候補なし）")
                            else:
                                st.error(f"❌ Discord通知失敗: ステータス {res.status_code} / Webhook URLを確認してください")
                        except Exception as e:
                            st.error(f"❌ Discord通知エラー: {e}")
                    else:
                        st.warning("⚠️ Discord Webhook URLが未入力です（⚙️ システム設定で入力してください）")

    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        if not res_df.empty and '判定' in res_df.columns:
            st.markdown("---")
            st.header("🔥 買い候補（トレンド抽出）")
            buy_df = res_df[res_df['判定'].str.contains('買い候補', na=False)].copy()

            if not buy_df.empty:
                if 'チャート' in buy_df.columns: buy_df['📊'] = buy_df['チャート']

                display_cols = ["コード","会社名","判定","現在値",
                                "BB位置","RSI(14)","MACD","ATR%","売買代金(百万)",
                                "反発サイン","出来高注意","損切り価格","利確目標","RRレシオ",
                                "200日乖離","25日乖離","BT勝率","BT平均損益","BT取引数","BT最大DD","📊"]
                disp = [c for c in display_cols if c in buy_df.columns]
                st.dataframe(buy_df[disp], use_container_width=True,
                             column_config={
                                 "📊": st.column_config.LinkColumn("📊", display_text="📊"),
                                 "損切り価格": st.column_config.NumberColumn("損切💀", format="%.0f"),
                                 "利確目標":   st.column_config.NumberColumn("利確🎯", format="%.0f"),
                             }, height=400)

                dl_cols  = [c for c in display_cols if c in buy_df.columns and c not in ["チャート","📊"]]
                csv_data = buy_df[dl_cols].to_csv(index=False, encoding='utf-8-sig')
                st.download_button(label=f"📥 買い候補CSV（{len(buy_df)}銘柄）",
                                   data=csv_data,
                                   file_name=f"買い候補_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                                   mime="text/csv", use_container_width=True)

                st.markdown("---")
                st.subheader("📊 買い候補チャート")
                try:
                    import matplotlib; matplotlib.use('Agg')
                    import matplotlib.pyplot as plt; HAS_MPL = True
                except: HAS_MPL = False

                def plot_stock_chart(code, name, stop_price, target_price):
                    try:
                        tk2 = yf.Ticker(f"{code}.T")
                        hist2 = tk2.history(period="6mo").reset_index()
                        if len(hist2) < 30: return None
                        hist2['MA25']  = hist2['Close'].rolling(25).mean()
                        hist2['MA200'] = hist2['Close'].rolling(200).mean()
                        bb_up2, bb_mid2, bb_lo2 = calculate_bb(hist2)
                        hist2['BB_upper'] = bb_up2; hist2['BB_mid'] = bb_mid2; hist2['BB_lower'] = bb_lo2
                        hist2['VolMA5'] = hist2['Volume'].rolling(5).mean()
                        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7),
                                                       gridspec_kw={'height_ratios': [3, 1]}, sharex=True)
                        fig.patch.set_facecolor('#0E1117')
                        for ax in [ax1, ax2]: ax.set_facecolor('#0E1117')
                        for i, row in hist2.iterrows():
                            color = '#FF4B4B' if row['Close'] >= row['Open'] else '#1F77B4'
                            ax1.plot([i,i],[row['Low'],row['High']], color=color, linewidth=0.8)
                            ax1.bar(i, abs(row['Close']-row['Open']),
                                    bottom=min(row['Open'],row['Close']), color=color, width=0.6, alpha=0.9)
                        ax1.plot(range(len(hist2)), hist2['MA25'],  color='orange',  linewidth=1.2, label='MA25')
                        ax1.plot(range(len(hist2)), hist2['MA200'], color='#00BFFF', linewidth=1.0, label='MA200', linestyle='--')
                        ax1.fill_between(range(len(hist2)), hist2['BB_upper'], hist2['BB_lower'], alpha=0.08, color='gray')
                        ax1.plot(range(len(hist2)), hist2['BB_mid'],   color='#AAAAAA', linewidth=0.8, linestyle=':')
                        ax1.plot(range(len(hist2)), hist2['BB_upper'], color='#888888', linewidth=0.6)
                        ax1.plot(range(len(hist2)), hist2['BB_lower'], color='#888888', linewidth=0.6)
                        ax1.axhline(y=stop_price,    color='red',    linestyle='--', linewidth=1.2, label=f'損切 {stop_price}')
                        ax1.axhline(y=target_price, color='#00FF7F', linestyle='--', linewidth=1.2, label=f'利確 {target_price}')
                        ax1.text(len(hist2)-1, stop_price,    f' 損切 {stop_price}',    color='red',     fontsize=8, va='center')
                        ax1.text(len(hist2)-1, target_price, f' 利確 {target_price}', color='#00FF7F', fontsize=8, va='center')
                        ax1.set_title(f'{code}  {name}', color='white', fontsize=13)
                        ax1.tick_params(colors='#AAAAAA'); ax1.set_ylabel('株価 (円)', color='#AAAAAA')
                        for spine in ax1.spines.values(): spine.set_edgecolor('#333333')
                        ax1.legend(loc='upper left', fontsize=8, facecolor='#1E1E1E', labelcolor='white', framealpha=0.7)
                        ax1.grid(axis='y', color='#222222', linewidth=0.5)
                        vol_colors = ['#FF4B4B' if c>=o else '#1F77B4' for c,o in zip(hist2['Close'],hist2['Open'])]
                        ax2.bar(range(len(hist2)), hist2['Volume'], color=vol_colors, alpha=0.7, width=0.6)
                        ax2.plot(range(len(hist2)), hist2['VolMA5'], color='yellow', linewidth=1)
                        ax2.set_ylabel('出来高', color='#AAAAAA'); ax2.tick_params(colors='#AAAAAA')
                        for spine in ax2.spines.values(): spine.set_edgecolor('#333333')
                        ax2.grid(axis='y', color='#222222', linewidth=0.5)
                        tick_step = max(1, len(hist2)//8); ticks = list(range(0, len(hist2), tick_step))
                        ax2.set_xticks(ticks)
                        ax2.set_xticklabels([str(hist2['Date'].iloc[t])[:10] for t in ticks],
                                            rotation=30, ha='right', color='#AAAAAA', fontsize=7)
                        plt.tight_layout(h_pad=0.5); return fig
                    except: return None

                for i, row in buy_df.iterrows():
                    code=row['コード']; name=row['会社名']
                    stop_p=row.get('損切り価格',0); tgt_p=row.get('利確目標',0)
                    with st.expander(f"📈 {code} {name} {row['判定']}", expanded=(i==buy_df.index[0])):
                        col_left, col_right = st.columns([4,1])
                        with col_left:
                            with st.spinner(f"{code} チャート読み込み中..."):
                                fig = plot_stock_chart(code, name, float(stop_p), float(tgt_p)) if HAS_MPL and stop_p else None
                            if fig: st.pyplot(fig, use_container_width=True); plt.close(fig)
                        with col_right:
                            st.markdown(f"**{name}**")
                            st.markdown(f"🔴 損切: **{stop_p}円**")
                            st.markdown(f"🟢 利確: **{tgt_p}円**")
                            st.markdown(f"📊 BB: {row.get('BB位置','-')}")
                            st.markdown(f"🕯 サイン: {row.get('反発サイン','-')}")
                            st.markdown(f"📈 ATR: {row.get('ATR%','-')}")
                            st.markdown(f"BT勝率: {row.get('BT勝率','-')}")
                            if row.get('チャート'): st.link_button("🔗 TradingView", row['チャート'])
            else:
                st.info("本日、条件を満たす買い候補はありません。")

            st.markdown("---")
            st.subheader("📊 スキャン統計")
            st.metric("🔥 買い候補（抽出数）", len(buy_df))

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
        with sc1: rsi_short_min = st.slider("RSI下限（空売り）", 30, 60, 40, key="s_rsi_min")
        with sc2: rsi_short_max = st.slider("RSI上限（空売り）", 45, 75, 60, key="s_rsi_max")
        with sc3: short_stop_pct   = st.slider("損切ライン(%)", 2, 10, 4, key="s_stop")
        with sc4: short_target_pct = st.slider("利確ライン(%)", 4, 20, 8, key="s_target")

    col_up1, col_up2 = st.columns(2)
    with col_up1: short_credit_files = st.file_uploader("📂 ① 信用CSV", type=['csv'], accept_multiple_files=True, key="short_credit_csv")
    with col_up2: short_tech_files   = st.file_uploader("📂 ② テクニカルCSV", type=['csv'], accept_multiple_files=True, key="short_tech_csv")

    # CSV未アップロード時はsession_stateをリセット
    if not short_credit_files:
        st.session_state.df_merged = None
    df_merged = st.session_state.df_merged
    if short_credit_files:
        dfs_credit = []
        for f in short_credit_files:
            for enc in ['utf-8-sig', 'shift-jis', 'utf-8']:
                try:
                    f.seek(0)
                    dfs_credit.append(pd.read_csv(f, encoding=enc))
                    break
                except Exception:
                    continue
        if not dfs_credit:
            st.error("❌ 信用CSVの読み込みに失敗しました。")
        else:
            df_credit = pd.concat(dfs_credit, ignore_index=True).drop_duplicates()
            # コード列を柔軟に検出
            credit_code_cols = [c for c in df_credit.columns if 'コード' in c or 'code' in c.lower()]
            if not credit_code_cols:
                st.error(f"❌ 信用CSVに「コード」列が見つかりません。列名: {list(df_credit.columns)}")
            else:
                df_credit['_code'] = df_credit[credit_code_cols[0]].astype(str).str.strip()
                # 銘柄名列を検出
                credit_name_cols = [c for c in df_credit.columns if '銘柄名' in c or '会社名' in c or '銘柄' in c]

                if short_tech_files:
                    dfs_tech = []
                    for f in short_tech_files:
                        for enc in ['shift-jis', 'utf-8-sig', 'utf-8']:
                            try:
                                f.seek(0)
                                dfs_tech.append(pd.read_csv(f, encoding=enc))
                                break
                            except Exception:
                                continue
                    if dfs_tech:
                        df_tech = pd.concat(dfs_tech, ignore_index=True).drop_duplicates()
                        tech_code_col = [c for c in df_tech.columns if 'コード' in c or 'code' in c.lower()]
                        tech_name_col = [c for c in df_tech.columns if '銘柄名' in c or '会社名' in c]
                        if tech_code_col:
                            df_tech['_code'] = df_tech[tech_code_col[0]].astype(str).str.strip()
                            if tech_name_col:
                                # mergeしてから列名で安全にアクセス
                                merge_col = tech_name_col[0]
                                df_merged = df_credit.merge(
                                    df_tech[['_code', merge_col]].rename(columns={merge_col: '_tech_name'}),
                                    on='_code', how='left'
                                )
                                # _tech_name → _nameへ。なければ信用CSV銘柄名 → コードの順でフォールバック
                                if credit_name_cols:
                                    df_merged['_name'] = df_merged['_tech_name'].fillna(df_merged[credit_name_cols[0]]).fillna(df_merged['_code'])
                                else:
                                    df_merged['_name'] = df_merged['_tech_name'].fillna(df_merged['_code'])
                            else:
                                df_merged = df_credit.copy()
                                df_merged['_name'] = df_merged[credit_name_cols[0]] if credit_name_cols else df_merged['_code']
                        else:
                            st.error(f"❌ テクニカルCSVにコード列が見つかりません: {list(df_tech.columns)}")
                else:
                    # 信用CSVのみ
                    df_merged = df_credit.copy()
                    df_merged['_name'] = df_merged[credit_name_cols[0]] if credit_name_cols else df_merged['_code']

                if df_merged is not None:
                    st.session_state.df_merged = df_merged
                    st.success(f"✅ {len(df_merged)}銘柄を読み込みました。スキャンを実行してください。")
                    st.caption(f"信用CSV列: {list(df_credit.columns)}")

    df_merged = st.session_state.df_merged
    btn_label = f"🔻 {len(df_merged)}銘柄を空売りスキャン実行" if df_merged is not None else "🔻 空売りスキャン実行（先にCSVをアップロード）"
    if st.button(btn_label, type="primary", disabled=(df_merged is None)):
        short_results = []; s_bar = st.progress(0); s_status = st.empty()
        def safe_float_local(val):
            try: return float(str(val).replace(',', ''))
            except: return np.nan
        def get_credit_col(row, keywords):
            for k in keywords:
                for c in row.index:
                    if k in str(c):
                        v = row[c]
                        if pd.notna(v): return safe_float_local(v)
            return np.nan
        for i, (_, row) in enumerate(df_merged.iterrows()):
            code=str(row['_code']); name=str(row['_name'])
            cr=get_credit_col(row,['信用倍率']); sc=get_credit_col(row,['前週比(売)','前週比（売）'])
            sblr=get_credit_col(row,['売買高レシオ'])
            s_status.text(f"空売りスキャン中... {code} {name} ({i+1}/{len(df_merged)}) ✅{len(short_results)}件")
            res = analyze_short(code, name, cr, sc, sblr, stop_pct=short_stop_pct,
                                target_pct=short_target_pct, rsi_short_min=rsi_short_min, rsi_short_max=rsi_short_max)
            if res: short_results.append(res)
            s_bar.progress((i+1)/len(df_merged))
        s_status.text(f"✅ 完了！ {len(short_results)}件")
        st.session_state.short_results = short_results

    if st.session_state.short_results:
        short_df = pd.DataFrame(st.session_state.short_results)
        st.markdown("---"); st.header("🔻 【厳選】空売り候補")
        short_cands = short_df[short_df['判定'] == "🔻 空売り候補"].sort_values('スコア', ascending=False)
        if not short_cands.empty:
            sc_display = short_cands.copy()
            if 'チャート' in sc_display.columns: sc_display['📊'] = sc_display['チャート']
            disp_cols = ["コード","会社名","スコア","現在値","RSI(14)","MACD","BB位置",
                         "信用倍率","売り残前週比","失速サイン","陽線日数",
                         "損切り価格","利確目標","RRレシオ","200日乖離","25日乖離","MA配列","根拠","注意点","📊"]
            disp = [c for c in disp_cols if c in sc_display.columns]
            st.dataframe(sc_display[disp], use_container_width=True,
                         column_config={"📊": st.column_config.LinkColumn("📊", display_text="📊"),
                                        "損切り価格": st.column_config.NumberColumn("損切💀", format="%.0f"),
                                        "利確目標":   st.column_config.NumberColumn("利確🎯", format="%.0f"),
                                        "スコア":     st.column_config.ProgressColumn("スコア", min_value=0, max_value=15)},
                         height=400)
            csv_short = short_cands[[c for c in disp_cols if c in short_cands.columns and c!='📊']].to_csv(index=False, encoding='utf-8-sig')
            st.download_button(label=f"📥 空売り候補CSV（{len(short_cands)}銘柄）", data=csv_short,
                               file_name=f"空売り候補_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.info("空売り候補は現在ありません。")

        for status_label in ["👀 監視","⏳ 様子見"]:
            st.markdown("---"); st.subheader(status_label)
            tmp = short_df[short_df['判定'] == status_label].sort_values('スコア', ascending=False)
            if not tmp.empty:
                st.dataframe(tmp, use_container_width=True, column_config={"チャート": st.column_config.LinkColumn("📊")})
            else:
                st.write("該当なし")

        st.markdown("---"); st.subheader("📋 全銘柄一覧（空売り）")
        st.dataframe(short_df, use_container_width=True, column_config={"チャート": st.column_config.LinkColumn("📊")})

# ================================================================
# タブ3: AIニュース分析
# ================================================================
with tab_ai:
    st.subheader("📰 AI 投資判断（Gemini）")
    news_input = st.text_area("ニュースをペースト", height=150)
    if st.button("AI分析を実行"):
        if not gemini_key: st.warning("⚠️ Gemini API Keyを入力してください")
        elif not news_input: st.warning("⚠️ ニュースを貼り付けてください")
        else:
            try:
                genai.configure(api_key=gemini_key)
                target_model_name = ""
                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        target_model_name = m.name
                        if 'flash' in m.name or 'pro' in m.name: break
                if not target_model_name: st.error("使えるモデルが見つかりません")
                else:
                    model = genai.GenerativeModel(target_model_name)
                    with st.spinner("AI分析中..."):
                        prompt = ("あなたは日本株スイングトレードの専門家です。\n"
                                  "以下のニュースを読み、\n①相場全体への影響\n②押し目買いが狙えるセクター・銘柄\n"
                                  "③空売りが有効なセクター・銘柄\n④注目すべきポイント\nを簡潔に解説してください。\n\nニュース:\n" + news_input)
                        res = model.generate_content(prompt)
                        st.success("分析完了！"); st.info(res.text)
            except Exception as e: st.error(f"AI解析エラー: {e}")

# ================================================================
# タブ4: 👁 監視銘柄登録
# ================================================================
with tab_watch:
    st.subheader("👁 監視銘柄登録（kabu-alert連携）")
    st.caption("ここで登録した銘柄をGitHub Actionsが15分ごとに監視し、パラボリック上転換でDiscordに通知します")

    try:
        github_token = st.secrets["GITHUB_TOKEN"]
    except:
        github_token = ""
    try:
        github_repo = st.secrets["GITHUB_REPO"]
    except:
        github_repo = "syake1/kabu-alert"

    if not github_token:
        st.warning("⚠️ Streamlit CloudのSecretsに「GITHUB_TOKEN」を設定してください")
    else:
        watchlist_data, sha = get_watchlist_from_github(github_token, github_repo)

        if watchlist_data:
            watchlist = watchlist_data.get('watchlist', [])
            updated   = watchlist_data.get('updated', '-')
            st.info(f"📂 最終更新: {updated} / 監視銘柄数: {len(watchlist)}銘柄")

            if watchlist:
                st.markdown("**現在の監視銘柄：**")
                watch_df = pd.DataFrame(watchlist)
                st.dataframe(watch_df, use_container_width=True)
        else:
            watchlist = []
            sha       = None
            st.warning("watchlist.jsonの取得に失敗しました")

        st.markdown("---")

        st.markdown("**📝 銘柄を追加：**")
        col1, col2, col3 = st.columns([2, 3, 2])
        with col1:
            new_code = st.text_input("銘柄コード", placeholder="例：6507")
        with col2:
            new_name = st.text_input("銘柄名", placeholder="例：シンフォニア")
        with col3:
            new_mode = st.selectbox("監視モード", ["both（買い＋空売り）", "buy（買いのみ）", "short（空売りのみ）"])
            mode_val = new_mode.split("（")[0]

        if st.button("➕ 監視リストに追加", type="primary"):
            if not new_code or not new_name:
                st.error("コードと銘柄名を入力してください")
            elif sha is None:
                st.error("GitHubとの接続に失敗しています")
            else:
                existing_codes = [s['code'] for s in watchlist]
                if new_code in existing_codes:
                    for s in watchlist:
                        if s['code'] == new_code:
                            s['days'] = s.get('days', 0) + 1
                            s['mode'] = mode_val
                    st.info(f"✅ {new_code} {new_name} の連続日数を更新しました")
                else:
                    watchlist.append({
                        "code": new_code,
                        "name": new_name,
                        "days": 1,
                        "mode": mode_val
                    })
                    st.success(f"✅ {new_code} {new_name} を追加しました")

                new_data = {
                    "watchlist": watchlist,
                    "updated":   datetime.now().strftime('%Y/%m/%d %H:%M'),
                    "count":     len(watchlist)
                }
                if update_watchlist_to_github(github_token, github_repo, new_data, sha):
                    st.success("✅ GitHubを更新しました！次の15分チェックから監視開始します")
                    st.rerun()
                else:
                    st.error("❌ GitHub更新に失敗しました")

        st.markdown("---")

        if watchlist:
            st.markdown("**🗑 銘柄を削除：**")
            del_options = [f"{s['code']} {s['name']}" for s in watchlist]
            del_target  = st.selectbox("削除する銘柄を選択", del_options)
            if st.button("🗑 削除", type="secondary"):
                del_code = del_target.split(" ")[0]
                watchlist = [s for s in watchlist if s['code'] != del_code]
                new_data = {
                    "watchlist": watchlist,
                    "updated":   datetime.now().strftime('%Y/%m/%d %H:%M'),
                    "count":     len(watchlist)
                }
                if update_watchlist_to_github(github_token, github_repo, new_data, sha):
                    st.success(f"✅ {del_target} を削除しました")
                    st.rerun()
                else:
                    st.error("❌ GitHub更新に失敗しました")

        st.markdown("---")

        if st.session_state.analysis_results is not None:
            res_df = st.session_state.analysis_results
            buy_df = res_df[res_df['判定'].str.contains('買い候補', na=False)] if '判定' in res_df.columns else pd.DataFrame()

            if not buy_df.empty:
                st.markdown("**⚡ 本日の買い候補を一括追加：**")
                st.dataframe(buy_df[['コード','会社名','判定']].reset_index(drop=True), use_container_width=True)

                if st.button("⚡ 買い候補を全て監視リストに追加", type="primary"):
                    if sha is None:
                        st.error("GitHubとの接続に失敗しています")
                    else:
                        added = 0
                        existing_codes = [s['code'] for s in watchlist]
                        for _, row in buy_df.iterrows():
                            code = str(row['コード'])
                            name = str(row['会社名'])
                            if code in existing_codes:
                                for s in watchlist:
                                    if s['code'] == code:
                                        s['days'] = s.get('days', 0) + 1
                            else:
                                watchlist.append({
                                    "code": code,
                                    "name": name,
                                    "days": 1,
                                    "mode": "both"
                                })
                                added += 1

                        new_data = {
                            "watchlist": watchlist,
                            "updated":   datetime.now().strftime('%Y/%m/%d %H:%M'),
                            "count":     len(watchlist)
                        }
                        if update_watchlist_to_github(github_token, github_repo, new_data, sha):
                            st.success(f"✅ {added}銘柄を追加しました！GitHubが更新されました")
                            st.rerun()
                        else:
                            st.error("❌ GitHub更新に失敗しました")

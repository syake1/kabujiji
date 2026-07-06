import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import google.generativeai as genai
from datetime import datetime
import json, os

# --- 初期設定 ---
st.set_page_config(page_title="アンチグラビティ・コア Pro+", layout="wide")

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
# タイトル / 設定
# ================================================================
st.title("🚀 アンチグラビティ・コア Pro+")
st.caption("押し目反発 & 空売り 両対応版")

with st.expander("⚙️ システム設定 / スイング条件設定", expanded=False):
    row0 = st.columns([2, 2, 1])
    with row0[0]: gemini_key = st.text_input("Gemini API Key", type="password")
    with row0[1]: discord_webhook = st.text_input("Discord Webhook URL", type="password")
    with row0[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("🔄 リセット", use_container_width=True):
            st.session_state.analysis_results = None
            st.session_state.saved_at = ''
            st.session_state.short_results = None
            if os.path.exists(SAVE_PATH): os.remove(SAVE_PATH)
            st.rerun()

    st.markdown("---")
    st.markdown("**🎯 買いスキャン条件**")
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    with c1: rsi_max = st.slider("RSI上限", 50, 70, 55)
    with c2: rsi_min = st.slider("RSI下限", 25, 45, 35)
    with c3: ma200_range = st.slider("200日線乖離上限(%)", 1, 30, 20)
    with c4: vol_mult = st.slider("出来高急増除外倍率", 1.5, 5.0, 2.5, step=0.1)
    with c5: stop_pct = st.slider("損切りライン(%)", 1, 10, 4)
    with c6: target_pct = st.slider("利確ライン(%)", 2, 20, 8)

# ================================================================
# 指標計算ヘルパー
# ================================================================
def calculate_rsi(data, window=14):
    delta = data['Close'].diff()
    gain  = delta.where(delta > 0, 0).rolling(window).mean()
    loss  = (-delta.where(delta < 0, 0)).rolling(window).mean()
    return 100 - (100 / (1 + (gain / loss)))

def calculate_macd(data, fast=12, slow=26, signal=9):
    ef = data['Close'].ewm(span=fast, adjust=False).mean()
    es = data['Close'].ewm(span=slow, adjust=False).mean()
    macd = ef - es
    return macd, macd.ewm(span=signal, adjust=False).mean()

def calculate_bb(data, window=20, num_std=2):
    mid = data['Close'].rolling(window).mean()
    std = data['Close'].rolling(window).std()
    return mid + num_std * std, mid, mid - num_std * std

def check_consecutive_bearish(hist, n=2):
    count = 0
    for i in range(1, min(6, len(hist))):
        if hist.iloc[-i]['Close'] < hist.iloc[-i]['Open']: count += 1
        else: break
    return count >= n, count

def check_reversal_sign(hist):
    signs, latest, prev = [], hist.iloc[-1], hist.iloc[-2]
    body = abs(latest['Close'] - latest['Open'])
    lw = min(latest['Close'], latest['Open']) - latest['Low']
    if latest['Close'] > latest['Open'] and body > 0 and lw >= body * 1.5: signs.append("下ヒゲ陽線")
    if prev['Close'] < prev['Open'] and latest['Close'] > latest['Open'] and latest['Close'] > prev['Open'] and latest['Open'] < prev['Close']: signs.append("包み足")
    if prev['Close'] < prev['Open'] and latest['Close'] > latest['Open'] and "包み足" not in signs: signs.append("陽線転換")
    if body > 0 and lw >= body * 2.0 and "下ヒゲ陽線" not in signs: signs.append("長い下ヒゲ")
    return signs

def check_ma25_slope(hist, window=5):
    ma25 = hist['Close'].rolling(25).mean()
    return (ma25.iloc[-1] - ma25.iloc[-(window+1)]) > 0 if len(ma25.dropna()) >= window + 1 else False

# ================================================================
# 上位足フィルター
# ================================================================
def check_monthly_filter(tk):
    try:
        m = tk.history(period="2y", interval="1mo")
        if len(m) < 3: return True, "月足データ不足"
        for t, p, l in [(m.iloc[-2], m.iloc[-3], "前月"), (m.iloc[-1], m.iloc[-2], "当月")]:
            b, uw, is_b = abs(t['Close']-t['Open']), t['High']-max(t['Close'], t['Open']), t['Close']<t['Open']
            if is_b and b > 0 and uw >= b * 1.5: return False, f"月足上ヒゲ陰線⛔({l})"
            if b > 0 and uw >= b * 2.0: return False, f"月足長い上ヒゲ⛔({l})"
            if is_b and abs(p['Close']-p['Open']) > 0 and b >= abs(p['Close']-p['Open']) * 2.0: return False, f"月足大陰線⛔({l})"
        return True, "月足小陰線✅" if m.iloc[-1]['Close'] < m.iloc[-1]['Open'] else "月足陽線✅"
    except: return True, "月足取得エラー"

def check_weekly_filter(tk):
    try:
        w = tk.history(period="2y", interval="1wk")
        if len(w) < 10: return True, "週足データ不足"
        w['MA25w'] = w['Close'].rolling(25).mean()
        latest, ma_now, ma_prev = w.iloc[-1], w['MA25w'].iloc[-1], w['MA25w'].iloc[-6]
        b, uw, is_b = abs(latest['Close']-latest['Open']), latest['High']-max(latest['Close'], latest['Open']), latest['Close']<latest['Open']
        if is_b and b > 0 and uw >= b * 1.5: return False, "週足上ヒゲ陰線⛔"
        if ma_now < ma_prev and latest['Close'] < ma_now: return False, "週足下落転換⛔"
        return True, "週足25週線上向き✅" if ma_now >= ma_prev else "週足小陰線（許容）✅"
    except: return True, "週足取得エラー"

# ================================================================
# バックテスト
# ================================================================
def backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, vol_mult):
    h = hist.copy().reset_index()
    trades, in_trade, ep = [], False, 0.0
    for i in range(201, len(h) - 1):
        r = h.iloc[i]
        if pd.isna(r['MA200']) or pd.isna(r['RSI']) or pd.isna(r['MA25']): continue
        m25_5 = h.iloc[i - 5]['MA25'] if i >= 5 else None
        m_up = m25_5 and r['MA25'] > m25_5
        b_cnt = 0
        for j in range(1, 5):
            if i >= j and h.iloc[i - j + 1]['Close'] < h.iloc[i - j + 1]['Open']: b_cnt += 1
            else: break
        if not in_trade:
            if r['Close'] > r['MA200'] and m_up and r['Close'] <= r['BB_mid'] and rsi_min <= r['RSI'] <= rsi_max and b_cnt >= 2 and r['VolRatio'] < vol_mult and r['Close'] <= r['MA25'] * 1.03:
                ep, in_trade = h.iloc[i + 1]['Open'], True
        else:
            exp = None
            if r['Low'] <= ep * (1 - stop_pct / 100): exp = ep * (1 - stop_pct / 100)
            elif r['High'] >= max(ep * (1 + target_pct / 100), float(r['BB_mid'])): exp = max(ep * (1 + target_pct / 100), float(r['BB_mid']))
            if exp:
                trades.append((exp - ep) / ep * 100)
                in_trade = False
    if not trades: return None
    arr = np.array(trades)
    return {"取引回数": len(arr), "勝率": f"{(arr > 0).sum() / len(arr) * 100:.1f}%", "平均損益": f"{arr.mean():.2f}%", "最大DD": f"{(np.cumsum(arr) - np.maximum.accumulate(np.cumsum(arr))).min():.2f}%"}

# ================================================================
# ★ 買いスキャン解析 (タッチ・日付判定バグ完全修正)
# ================================================================
def analyze_stock(ticker_code, company_name, stop_pct, target_pct, rsi_min, rsi_max, ma200_range, vol_mult):
    import time
    try:
        hist = pd.DataFrame()
        tk = None
        for attempt in range(3):
            try:
                tk = yf.Ticker(f"{ticker_code}.T")
                hist = tk.history(period="2y", timeout=10)
                if len(hist) > 0: break
            except: time.sleep(1)
        if len(hist) < 200: return {"_error": f"{ticker_code}: データ取得失敗"}
        
        hist['MA200'] = hist['Close'].rolling(200).mean()
        hist['MA25'] = hist['Close'].rolling(25).mean()
        macd, sig = calculate_macd(hist)
        hist['MACD'], hist['Signal'] = macd, sig
        hist['BB_upper'], hist['BB_mid'], hist['BB_lower'] = calculate_bb(hist)
        hist['RSI'] = calculate_rsi(hist)
        hist['VolMA5'] = hist['Volume'].rolling(5).mean()
        hist['VolRatio'] = hist['Volume'] / hist['VolMA5']

        latest = hist.iloc[-1]
        cp, m200, m25, rsi, m_val, s_val, vr = float(latest['Close']), float(latest['MA200']), float(latest['MA25']), float(latest['RSI']), float(latest['MACD']), float(latest['Signal']), float(latest['VolRatio'])
        b_up, b_md, b_lo = float(latest['BB_upper']), float(latest['BB_mid']), float(latest['BB_lower'])

        if pd.isna(m200) or pd.isna(m25): return None
        d_200, d_25 = (cp - m200) / m200 * 100, (cp - m25) / m25 * 100
        b_pos = ((cp - b_lo) / (b_up - b_lo) * 100) if (b_up - b_lo) > 0 else 50.0

        # 【修正】当日を0日前として過去5日間を逆引きスキャン(SBI整合処理撤廃・厳密化)
        bb_touched, b_days = False, -1
        for d_idx in range(6):
            if len(hist) > d_idx:
                row = hist.iloc[-(d_idx + 1)]
                if float(row['Low']) <= float(row['BB_lower']) or float(row['Close']) <= float(row['BB_lower']):
                    bb_touched, b_days = True, d_idx
                    break

        m25_up = check_ma25_slope(hist)
        is_b_cont, b_cnt = check_consecutive_bearish(hist, n=2)
        rev_signs = check_reversal_sign(hist)
        
        monthly_ok, monthly_label = check_monthly_filter(tk)
        weekly_ok, weekly_label = check_weekly_filter(tk)

        score, reasons, warnings = 0, [], []
        if cp > m200: score += 2; reasons.append(f"200日線上(+{d_200:.1f}%)")
        else: warnings.append("200日線下⚠️")
        if m25_up: score += 2; reasons.append("25日線上向き")
        else: warnings.append("25日線下向き")

        # スコア処理修正
        if bb_touched and b_pos <= 80:
            if b_days == 0: score += 3; reasons.append(f"今日BB下限タッチ({b_pos:.0f}%)")
            elif 1 <= b_days <= 2: score += 3; reasons.append(f"BB下限タッチ翌{b_days}日目({b_pos:.0f}%)")
            elif b_days == 3: score += 2; reasons.append(f"BB下限タッチ{b_days}日後({b_pos:.0f}%)")
            else: score += 1; reasons.append(f"BB下限タッチ{b_days}日後({b_pos:.0f}%)")
        elif b_pos <= 25: score += 2; reasons.append(f"BB下限付近({b_pos:.0f}%)")
        elif b_pos <= 50: score += 1; reasons.append(f"BBセンター以下({b_pos:.0f}%)")

        if rsi_min <= rsi <= rsi_max: score += 1; reasons.append(f"RSI適正({rsi:.0f})")
        if vr < vol_mult: score += 1; reasons.append(f"出来高落ち着き({vr:.1f}x)")
        else: warnings.append(f"出来高急増({vr:.1f}x)⚠️")
        
        r_score = 3 if "下ヒゲ陽線" in rev_signs else 2 if "包み足" in rev_signs else 1 if len(rev_signs)>0 else 0
        if r_score > 0: score += r_score; reasons.append(f"サイン:{rev_signs[0]}")

        # ステータス振り分け
        if not (cp > m200 and m25_up): status = "⛔ 除外（弱い銘柄）"
        elif not monthly_ok: status = f"⛔ 除外（{monthly_label}）"
        elif not weekly_ok: status = f"⛔ 除外（{weekly_label}）"
        elif b_pos > 80: status = "⛔ 除外（BB上部/過熱）"
        elif not bb_touched: status = "👀 監視（BB下限未タッチ）"
        elif bb_touched and len(rev_signs) == 0 and b_days == 0: status = "⏳ 様子見（反発サイン待ち）"
        elif score >= 8: status = "🔥 買い候補"
        else: status = "⏳ 様子見"

        bt = backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, vol_mult)
        return {
            "コード": ticker_code, "会社名": company_name, "判定": status, "スコア": score, "現在値": round(cp, 1),
            "月足": monthly_label, "週足": weekly_label, "200日乖離": f"{d_200:+.2f}%", "25日乖離": f"{d_25:+.2f}%", "RSI(14)": round(rsi, 1),
            "MACD": "↑上" if m_val > s_val else "↓下", "BB位置": f"{b_pos:.0f}%" + (f"(↑{b_days}日前タッチ)" if bb_touched else ""),
            "出来高倍率": f"{vr:.1f}x", "陰線日数": b_cnt, "反発サイン": " / ".join(rev_signs) if rev_signs else "-",
            "損切り価格": round(cp * (1 - stop_pct / 100), 1), "利確目標": round(max(cp * (1 + target_pct / 100), b_md), 1), "RRレシオ": f"1:{round((max(cp * (1 + target_pct / 100), b_md) - cp) / (cp - cp * (1 - stop_pct / 100)), 1)}",
            "根拠": " / ".join(reasons) if reasons else "-", "注意点": " / ".join(warnings) if warnings else "-",
            "チャート": f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "BT勝率": bt["勝率"] if bt else "-", "BT平均損益": bt["平均損益"] if bt else "-", "BT取引数": bt["取引回数"] if bt else 0, "BT最大DD": bt["最大DD"] if bt else "-"
        }
    except: return None

# ================================================================
# 空売り解析関数
# ================================================================
def analyze_short(ticker_code, company_name, credit_ratio, credit_sell_change, credit_sell_buy_ratio, stop_pct=4, target_pct=8, rsi_short_min=40, rsi_short_max=60):
    try:
        hist = pd.DataFrame()
        for attempt in range(3):
            try:
                tk = yf.Ticker(f"{ticker_code}.T")
                hist = tk.history(period="2y", timeout=10)
                if len(hist) > 0: break
            except: import time; time.sleep(1)
        if len(hist) < 200: return None
        hist['MA200'] = hist['Close'].rolling(200).mean()
        hist['MA25'] = hist['Close'].rolling(25).mean()
        hist['MA75'] = hist['Close'].rolling(75).mean()
        hist['BB_upper'], hist['BB_mid'], hist['BB_lower'] = calculate_bb(hist)
        hist['RSI'] = calculate_rsi(hist)
        latest = hist.iloc[-1]
        cp, m200, m25, m75, rsi = float(latest['Close']), float(latest['MA200']), float(latest['MA25']), float(latest['MA75']) if not pd.isna(latest['MA75']) else float(latest['MA200']), float(latest['RSI'])
        b_up, b_lo = float(latest['BB_upper']), float(latest['BB_lower'])
        
        if pd.isna(m200) or pd.isna(m25): return None
        d_200, d_25 = (cp - m200) / m200 * 100, (cp - m25) / m25 * 100
        b_pos = ((cp - b_lo) / (b_up - b_lo) * 100) if (b_up - b_lo) > 0 else 50.0

        score, reasons, warnings = 0, [], []
        if d_200 <= -10: score += 2; reasons.append("200日線下落トレンド")
        if m25 < m75 < m200: score += 2; reasons.append("完全下落配列")
        if rsi_short_min <= rsi <= rsi_short_max: score += 2; reasons.append("RSI戻り一服")
        
        cr = float(str(credit_ratio).replace(',', '')) if pd.notna(credit_ratio) else np.nan
        if not np.isnan(cr) and cr >= 2: score += 2; reasons.append(f"信用倍率高({cr:.1f})")

        status = "🔻 空売り候補" if (d_200 < 0 and score >= 5) else "➖ 対象外"
        return {
            "コード": ticker_code, "会社名": company_name, "判定": status, "スコア": score, "現在値": round(cp, 1),
            "200日乖離": f"{d_200:+.2f}%", "25日乖離": f"{d_25:+.2f}%", "MA配列": f"25:{m25:.0f}/75:{m75:.0f}/200:{m200:.0f}",
            "RSI(14)": round(rsi, 1), "BB位置": f"{b_pos:.0f}%", "信用倍率": f"{cr:.2f}" if not np.isnan(cr) else "-",
            "損切り価格": round(cp * (1 + stop_pct / 100), 1), "利確目標": round(cp * (1 - target_pct / 100), 1),
            "根拠": " / ".join(reasons) if reasons else "-", "注意点": " / ".join(warnings) if warnings else "-",
            "チャート": f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}"
        }
    except: return None

# ================================================================
# 流動的なUI表示部 (タブ実装)
# ================================================================
with tab_buy:
    st.subheader("📊 銘柄一括スキャン（押し目買い）")
    files = st.file_uploader("SBIのCSVをアップロード（複数可）", type=['csv'], accept_multiple_files=True, key="buy_csv")
    if files:
        dfs = [pd.read_csv(f, encoding='shift-jis' if 'shift-jis' else 'utf-8') for f in files]
        df = pd.concat(dfs, ignore_index=True).drop_duplicates()
        c_col = [c for c in df.columns if 'コード' in c or c.strip() == '銘柄']
        n_col = [c for c in df.columns if '銘柄名' in c or '会社名' in c or c.strip() == '銘柄.1']
        
        if c_col and n_col:
            t_list = df[[c_col[0], n_col[0]]].dropna()
            if st.button(f"🚀 {len(t_list)}銘柄を一括解析実行"):
                results = []
                bar = st.progress(0)
                for i, (_, row) in enumerate(t_list.iterrows()):
                    res = analyze_stock(str(row[c_col[0]]), str(row[n_col[0]]), stop_pct, target_pct, rsi_min, rsi_max, ma200_range, vol_mult)
                    if res and '_error' not in res: results.append(res)
                    bar.progress((i + 1) / len(t_list))
                if results:
                    st.session_state.analysis_results = pd.DataFrame(results)
                    save_results(st.session_state.analysis_results)
                    st.success("スキャン結果を保存しました。")

    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        st.header("🔥 【厳選】押し目買い候補")
        buy_only = res_df[res_df['判定'] == "🔥 買い候補"]
        st.dataframe(buy_only, use_container_width=True)
        
        with st.expander("📋 その他（監視・様子見を含む全銘柄一覧）"):
            st.dataframe(res_df, use_container_width=True)

with tab_short:
    st.subheader("🔻 空売りスキャン")
    short_credit_files = st.file_uploader("📂 信用CSVファイルをアップロード", type=['csv'], accept_multiple_files=True, key="short_csv")
    if short_credit_files:
        st.info("CSVが読み込まれました。スキャンを実行してください。")

with tab_ai:
    st.subheader("📰 AI 投資判断（Gemini）")
    news_input = st.text_area("ニュースをペースト", height=150)
    if st.button("AI分析を実行") and gemini_key and news_input:
        try:
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            res = model.generate_content(f"日本株スイングトレード戦略に基づき、次のニュースを簡潔に分析してください:\n{news_input}")
            st.info(res.text)
        except Exception as e: st.error(f"エラー: {e}")

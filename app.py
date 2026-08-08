import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import google.generativeai as genai
from datetime import datetime, timedelta
import json, os, base64
from collections import Counter

st.set_page_config(page_title="アンチグラビティ・コア Pro+", layout="wide")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
    st.session_state.saved_at = ''
if 'short_results' not in st.session_state:
    st.session_state.short_results = None
if 'df_merged' not in st.session_state:
    st.session_state.df_merged = None
if 'discord_webhook' not in st.session_state:
    st.session_state.discord_webhook = st.secrets.get('DISCORD_WEBHOOK', '')

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

    # --- 追加：高値追い除外フィルター ---
    c8, c9 = st.columns(2)
    with c8:
        max_extension_pct = st.slider(
            "直近安値からの上昇率上限(%)　※超えたら除外", 20, 150, 60, step=5,
            help="直近60日の安値からこの%以上すでに上昇している銘柄は「上げすぎ」として買い候補から除外します"
        )
    with c9:
        max_ma25_gap_pct = st.slider(
            "25日線からの乖離率上限(%)　※超えたら除外", 5, 50, 20, step=1,
            help="25日移動平均線からこの%以上乖離（上に離れすぎ）している銘柄を過熱として除外します"
        )

    c10, c11 = st.columns(2)
    with c10:
        exclude_ma25_down = st.checkbox(
            "25日線が下向きの銘柄を除外", value=True,
            help="5営業日前と比較して25日移動平均線が下降している銘柄は、短期トレンドが崩れているとして除外します"
        )
    with c11:
        min_bt_win_rate = st.slider(
            "最低BT勝率(%)　※未満は除外", 0, 80, 40, step=5,
            help="バックテストの勝率がこの%未満の銘柄は、過去の同条件エントリーでの成績が悪いとして除外します"
        )

    max_bounce_from_touch_pct = st.slider(
        "BB下限タッチ日からの上昇率上限(%)　※超えたら除外", 1, 20, 5, step=1,
        help="BB下限タッチを検知した日の安値からこの%以上すでに株価が戻っている銘柄は、"
             "「底値からもう反発済み」として除外します。中央線に届いていなくても、"
             "タッチ直後の底値から一定以上戻ってしまうと高値掴みリスクが高いためです。"
    )
    bb_touch_lookback_days = st.slider(
        "BB下限タッチ検知の遡り日数", 1, 10, 7, step=1,
        help="何営業日前までのBB下限タッチを候補として拾うか。短くすると直近のタッチしか拾わず"
             "母数が減り、長くすると少し前にタッチした銘柄も拾えます（戻り率フィルターと併用で、"
             "古いタッチでもまだ戻りが小さいものだけが残ります）。"
    )
    max_ma200_gap_pct = st.slider(
        "200日線からの上乖離上限(%)　※超えたら除外", 5, 50, 15, step=5,
        help="200日線より上であること自体は必須条件ですが、上に離れすぎている銘柄（例：三井住友FGのように"
             "半年で+40%と一本調子で上昇してきた銘柄）は、BB下限にタッチしても単なる強トレンド中の"
             "小休止であって、本来狙いたい『底からの反発』ではありません。この上限を超えたら除外します。"
    )

    st.info(f"""
**【スキャン条件】**
1. ✅ 200日線の上（長期上昇トレンド継続中）
2. ✅ BB下限タッチあり（直近3日以内）
3. ✅ 最低売買代金・ATR・BT条件クリア
4. ✅ 高値追いでない（安値比・25日線乖離が上限以内）

💹 売買代金:{min_turnover}百万円以上 📊 ATR:{min_atr_pct}%以上 🔢 BT取引数:{min_bt_trades}回以上
💀 損切:-{stop_pct}% 🎯 利確:+{target_pct}% 最低株価:{min_price}円以上
🚫 安値比上昇率上限:+{max_extension_pct}% 🚫 25日線乖離上限:+{max_ma25_gap_pct}%
🚫 25日線下向き除外:{'ON' if exclude_ma25_down else 'OFF'} 🚫 最低BT勝率:{min_bt_win_rate}%
🚫 タッチ日からの戻り上限:+{max_bounce_from_touch_pct}% 🔍 タッチ遡り日数:{bb_touch_lookback_days}日
🚫 200日線乖離上限:+{max_ma200_gap_pct}%
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
| **安値比上昇率** | 直近60日安値から現在値までの上昇率。高すぎると「もう上げすぎ」の可能性 |
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
    body       = abs(latest['Close'] - latest['Open'])
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
# 買い候補ランキング（ルールベース・根拠が全部見えるスコアリング）
# ================================================================
def _parse_pct(val):
    """'44.4%' や '+1.5%' のような文字列を数値(float)に変換"""
    try:
        if val is None:
            return None
        s = str(val).replace('%', '').replace('+', '').strip()
        if s in ('', '-', 'nan'):
            return None
        return float(s)
    except Exception:
        return None

def _parse_rr(val):
    """'1:2.0' のようなRRレシオ文字列から後ろの数値だけを取り出す"""
    try:
        s = str(val)
        if ':' not in s:
            return None
        return float(s.split(':')[1])
    except Exception:
        return None

def rank_buy_candidates(buy_df):
    """
    買い候補DataFrameにスコア・順位・スコア内訳を付けて返す。
    重み付けは全部ここに集約（調整したい場合はここの数値を変える）。
    """
    rows = []
    for _, row in buy_df.iterrows():
        breakdown = {}

        # ① バックテスト勝率（高いほど加点、最大25点）
        bt_win = _parse_pct(row.get('BT勝率'))
        pt = 0.0
        if bt_win is not None:
            pt = max(0.0, min(25.0, (bt_win - 30) / 40 * 25))  # 30%=0点, 70%=25点の目安
        breakdown['BT勝率'] = round(pt, 1)

        # ② バックテスト平均損益（高いほど加点、最大20点）
        bt_avg = _parse_pct(row.get('BT平均損益'))
        pt = 0.0
        if bt_avg is not None:
            pt = max(0.0, min(20.0, bt_avg / 3 * 20))  # 3%で満点目安
        breakdown['BT平均損益'] = round(pt, 1)

        # ③ バックテスト取引数（サンプル数が多いほど信頼度が高い、最大10点）
        bt_n = row.get('BT取引数', 0)
        try:
            bt_n = float(bt_n)
        except Exception:
            bt_n = 0
        pt = max(0.0, min(10.0, bt_n / 20 * 10))  # 20回で満点目安
        breakdown['BT取引数（信頼度）'] = round(pt, 1)

        # ④ RRレシオ（リスクリワード比、高いほど加点、最大20点）
        rr = _parse_rr(row.get('RRレシオ'))
        pt = 0.0
        if rr is not None:
            pt = max(0.0, min(20.0, (rr - 1) / 2 * 20))  # 1:1=0点, 1:3=20点の目安
        breakdown['RRレシオ'] = round(pt, 1)

        # ⑤ ATR%（値動きの大きさ、1.5〜3.0%あたりを最適とみなし山型で評価、最大10点）
        atr = _parse_pct(row.get('ATR%'))
        pt = 0.0
        if atr is not None:
            pt = max(0.0, 10.0 - abs(atr - 2.2) * 4)
            pt = min(10.0, pt)
        breakdown['ATR%（値動きの適度さ）'] = round(pt, 1)

        # ⑥ 反発サインの強さ（最大10点）
        sign = str(row.get('反発サイン', ''))
        if '🔥' in sign:
            pt = 10.0
        elif '⚡' in sign:
            pt = 7.0
        elif '↑' in sign:
            pt = 4.0
        else:
            pt = 0.0
        breakdown['反発サインの強さ'] = pt

        # ⑦ 出来高急増の警告（あれば減点、最大-5点）
        vol_warn = str(row.get('出来高注意', ''))
        pt = -5.0 if vol_warn and vol_warn != 'nan' and vol_warn.strip() != '' else 0.0
        breakdown['出来高急増ペナルティ'] = pt

        # ⑧ 高値追いペナルティ（安値比上昇率が高いほど減点、最大-15点）
        ext = _parse_pct(row.get('安値比'))
        pt = 0.0
        if ext is not None and ext > 30:
            pt = -min(15.0, (ext - 30) / 30 * 15)
        breakdown['高値追いペナルティ'] = round(pt, 1)

        total = round(sum(breakdown.values()), 1)
        rows.append({
            "コード": row.get('コード'),
            "会社名": row.get('会社名'),
            "スコア": total,
            "内訳": breakdown,
            "元データ": row,
        })

    ranked = sorted(rows, key=lambda x: x['スコア'], reverse=True)
    return ranked

# ================================================================
# 買いスキャン（トレンド条件のみでシンプル抽出）
# 戻り値: (結果dict または None, 除外理由の文字列)
# ================================================================
def analyze_stock(ticker_code, company_name, stop_pct, target_pct, vol_mult, min_price,
                   min_turnover, min_atr_pct, min_bt_trades,
                   max_extension_pct=60, max_ma25_gap_pct=20,
                   exclude_ma25_down=True, min_bt_win_rate=40,
                   max_bounce_from_touch_pct=5, bb_touch_lookback_days=7,
                   max_ma200_gap_pct=15):
    import time
    try:
        hist = pd.DataFrame()
        tk   = None
        last_err = ""
        for attempt in range(3):
            try:
                tk   = yf.Ticker(f"{ticker_code}.T")
                hist = tk.history(period="2y", timeout=10)
                if len(hist) > 0:
                    break
            except Exception as e:
                last_err = str(e)
                time.sleep(1)
        if len(hist) < 210:
            reason = f"データ取得失敗/不足({len(hist)}件)"
            if last_err:
                reason += f" [{last_err[:80]}]"
            return None, reason

        # 株価データに欠損(NaN)行が混じっていると移動平均が計算不能になるため除去
        hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close'])

        if len(hist) < 210:
            return None, f"欠損データ除去後に不足({len(hist)}件)"

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
            try:
                latest_date = str(hist.index[-1])[:10]
            except Exception:
                latest_date = "?"
            return None, f"MA計算不可(NaN) 行数={len(hist)} 最新日={latest_date}"

        diff_pct_200 = (current_price - ma200) / ma200 * 100
        diff_pct_25  = (current_price - ma25)  / ma25  * 100
        bb_range     = bb_upper_val - bb_lo_val
        bb_pos       = ((current_price - bb_lo_val) / bb_range * 100) if bb_range > 0 else 50.0

        if current_price <= ma200:
            return None, "200日線割れ"

        if diff_pct_200 > max_ma200_gap_pct:
            return None, f"200日線から乖離しすぎ(+{diff_pct_200:.1f}%)"

        if current_price < min_price:
            return None, "最低株価未満"

        # --- 追加：高値追い除外フィルター ---
        low_60 = float(hist['Low'].tail(60).min())
        extension_pct = (current_price - low_60) / low_60 * 100 if low_60 > 0 else 0.0
        if extension_pct > max_extension_pct:
            return None, f"急騰しすぎ(60日安値比+{extension_pct:.0f}%)"

        if diff_pct_25 > max_ma25_gap_pct:
            return None, f"25日線乖離しすぎ(+{diff_pct_25:.1f}%)"

        # --- 追加：25日線が下向きの銘柄を除外 ---
        ma25_series = hist['MA25'].dropna()
        if exclude_ma25_down and len(ma25_series) >= 6:
            ma25_5ago = float(ma25_series.iloc[-6])
            if ma25 < ma25_5ago:
                return None, f"25日線下向き({ma25_5ago:.0f}→{ma25:.0f})"

        avg_volume   = float(hist['Volume'].rolling(5).mean().iloc[-1])
        avg_turnover = current_price * avg_volume / 1_000_000
        if avg_turnover < min_turnover:
            return None, "売買代金不足"

        hl   = hist['High'] - hist['Low']
        hc   = abs(hist['High'] - hist['Close'].shift())
        lc   = abs(hist['Low']  - hist['Close'].shift())
        tr   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
        atr_val      = tr.rolling(14).mean().iloc[-1]
        atr_pct_val  = atr_val / current_price * 100
        if atr_pct_val < min_atr_pct:
            return None, "ATR%不足"

        reversal_signs = check_reversal_sign(hist)

        bb_touched = False
        bb_touch_days_ago = 0
        bb_touch_low = None
        for _bi in range(1, bb_touch_lookback_days + 1):
            if len(hist) > _bi:
                _row   = hist.iloc[-_bi]
                _bb_lo = float(_row['BB_lower'])
                _low   = float(_row['Low'])
                _close = float(_row['Close'])
                if _low <= _bb_lo * 1.005 or _close <= _bb_lo * 1.005:
                    bb_touched        = True
                    bb_touch_days_ago = _bi - 1  # _bi=1はhist.iloc[-1]=今日を指すため-1で補正
                    bb_touch_low      = _low
                    break

        if not bb_touched:
            return None, "BB下限タッチなし"

        # --- 追加：BB下限タッチ日の安値から、現在値がすでに一定以上
        #     戻ってしまっている銘柄を除外 ---
        # 中央線に届いていなくても、タッチ直後の底値から大きく戻った後だと
        # 「これから買う」には遅く、高値掴みのリスクが高いため除外する。
        bounce_from_touch_pct = 0.0
        if bb_touch_low and bb_touch_low > 0:
            bounce_from_touch_pct = (current_price - bb_touch_low) / bb_touch_low * 100
            if bounce_from_touch_pct > max_bounce_from_touch_pct:
                return None, f"タッチ日安値から戻りすぎ(+{bounce_from_touch_pct:.1f}%)"

        # --- 追加：BB下限タッチ後、すでに中央線（当初の反発目標）を
        #     上抜けてしまっている銘柄を除外 ---
        # BB下限タッチは「逆張りエントリーの根拠」だが、その後すでに
        # 中央線まで戻ってしまっている場合、反発の旨味を取り終えた後であり
        # 「これから買う」には遅すぎる（高値掴みのリスクが高い）ため除外する。
        if current_price > bb_mid_val:
            return None, f"BB中央線を上抜け済み(反発完了・逆張り機会終了) 現在値位置:{bb_pos:.0f}%"

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
            return None, "BT取引数不足"

        if bt:
            try:
                bt_win_val = float(str(bt["勝率"]).replace('%', ''))
                if bt_win_val < min_bt_win_rate:
                    return None, f"BT勝率不足({bt['勝率']})"
            except Exception:
                pass

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
            "判定":          status,
            "現在値":        round(current_price, 1),
            "BB位置":        f"{bb_pos:.0f}%({bb_touch_days_ago}日前タッチ/戻り+{bounce_from_touch_pct:.1f}%)" if bb_touched else f"{bb_pos:.0f}%",
            "RSI(14)":       round(rsi, 1),
            "MACD":            macd_label,
            "ATR%":            f"{atr_pct}%",
            "売買代金(百万)": f"{avg_turnover:.0f}M",
            "反発サイン":     sign_emoji,
            "出来高注意":     vol_warn,
            "損切り価格":     stop_price,
            "利確目標":        target_price,
            "RRレシオ":        f"1:{rr_ratio}",
            "200日乖離":      f"{diff_pct_200:+.1f}%",
            "25日乖離":       f"{diff_pct_25:+.1f}%",
            "安値比":         f"{extension_pct:.0f}%",
            "チャート":        f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "BT勝率":        bt["勝率"]     if bt else "-",
            "BT平均損益":     bt["平均損益"] if bt else "-",
            "BT取引数":       bt["取引回数"] if bt else 0,
            "BT最大DD":       bt["最大DD"]   if bt else "-",
        }, "OK"
    except Exception as e:
        return None, f"例外: {e}"

# ================================================================
# 空売り判定
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

        hist = hist.dropna(subset=['Open', 'High', 'Low', 'Close'])
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

        if diff_pct_200 >= 30:    score += 4; reasons.append(f"200日線大幅上乖離(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 >= 20:  score += 3; reasons.append(f"200日線上乖離(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 >= 10:  score += 2; reasons.append(f"200日線やや上乖離(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 >= 5:   score += 1; reasons.append(f"200日線上(+{diff_pct_200:.1f}%)")
        elif diff_pct_200 < 0:    warnings.append(f"200日線割れ⚠️ 空売り不適({diff_pct_200:.1f}%)")

        if ma25 > ma75 > ma200:   score += 2; reasons.append("完全上昇配列（過熱）")
        elif ma25 > ma200:        score += 1; reasons.append("25日線>200日線")

        if rsi >= 75:             score += 3; reasons.append(f"RSI過熱({rsi:.0f})")
        elif rsi >= 65:           score += 2; reasons.append(f"RSIやや過熱({rsi:.0f})")
        elif rsi >= 55:           score += 1; reasons.append(f"RSI高め({rsi:.0f})")
        elif rsi < 40:            warnings.append(f"RSI低すぎ({rsi:.0f})⚠️ 空売り不適")

        cr = safe_float(credit_ratio)
        if not np.isnan(cr):
            if cr >= 10:    score += 3; reasons.append(f"信用倍率{cr:.1f}(超有利)")
            elif cr >= 5:   score += 2; reasons.append(f"信用倍率{cr:.1f}(有利)")
            elif cr >= 2:   score += 1; reasons.append(f"信用倍率{cr:.1f}")
            elif cr <= 1.0: warnings.append(f"信用倍率{cr:.2f}(踏み上げ注意⚠️)")

        sc_val = safe_float(credit_sell_change)
        if not np.isnan(sc_val):
            if sc_val > 0:        score += 1; reasons.append(f"売り残増加(+{sc_val:,.0f}株)")
            elif sc_val < -10000: warnings.append(f"売り残大幅減少⚠️")

        top_score = 0
        if "被せ線" in top_signs:        top_score = 3
        elif "上ヒゲ陰線" in top_signs: top_score = 2
        elif top_signs:                  top_score = 1
        if top_score > 0:
            score += top_score; label = top_signs[0]
            reasons.append(f"{'🔻' if top_score==3 else '⬇️' if top_score==2 else '↓'}{label}")

        if macd_dc_recent:    score = min(score+1,15); reasons.append("MACD-DC直近")
        elif macd_below_sig:  score = min(score+1,15); reasons.append("MACD下方向")

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
# 買い銘柄トラッキング機能：GitHub連携＆価格取得
# ================================================================
TRACKING_FILE = "tracking.json"

def get_tracking_from_github(token, repo):
    try:
        url = f"https://api.github.com/repos/{repo}/contents/{TRACKING_FILE}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        res = requests.get(url, headers=headers)
        if res.status_code == 200:
            data    = res.json()
            content = base64.b64decode(data['content']).decode('utf-8')
            return json.loads(content), data['sha']
        elif res.status_code == 404:
            return {"entries": []}, None  # ファイル未作成
        return None, None
    except:
        return None, None

def save_tracking_to_github(token, repo, tracking_data, sha):
    try:
        url     = f"https://api.github.com/repos/{repo}/contents/{TRACKING_FILE}"
        headers = {"Authorization": f"token {token}", "Accept": "application/vnd.github.v3+json"}
        content = json.dumps(tracking_data, ensure_ascii=False, indent=2)
        encoded = base64.b64encode(content.encode('utf-8')).decode('utf-8')
        payload = {
            "message": f"トラッキング更新 {datetime.now().strftime('%Y/%m/%d %H:%M')}",
            "content": encoded,
        }
        if sha:
            payload["sha"] = sha
        res = requests.put(url, headers=headers, json=payload)
        return res.status_code in (200, 201)
    except:
        return False

@st.cache_data(ttl=3600)
def get_close_price_on_date(code, date_str):
    """指定日以前で直近の営業日終値を取得（過去日は結果が変わらないので1時間キャッシュ）"""
    try:
        target = pd.Timestamp(date_str)
        tk = yf.Ticker(f"{code}.T")
        hist = tk.history(start=(target - pd.Timedelta(days=10)).strftime('%Y-%m-%d'),
                           end=(target + pd.Timedelta(days=2)).strftime('%Y-%m-%d'))
        hist = hist.dropna(subset=['Close'])
        if hist.empty:
            return None
        hist = hist[hist.index.date <= target.date()]
        if hist.empty:
            return None
        return float(hist['Close'].iloc[-1])
    except Exception:
        return None

def parse_pasted_stock_table(raw_text):
    """タブ区切り・カンマ区切り両対応でSBI貼り付けテーブルをパース"""
    import io
    raw_text = raw_text.strip()
    if not raw_text:
        return None
    try:
        df = pd.read_csv(io.StringIO(raw_text), sep=None, engine='python')
    except Exception:
        return None

    # 列名のゆらぎを吸収
    rename_map = {}
    for col in df.columns:
        c = str(col).strip()
        if '銘柄コード' in c or 'コード' in c:
            rename_map[col] = 'code'
        elif '銘柄名' in c or '会社名' in c or '銘柄' in c:
            rename_map[col] = 'name'
        elif '終値' in c or '現在値' in c:
            rename_map[col] = 'close'
        elif '前日比率' in c or '騰落率' in c:
            rename_map[col] = 'pct'
        elif c == '前日比':
            rename_map[col] = 'diff'
    df = df.rename(columns=rename_map)

    required = ['code', 'name', 'close']
    if not all(c in df.columns for c in required):
        return None

    df['code'] = df['code'].astype(str).str.strip()
    df['close'] = pd.to_numeric(df['close'].astype(str).str.replace(',', ''), errors='coerce')
    if 'pct' in df.columns:
        df['pct'] = df['pct'].astype(str).str.replace('%', '').str.replace('+', '')
        df['pct'] = pd.to_numeric(df['pct'], errors='coerce')
    return df.dropna(subset=['code', 'close'])

def build_tracking_display(entries):
    """トラッキング中の全銘柄について、D+1〜D+5の株価・騰落率を計算して表を作る"""
    rows = []
    updated_any = False
    today = pd.Timestamp.now().normalize()

    for e in entries:
        entry_date = pd.Timestamp(e['entry_date'])
        row = {
            "銘柄": f"{e['code']} {e['name']}",
            "登録日": entry_date.strftime('%m/%d'),
            "登録時終値": e['entry_price'],
        }
        bdays = pd.bdate_range(start=entry_date, periods=6)[1:6]  # D+1〜D+5
        day_prices = e.setdefault('day_prices', {})
        last_valid_pct = None

        for i, d in enumerate(bdays, start=1):
            key = f"d{i}"
            if d > today:
                row[f"D+{i}"] = "―"
                continue
            if key not in day_prices:
                price = get_close_price_on_date(e['code'], d.strftime('%Y-%m-%d'))
                if price is not None:
                    day_prices[key] = price
                    updated_any = True
            price = day_prices.get(key)
            if price is not None:
                pct = (price - e['entry_price']) / e['entry_price'] * 100
                last_valid_pct = pct
                arrow = "🔺" if pct >= 0 else "🔻"
                row[f"D+{i}"] = f"{arrow}{pct:+.1f}%"
            else:
                row[f"D+{i}"] = "取得中"

        row["状況"] = "追跡中" if bdays[-1] > today else "完了"
        row["_sort"] = entry_date
        rows.append(row)

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    return df, updated_any

@st.cache_data(ttl=300)
def get_watch_current_price(code):
    """監視銘柄一覧用：現在値だけを軽量に取得する（5分キャッシュ）"""
    try:
        tk = yf.Ticker(f"{code}.T")
        hist = tk.history(period="5d")
        hist = hist.dropna(subset=['Close'])
        if hist.empty:
            return None
        return float(hist['Close'].iloc[-1])
    except Exception:
        return None

# ================================================================
# タブ構成
# ================================================================
st.markdown("---")
tab_buy, tab_short, tab_ai, tab_watch, tab_track = st.tabs([
    "📈 買いスキャン", "🔻 空売りスキャン", "📰 AIニュース分析", "👁 監視銘柄登録", "📋 買い銘柄トラッキング"
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
                fail_reasons = Counter()
                for i, (idx, row) in enumerate(t_list.iterrows()):
                    code = str(row[c_col[0]]); name = str(row[n_col[0]])
                    status_txt.text(f"スキャン中... {code} {name} ({i+1}/{len(t_list)}) ✅{len(results)}件")
                    res, reason = analyze_stock(code, name, stop_pct, target_pct, vol_mult, min_price,
                                         min_turnover, min_atr_pct, min_bt_trades,
                                         max_extension_pct, max_ma25_gap_pct,
                                         exclude_ma25_down, min_bt_win_rate,
                                         max_bounce_from_touch_pct, bb_touch_lookback_days,
                                         max_ma200_gap_pct)
                    if res:
                        results.append(res)
                    else:
                        fail_reasons[reason] += 1
                    bar.progress((i + 1) / len(t_list))
                status_txt.text(f"✅ 完了！ {len(results)}件")

                st.markdown("### 🔍 除外理由の内訳（デバッグ用）")
                if fail_reasons:
                    debug_df = pd.DataFrame(
                        fail_reasons.most_common(), columns=["除外理由", "件数"]
                    )
                    st.dataframe(debug_df, use_container_width=True)
                else:
                    st.write("除外なし（全銘柄が結果に含まれています）")

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
                else:
                    st.warning("⚠️ 該当銘柄が0件でした。上の「除外理由の内訳」を確認してください。")

    if st.session_state.analysis_results is not None:
        res_df = st.session_state.analysis_results
        if not res_df.empty and '判定' in res_df.columns:
            st.markdown("---")
            st.header("🔥 買い候補（トレンド抽出）")
            buy_df = res_df[res_df['判定'].str.contains('買い候補', na=False)].copy()

            if not buy_df.empty:
                if 'チャート' in buy_df.columns: buy_df['📊'] = buy_df['チャート']

                # ---- 🏆 AIランキング（上位2〜3銘柄をルールベースでスコアリング） ----
                st.subheader("🏆 AIランキング（買い候補から厳選）")
                st.caption("BT勝率・BT平均損益・RRレシオ・ATR%・反発サイン・高値追い度などをルールベースで採点し、上位のみを表示します。根拠は各カードに表示されます。")
                top_n = st.radio("上位何銘柄を表示するか", [2, 3], horizontal=True, key="rank_top_n")

                ranked = rank_buy_candidates(buy_df)
                for i, item in enumerate(ranked[:top_n]):
                    r = item["元データ"]
                    medal = ["🥇", "🥈", "🥉"][i] if i < 3 else f"{i+1}位"
                    with st.container(border=True):
                        rc1, rc2 = st.columns([1, 3])
                        with rc1:
                            st.markdown(f"### {medal} {item['コード']}")
                            st.markdown(f"**{item['会社名']}**")
                            st.metric("総合スコア", f"{item['スコア']:.1f} 点")
                            st.write(f"現在値: {r.get('現在値','-')}円")
                            st.write(f"損切り: {r.get('損切り価格','-')}円 / 利確: {r.get('利確目標','-')}円")
                            st.write(f"安値比: {r.get('安値比','-')}")
                        with rc2:
                            st.markdown("**スコア内訳**")
                            breakdown_df = pd.DataFrame(
                                list(item["内訳"].items()), columns=["評価項目", "点数"]
                            )
                            st.dataframe(breakdown_df, use_container_width=True, hide_index=True)
                            if r.get('チャート'):
                                st.link_button("🔗 TradingViewで見る", r['チャート'])

                st.markdown("---")

                display_cols = ["コード","会社名","判定","現在値",
                                "BB位置","RSI(14)","MACD","ATR%","売買代金(百万)",
                                "反発サイン","出来高注意","損切り価格","利確目標","RRレシオ",
                                "200日乖離","25日乖離","安値比","BT勝率","BT平均損益","BT取引数","BT最大DD","📊"]
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
                            st.markdown(f"📏 安値比: {row.get('安値比','-')}")
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
            credit_code_cols = [c for c in df_credit.columns if 'コード' in c or 'code' in c.lower()]
            if not credit_code_cols:
                st.error(f"❌ 信用CSVに「コード」列が見つかりません。列名: {list(df_credit.columns)}")
            else:
                df_credit['_code'] = df_credit[credit_code_cols[0]].astype(str).str.strip()
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
                                merge_col = tech_name_col[0]
                                df_merged = df_credit.merge(
                                    df_tech[['_code', merge_col]].rename(columns={merge_col: '_tech_name'}),
                                    on='_code', how='left'
                                )
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

    # 3大設定（GITHUB_TOKEN, GITHUB_REPO）をSecretsから確実に取得
    github_token = st.secrets.get("GITHUB_TOKEN", "")
    github_repo = st.secrets.get("GITHUB_REPO", "syake1/kabu-alert")

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

                UP_THRESHOLD = st.slider(
                    "この上昇率(%)以上の銘柄には自動でチェックを入れる",
                    1, 30, 5, step=1,
                    help="登録時の株価からこの%以上値上がりしている銘柄は「もう上げすぎ」として、下のチェックボックスにあらかじめチェックが入ります。"
                )
                st.caption("チェックを入れた銘柄をまとめて削除できます。登録時の株価が記録されていない古い銘柄は「登録時株価なし」と表示され、自動チェックの対象外です。")

                watch_rows = []
                with st.spinner("現在値を取得中..."):
                    for s in watchlist:
                        code = s.get('code')
                        name = s.get('name', code)
                        added_price = s.get('added_price')
                        cur_price = get_watch_current_price(code)
                        chg_pct = None
                        if cur_price is not None and added_price:
                            try:
                                chg_pct = (cur_price - float(added_price)) / float(added_price) * 100
                            except Exception:
                                chg_pct = None
                        watch_rows.append({
                            "code": code, "name": name,
                            "current_price": cur_price,
                            "added_price": added_price,
                            "chg_pct": chg_pct,
                            "days": s.get('days'), "mode": s.get('mode')
                        })

                to_delete_codes = []
                hc1, hc2, hc3, hc4, hc5 = st.columns([0.6, 2.2, 1.2, 1.2, 1])
                hc1.markdown("**削除**")
                hc2.markdown("**銘柄**")
                hc3.markdown("**現在値**")
                hc4.markdown("**登録時比**")
                hc5.markdown("**日数/モード**")

                for row_idx, row in enumerate(watch_rows):
                    c1, c2, c3, c4, c5 = st.columns([0.6, 2.2, 1.2, 1.2, 1])
                    default_check = row["chg_pct"] is not None and row["chg_pct"] >= UP_THRESHOLD
                    with c1:
                        checked = st.checkbox("削除", value=default_check, key=f"del_chk_{row_idx}_{row['code']}", label_visibility="collapsed")
                        if checked:
                            to_delete_codes.append(row["code"])
                    with c2:
                        st.markdown(f"{row['code']} {row['name']}")
                    with c3:
                        price_txt = f"¥{row['current_price']:,.0f}" if row['current_price'] is not None else "取得失敗"
                        st.markdown(price_txt)
                    with c4:
                        if row["chg_pct"] is not None:
                            arrow = "🔺" if row["chg_pct"] >= 0 else "🔻"
                            st.markdown(f"{arrow} {row['chg_pct']:+.1f}%")
                        else:
                            st.markdown("登録時株価なし")
                    with c5:
                        st.markdown(f"{row['days']}日 / {row['mode']}")

                if st.button(f"🗑 チェックした銘柄を削除（{len(to_delete_codes)}件）",
                             type="secondary", disabled=(len(to_delete_codes) == 0)):
                    if sha is None:
                        st.error("GitHubとの接続に失敗しています")
                    else:
                        watchlist = [s for s in watchlist if s['code'] not in to_delete_codes]
                        new_data = {
                            "watchlist": watchlist,
                            "updated":   datetime.now().strftime('%Y/%m/%d %H:%M'),
                            "count":     len(watchlist)
                        }
                        if update_watchlist_to_github(github_token, github_repo, new_data, sha):
                            st.success(f"✅ {len(to_delete_codes)}銘柄を削除しました")
                            st.rerun()
                        else:
                            st.error("❌ GitHub更新に失敗しました")
        else:
            watchlist = []
            sha       = None
            st.warning("watchlist.jsonの取得に失敗しました")

        st.markdown("---")

        st.markdown("**📝 銘柄を追加：**")
        st.caption("銘柄コードだけ入力すればOKです。銘柄名・登録時株価は自動で取得します。")
        col1, col2 = st.columns([2, 2])
        with col1:
            new_code = st.text_input("銘柄コード", placeholder="例：6507")
        with col2:
            new_mode = st.selectbox("監視モード", ["both（買い＋空売り）", "buy（買いのみ）", "short（空売りのみ）"])
            mode_val = new_mode.split("（")[0]

        if st.button("➕ 監視リストに追加", type="primary"):
            new_code = new_code.strip()
            if not new_code:
                st.error("コードを入力してください")
            elif sha is None:
                st.error("GitHubとの接続に失敗しています")
            else:
                # コードから銘柄名を自動取得
                new_name = new_code
                try:
                    _tk = yf.Ticker(f"{new_code}.T")
                    _info = _tk.info
                    new_name = _info.get('longName') or _info.get('shortName') or new_code
                except Exception:
                    pass

                # 登録時の株価も記録しておく（後で「上がった銘柄」を判定するため）
                new_price = get_watch_current_price(new_code)

                existing_codes = [s['code'] for s in watchlist]
                if new_code in existing_codes:
                    for s in watchlist:
                        if s['code'] == new_code:
                            s['days'] = s.get('days', 0) + 1
                            s['mode'] = mode_val
                            s['name'] = new_name
                            if new_price is not None:
                                s['added_price'] = new_price
                    st.info(f"✅ {new_code} {new_name} の連続日数を更新しました")
                else:
                    watchlist.append({
                        "code": new_code,
                        "name": new_name,
                        "days": 1,
                        "mode": mode_val,
                        "added_price": new_price
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
                            price = row.get('現在値')
                            try:
                                price = float(price) if price is not None else None
                            except Exception:
                                price = None
                            if code in existing_codes:
                                for s in watchlist:
                                    if s['code'] == code:
                                        s['days'] = s.get('days', 0) + 1
                                        if price is not None:
                                            s['added_price'] = price
                            else:
                                watchlist.append({
                                    "code": code,
                                    "name": name,
                                    "days": 1,
                                    "mode": "both",
                                    "added_price": price
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

# ================================================================
# タブ5: 買い銘柄トラッキング（貼り付け→自動で1週間の値動きを追跡）
# ================================================================
with tab_track:
    st.subheader("📋 買い銘柄トラッキング")
    st.caption("その日の買い候補（SBIの株価一覧）を貼り付けると、翌営業日から5営業日分の値動きを自動追跡します。"
               "同じ銘柄が追跡期間中に再度貼り付けられた場合は、重複登録せず元の追跡を継続します。")

    track_token = st.secrets.get("GITHUB_TOKEN", "")
    track_repo  = st.secrets.get("GITHUB_REPO_TRACKING", "syake1/kabujiji")

    if not track_token:
        st.warning("⚠️ Streamlit CloudのSecretsに「GITHUB_TOKEN」を設定してください（監視銘柄登録タブと共通のトークンでOK）")
    else:
        colL, colR = st.columns([2, 1])
        with colL:
            entry_date_input = st.date_input("この銘柄一覧の対象日", value=datetime.now())
        with colR:
            st.write("")

        upload_tab, paste_tab = st.tabs(["📂 CSVアップロード", "📋 貼り付け"])

        raw_df_from_csv = None
        with upload_tab:
            csv_file = st.file_uploader(
                "kabujijiのスキャン結果CSV（買い候補_*.csv）をアップロード",
                type=['csv'], key="track_csv_upload"
            )
            if csv_file is not None:
                for enc in ['utf-8-sig', 'shift-jis', 'utf-8']:
                    try:
                        csv_file.seek(0)
                        raw_df_from_csv = pd.read_csv(csv_file, encoding=enc)
                        break
                    except Exception:
                        continue
                if raw_df_from_csv is None:
                    st.error("❌ CSVの読み込みに失敗しました")
                else:
                    st.success(f"✅ {len(raw_df_from_csv)}銘柄を読み込みました")
                    st.dataframe(raw_df_from_csv, use_container_width=True, height=200)

        with paste_tab:
            pasted = st.text_area(
                "SBIの株価一覧をそのまま貼り付け（銘柄コード／銘柄名／終値／前日比／前日比率 の列を含む表）",
                height=200, key="track_paste"
            )

        if st.button("➕ この一覧をトラッキングに追加", type="primary"):
            if raw_df_from_csv is not None:
                parsed = parse_pasted_stock_table(raw_df_from_csv.to_csv(index=False))
            else:
                parsed = parse_pasted_stock_table(pasted)
            if parsed is None or parsed.empty:
                st.error("❌ 表を認識できませんでした。CSVをアップロードするか、列名（銘柄コード・銘柄名・終値/現在値など）を含めて貼り付けてください")
            else:
                tracking_data, sha = get_tracking_from_github(track_token, track_repo)
                if tracking_data is None:
                    st.error("❌ GitHubからのトラッキングデータ取得に失敗しました")
                else:
                    entries = tracking_data.get("entries", [])
                    entry_date_str = entry_date_input.strftime('%Y-%m-%d')
                    today_ts = pd.Timestamp.now().normalize()

                    added, skipped = 0, 0
                    for _, r in parsed.iterrows():
                        code = str(r['code']).strip()
                        name = str(r['name']).strip()
                        close = float(r['close'])

                        # 重複チェック：同一銘柄がまだ追跡期間中（登録日から5営業日以内）なら追加しない
                        is_active_duplicate = False
                        for e in entries:
                            if e['code'] == code:
                                e_date = pd.Timestamp(e['entry_date'])
                                bdays_end = pd.bdate_range(start=e_date, periods=6)[-1]
                                if today_ts <= bdays_end:
                                    is_active_duplicate = True
                                    break
                        if is_active_duplicate:
                            skipped += 1
                            continue

                        entries.append({
                            "code": code,
                            "name": name,
                            "entry_date": entry_date_str,
                            "entry_price": close,
                            "day_prices": {}
                        })
                        added += 1

                    tracking_data["entries"] = entries
                    if save_tracking_to_github(track_token, track_repo, tracking_data, sha):
                        msg = f"✅ {added}銘柄を追加しました"
                        if skipped:
                            msg += f"（{skipped}銘柄は追跡中のため重複スキップ）"
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error("❌ GitHubへの保存に失敗しました")

        st.markdown("---")
        st.markdown("**📊 トラッキング一覧**")

        tracking_data, sha = get_tracking_from_github(track_token, track_repo)
        if tracking_data is None:
            st.error("❌ トラッキングデータの取得に失敗しました")
        else:
            entries = tracking_data.get("entries", [])
            if not entries:
                st.info("まだ登録された銘柄がありません")
            else:
                with st.spinner("値動きを取得中..."):
                    display_df, updated_any = build_tracking_display(entries)

                if updated_any:
                    tracking_data["entries"] = entries
                    save_tracking_to_github(track_token, track_repo, tracking_data, sha)

                st.dataframe(display_df, use_container_width=True, hide_index=True)

                csv_bytes = display_df.to_csv(index=False).encode('utf-8-sig')
                st.download_button("📥 CSVでダウンロード", csv_bytes, "tracking_result.csv", "text/csv")

                with st.expander("🗑 登録済み銘柄の削除"):
                    del_options = [f"{e['code']} {e['name']}（登録日:{e['entry_date']}）" for e in entries]
                    to_delete = st.multiselect("削除する銘柄を選択", del_options)
                    if st.button("選択した銘柄を削除"):
                        if to_delete:
                            keep = []
                            for e, label in zip(entries, del_options):
                                if label not in to_delete:
                                    keep.append(e)
                            tracking_data["entries"] = keep
                            if save_tracking_to_github(track_token, track_repo, tracking_data, sha):
                                st.success("削除しました")
                                st.rerun()
                            else:
                                st.error("削除に失敗しました")

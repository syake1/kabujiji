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
    rsi_max     = st.slider("RSI 上限（過熱排除）",       50, 70,  55)
    rsi_min     = st.slider("RSI 下限（下落排除）",       25, 45,  35)
    ma200_range = st.slider("200日線乖離 上限 (%)",        1, 30,  20)
    vol_mult    = st.slider("出来高急増 除外倍率（5日比）", 1.5, 5.0, 2.5, step=0.1)
    stop_pct    = st.slider("損切りライン (%)",            1, 10,   4)
    target_pct  = st.slider("利確ライン (%)",              2, 20,   8)

    st.markdown("---")
    st.subheader("💡 押し目条件（戦略説明）")
    st.info(f"""
**【強い銘柄の怖い押し目を拾う】**

1. 株価 > **200日線**（上昇トレンド確認）
2. **25日線が上向き**（中期トレンド維持）
3. BB **センター以下**（理想は下限付近）
4. RSI **{rsi_min}〜{rsi_max}**（売られすぎ圏）
5. 直近**陰線続き**（怖い押し目）
6. 出来高**落ち着き**（急増は除外）
7. **25日線付近**まで調整済み
8. 下ヒゲ・陽線転換などの**反発サイン**

🎯 利確: BB センター付近（+{target_pct}%）
💀 損切: -{stop_pct}%
    """)
    st.markdown("---")
    st.subheader("⏰ 寄り天反発チェックリスト")
    st.warning("""
**10時半〜11時の反発狙い**

✅ 確認してからエントリー：
- [ ] 9〜10時で出来高が急減している
- [ ] 前場安値を2〜3回試して割らない
- [ ] 日経先物が底打ち・戻し始め
- [ ] その銘柄だけ下げ渋っている
- [ ] 小陽線 or 下ヒゲが出た

❌ 見送り条件：
- 個別の悪材料（決算ミス等）がある
- 出来高が増え続けている
- 節目サポートを割り込んだまま
- 日経先物も戻らない

💡 **損切りは前場安値割れ**
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
    """直近n日以上陰線が続いているか"""
    count = 0
    for i in range(1, min(6, len(hist))):
        row = hist.iloc[-i]
        if row['Close'] < row['Open']:
            count += 1
        else:
            break
    return count >= n, count

def check_reversal_sign(hist):
    """
    反発サインを検出
    - 下ヒゲ陽線
    - 包み足（陰線を陽線が包む）
    - 陽線転換（直前陰線 → 本日陽線）
    """
    signs = []
    latest = hist.iloc[-1]
    prev   = hist.iloc[-2]

    body   = abs(latest['Close'] - latest['Open'])
    lower_wick = min(latest['Close'], latest['Open']) - latest['Low']
    upper_wick = latest['High'] - max(latest['Close'], latest['Open'])

    # 下ヒゲ陽線（下ヒゲがボディの1.5倍以上 & 陽線）
    if latest['Close'] > latest['Open'] and body > 0 and lower_wick >= body * 1.5:
        signs.append("下ヒゲ陽線")

    # 包み足（前日陰線 & 本日陽線 & 本日の実体が前日を包む）
    if (prev['Close'] < prev['Open']
            and latest['Close'] > latest['Open']
            and latest['Close'] > prev['Open']
            and latest['Open'] < prev['Close']):
        signs.append("包み足")

    # 陽線転換（前日陰線 → 本日陽線）
    if prev['Close'] < prev['Open'] and latest['Close'] > latest['Open']:
        if "包み足" not in signs:
            signs.append("陽線転換")

    # 下ヒゲ（陰線でも下ヒゲが長い）
    if body > 0 and lower_wick >= body * 2.0 and "下ヒゲ陽線" not in signs:
        signs.append("長い下ヒゲ")

    return signs

def check_ma25_slope(hist, window=5):
    """25日線が上向きかチェック（直近window日の傾き）"""
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
        bb_lo  = r['BB_lower']
        vol_ratio = r['VolRatio']

        if pd.isna(ma200) or pd.isna(rsi) or pd.isna(ma25):
            continue

        # 25日線の傾き（簡易: 5日前より高い）
        ma25_5ago = hist.iloc[i - 5]['MA25'] if i >= 5 else None
        ma25_slope_up = (ma25_5ago is not None) and (not pd.isna(ma25_5ago)) and (ma25 > ma25_5ago)

        # 陰線カウント（直近2日以上）
        bearish_count = 0
        for _j in range(1, 5):
            if i >= _j:
                _r = hist.iloc[i - _j + 1]
                if _r['Close'] < _r['Open']:
                    bearish_count += 1
                else:
                    break

        # 出来高落ち着き（急増を除外）
        vol_calm = vol_ratio < vol_mult

        if not in_trade:
            # 押し目条件
            if (price > ma200                          # 200日線上
                    and ma25_slope_up                  # 25日線上向き
                    and price <= bb_mid                # BBセンター以下
                    and rsi_min <= rsi <= rsi_max      # RSI適正範囲
                    and bearish_count >= 2             # 陰線2日以上
                    and vol_calm                       # 出来高落ち着き
                    and price <= ma25 * 1.03):         # 25日線付近（±3%以内）
                entry_price = hist.iloc[i + 1]['Open']
                in_trade    = True
        else:
            exit_price = None
            # 損切り: -stop_pct%
            if r['Low'] <= entry_price * (1 - stop_pct / 100):
                exit_price = entry_price * (1 - stop_pct / 100)
            # 利確: BBセンター到達 or +target_pct%
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

        # BB位置（0%=下限, 50%=中央, 100%=上限）
        bb_range = bb_upper_val - bb_lo_val
        bb_pos   = ((current_price - bb_lo_val) / bb_range * 100) if bb_range > 0 else 50.0

        # BB下限タッチ判定（直近5日以内にBB下限に触れたか）
        # 「BB下限にいる」ではなく「BB下限から反発中」を検出
        bb_touched_lower  = False
        bb_touch_days_ago = 0
        for _bi in range(1, 6):
            if len(hist) > _bi:
                _row   = hist.iloc[-_bi]
                _bb_lo = float(_row['BB_lower'])
                _low   = float(_row['Low'])
                _close = float(_row['Close'])
                # 安値 or 終値がBB下限の102%以内ならタッチとみなす
                if _low <= _bb_lo * 1.02 or _close <= _bb_lo * 1.02:
                    bb_touched_lower  = True
                    bb_touch_days_ago = _bi
                    break

        # BB下限タッチ後にセンター方向へ戻り始めているか
        bb_rebounding = bb_touched_lower and bb_pos >= 20

        # 25日線の傾き
        ma25_slope = check_ma25_slope(hist)

        # 陰線続き
        is_bearish_cont, bearish_count = check_consecutive_bearish(hist, n=2)

        # 反発サイン
        reversal_signs = check_reversal_sign(hist)

        # 出来高落ち着き（急増除外）
        vol_calm = vol_ratio < vol_mult

        # RSI改善（直近3日でRSIが底打ち）
        rsi_series = hist['RSI'].dropna()
        rsi_improving = False
        if len(rsi_series) >= 4:
            rsi_3ago = float(rsi_series.iloc[-4])
            rsi_2ago = float(rsi_series.iloc[-3])
            rsi_prev = float(rsi_series.iloc[-2])
            rsi_now  = float(rsi_series.iloc[-1])
            # 下げ後に反転（谷を形成）
            if rsi_2ago <= rsi_3ago and rsi_prev <= rsi_2ago and rsi_now > rsi_prev:
                rsi_improving = True
            elif rsi_prev <= rsi_2ago and rsi_now > rsi_prev:
                rsi_improving = True

        # MACD判定（横ばい〜GC接近）
        macd_diff       = macd_val - sig_val
        macd_diff_prev  = float(hist.iloc[-2]['MACD']) - float(hist.iloc[-2]['Signal'])
        macd_narrowing  = abs(macd_diff) < abs(macd_diff_prev)  # 差が縮まっている
        macd_gc_recent  = False
        for _i in range(1, 4):
            if len(hist) > _i:
                _p = hist.iloc[-(_i+1)]
                _c = hist.iloc[-_i]
                if float(_p['MACD']) < float(_p['Signal']) and float(_c['MACD']) >= float(_c['Signal']):
                    macd_gc_recent = True
                    break

        # ================================================================
        # 押し目スコアリング（最大10点）
        # ================================================================
        score   = 0
        reasons = []
        warnings = []

        # --- 必須条件 ---
        # 1. 200日線上（+2点）
        if current_price > ma200:
            score += 2
            reasons.append(f"200日線上(+{diff_pct_200:.1f}%)")
        else:
            warnings.append("200日線下⚠️")

        # 2. 25日線上向き（+2点）
        if ma25_slope:
            score += 2
            reasons.append("25日線上向き")
        else:
            warnings.append("25日線下向き")

        # 3. BB判定（タッチ後反発 or 下限付近 or センター以下）
        if bb_rebounding and bb_touch_days_ago <= 3:
            # 直近1〜3日以内にBB下限タッチ → 今まさに反発中（最高評価）
            score += 3
            reasons.append(f"BB下限タッチ後反発({bb_touch_days_ago}日前タッチ/現在{bb_pos:.0f}%)")
        elif bb_rebounding and bb_touch_days_ago <= 5:
            # 4〜5日前にタッチして戻り始め（高評価）
            score += 2
            reasons.append(f"BB下限反発中({bb_touch_days_ago}日前タッチ/現在{bb_pos:.0f}%)")
        elif bb_pos <= 25:
            # 今もBB下限付近にいる（標準評価）
            score += 2
            reasons.append(f"BB下限付近({bb_pos:.0f}%)")
        elif bb_pos <= 50:
            # BBセンター以下（最低評価）
            score += 1
            reasons.append(f"BBセンター以下({bb_pos:.0f}%)")
        else:
            warnings.append(f"BB上部({bb_pos:.0f}%)")

        # 4. RSI適正範囲（+1点）/ RSI改善中（+1点追加）
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

        # 5. 陰線続き（+1点）
        if is_bearish_cont:
            score += 1
            reasons.append(f"陰線{bearish_count}日続き")

        # 6. 出来高落ち着き（+1点）
        if vol_calm:
            score += 1
            reasons.append(f"出来高落ち着き({vol_ratio:.1f}x)")
        else:
            warnings.append(f"出来高急増({vol_ratio:.1f}x)⚠️")

        # 7. 25日線付近まで調整（+1点）
        if abs(diff_pct_25) <= 5.0:
            score += 1
            reasons.append(f"25日線付近({diff_pct_25:+.1f}%)")

        # 8. 反発サイン（+1点）
        if reversal_signs:
            score += 1
            reasons.append(" ".join(reversal_signs))

        # 9. MACDがGC接近（ボーナス +1点、最大10点にクリップ）
        if macd_gc_recent:
            score = min(score + 1, 10)
            reasons.append("MACD-GC直近")
        elif macd_narrowing and macd_val < sig_val:
            score = min(score + 1, 10)
            reasons.append("MACD収束中")

        # --- 除外ペナルティ ---
        # 高値ブレイク直後（BB上部 & 出来高急増）
        if bb_pos > 80 and not vol_calm:
            score = max(score - 3, 0)
            warnings.append("高値圏過熱⛔")

        # ================================================================
        # 判定
        # ================================================================
        # 必須条件チェック
        must_ok = (current_price > ma200) and ma25_slope

        if not must_ok:
            status = "⛔ 除外（弱い銘柄）"
        elif score >= 8:
            status = "🔥 買い候補"
        elif score >= 6:
            status = "👀 監視（押し目形成中）"
        elif score >= 4:
            status = "⏳ 様子見"
        else:
            status = "➖ 対象外"

        # 価格目標
        stop_price   = round(current_price * (1 - stop_pct  / 100), 1)
        # 利確はBBセンターかtarget_pctの高い方
        target_price = round(max(current_price * (1 + target_pct / 100), bb_mid_val), 1)
        rr_ratio     = round((target_price - current_price) / (current_price - stop_price), 1) \
                       if current_price > stop_price else 0.0

        # MACD表示ラベル
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
# メイン画面
# ================================================================
st.title("🚀 アンチグラビティ・コア Pro+")
st.caption("押し目反発特化版 ｜ 強い銘柄の怖い押し目を拾い、平均回帰を狙う")

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
                if not target_model_name:
                    st.error("使えるモデルが見つかりません")
                else:
                    model = genai.GenerativeModel(target_model_name)
                    name  = target_model_name.replace('models/', '')
                    with st.spinner(f"AI ({name}) が分析中..."):
                        prompt = (
                            "あなたは日本株スイングトレードの専門家です。\n"
                            "戦略は「強い銘柄の押し目反発（平均回帰）」です。\n"
                            "以下のニュースを読み、\n"
                            "①相場全体への影響（強気/中立/弱気）\n"
                            "②一時的売りで押し目が生じやすいセクター・銘柄\n"
                            "③地合い悪化時に下げ渋る強い銘柄の特徴\n"
                            "④スイング押し目買いの観点で注目すべきポイント\n"
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
                    with st.expander(f"❌ 取得失敗 {len(errors)}件（クリックで詳細）"):
                        for e in errors[:20]:
                            st.text(e)
                        if len(errors) > 20:
                            st.text(f"... 他 {len(errors)-20}件")

                if results:
                    good = [r for r in results if '判定' in r]
                    st.session_state.analysis_results = pd.DataFrame(good) if good else None
                    # ---- スマホ用にJSONへ自動保存 ----
                    import json, os
                    buy_list = [r for r in good if r.get('判定') == '🔥 買い候補']
                    watch_list = [r for r in good if r.get('判定') == '👀 監視（押し目形成中）']
                    save_data = {
                        'updated': datetime.now().strftime('%Y/%m/%d %H:%M'),
                        'buy':   buy_list,
                        'watch': watch_list,
                        'total': len(good),
                    }
                    os.makedirs('data', exist_ok=True)
                    with open('data/scan_result.json', 'w', encoding='utf-8') as f:
                        json.dump(save_data, f, ensure_ascii=False, indent=2)
                    st.success("📱 スマホ用データを自動保存しました")
                else:
                    if errors:
                        st.error("⚠️ 全銘柄の取得に失敗しました。時間をおいて再実行してください。")
                    else:
                        st.warning("⚠️ 解析できた銘柄が0件でした（データ不足）。")
                    st.session_state.analysis_results = None

# ================================================================
# 結果表示
# ================================================================
if st.session_state.analysis_results is not None:
    res_df = st.session_state.analysis_results

    if res_df.empty or '判定' not in res_df.columns:
        st.warning("⚠️ 表示できる解析結果がありません。再度スキャンしてください。")
        st.stop()

    st.markdown("---")

    # ======== 買い候補 ========
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

        # --- %表示を数値に修正（ProgressColumnが%文字列を誤表示するため） ---
        def pct_to_float(col):
            """'12.3%' → 12.3 に変換"""
            return col.apply(lambda x: float(str(x).replace('%','').replace('+','')) if x != '-' else 0.0)

        buy_display = buy_only.copy()
        for c in ['200日乖離', '25日乖離', 'BT勝率', 'BT平均損益', 'BT最大DD']:
            if c in buy_display.columns:
                buy_display[c] = pct_to_float(buy_display[c])

        # 表示列の並び順
        display_cols = [
            "コード", "会社名", "スコア",
            "現在値", "200日乖離", "25日乖離",
            "RSI(14)", "MACD", "BB位置",
            "陰線日数", "出来高倍率", "反発サイン",
            "損切り価格", "利確目標", "RRレシオ",
            "根拠", "注意点", "チャート",
            "BT勝率", "BT平均損益", "BT取引数", "BT最大DD"
        ]
        disp = [c for c in display_cols if c in buy_display.columns]

        st.dataframe(
            buy_display[disp],
            use_container_width=True,
            column_config={
                "チャート":    st.column_config.LinkColumn("チャート"),
                "損切り価格":  st.column_config.NumberColumn("損切り💀", format="%.1f"),
                "利確目標":    st.column_config.NumberColumn("利確🎯",   format="%.1f"),
                "スコア":      st.column_config.ProgressColumn("スコア", min_value=0, max_value=10),
                "200日乖離":   st.column_config.NumberColumn("200日乖離%", format="%.2f"),
                "25日乖離":    st.column_config.NumberColumn("25日乖離%",  format="%.2f"),
                "BT勝率":      st.column_config.NumberColumn("BT勝率%",   format="%.1f"),
                "BT平均損益":  st.column_config.NumberColumn("BT平均損益%", format="%.2f"),
                "BT最大DD":    st.column_config.NumberColumn("BT最大DD%",  format="%.2f"),
                "陰線日数":    st.column_config.NumberColumn("陰線日数📉"),
                "反発サイン":  st.column_config.TextColumn("反発サイン✨"),
            }
        )

        # ========== CSVダウンロード ==========
        st.markdown("")
        dl_cols = [c for c in display_cols if c in buy_only.columns and c != "チャート"]
        csv_data = buy_only[dl_cols].to_csv(index=False, encoding='utf-8-sig')
        now_str  = datetime.now().strftime('%Y%m%d_%H%M')
        st.download_button(
            label     = f"📥 買い候補CSVをダウンロード（{len(buy_only)}銘柄）",
            data      = csv_data,
            file_name = f"買い候補_{now_str}.csv",
            mime      = "text/csv",
            use_container_width = True,
        )

        st.subheader("📈 バックテスト結果（買い候補）")
        bt_cols = ['コード', '会社名', 'BT勝率', 'BT平均損益', 'BT取引数', 'BT最大DD', 'RRレシオ', '根拠']
        st.dataframe(buy_only[[c for c in bt_cols if c in buy_only.columns]], use_container_width=True)

        # ========== インラインチャート ==========
        st.markdown("---")
        st.subheader("📊 買い候補チャート（ローソク足）")
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            HAS_MPL = True
        except ImportError:
            HAS_MPL = False

        def plot_stock_chart(code, name, stop_price, target_price):
            try:
                tk   = yf.Ticker(f"{code}.T")
                hist = tk.history(period="6mo")
                if len(hist) < 30:
                    return None
                hist = hist.reset_index()
                hist['MA25']   = hist['Close'].rolling(25).mean()
                hist['MA200']  = hist['Close'].rolling(200).mean()
                bb_up, bb_mid, bb_lo = calculate_bb(hist)
                hist['BB_upper'] = bb_up
                hist['BB_mid']   = bb_mid
                hist['BB_lower'] = bb_lo
                hist['VolMA5'] = hist['Volume'].rolling(5).mean()

                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(12, 7),
                    gridspec_kw={'height_ratios': [3, 1]},
                    sharex=True
                )
                fig.patch.set_facecolor('#0E1117')
                ax1.set_facecolor('#0E1117')
                ax2.set_facecolor('#0E1117')

                # ローソク足
                for i, row in hist.iterrows():
                    color = '#FF4B4B' if row['Close'] >= row['Open'] else '#1F77B4'
                    ax1.plot([i, i], [row['Low'], row['High']], color=color, linewidth=0.8)
                    ax1.bar(i, abs(row['Close'] - row['Open']),
                            bottom=min(row['Open'], row['Close']),
                            color=color, width=0.6, alpha=0.9)

                # 移動平均
                ax1.plot(range(len(hist)), hist['MA25'],  color='orange',  linewidth=1.2, label='MA25')
                ax1.plot(range(len(hist)), hist['MA200'], color='#00BFFF', linewidth=1.0, label='MA200', linestyle='--')

                # ボリンジャーバンド
                ax1.fill_between(range(len(hist)), hist['BB_upper'], hist['BB_lower'],
                                 alpha=0.08, color='gray')
                ax1.plot(range(len(hist)), hist['BB_mid'],   color='#AAAAAA', linewidth=0.8, linestyle=':', label='BB mid')
                ax1.plot(range(len(hist)), hist['BB_upper'], color='#888888', linewidth=0.6, linestyle='-')
                ax1.plot(range(len(hist)), hist['BB_lower'], color='#888888', linewidth=0.6, linestyle='-')

                # 損切り・利確ライン
                ax1.axhline(y=stop_price,   color='red',    linestyle='--', linewidth=1.2,
                            label=f'損切 {stop_price}円')
                ax1.axhline(y=target_price, color='#00FF7F', linestyle='--', linewidth=1.2,
                            label=f'利確 {target_price}円')
                ax1.text(len(hist)-1, stop_price,   f' 損切 {stop_price}',
                         color='red',     fontsize=8, va='center')
                ax1.text(len(hist)-1, target_price, f' 利確 {target_price}',
                         color='#00FF7F', fontsize=8, va='center')

                ax1.set_title(f'{code}  {name}', color='white', fontsize=13, pad=8)
                ax1.tick_params(colors='#AAAAAA')
                ax1.set_ylabel('株価 (円)', color='#AAAAAA')
                for spine in ax1.spines.values():
                    spine.set_edgecolor('#333333')
                ax1.legend(loc='upper left', fontsize=8,
                           facecolor='#1E1E1E', labelcolor='white', framealpha=0.7)
                ax1.grid(axis='y', color='#222222', linewidth=0.5)

                # 出来高
                vol_colors = ['#FF4B4B' if c >= o else '#1F77B4'
                              for c, o in zip(hist['Close'], hist['Open'])]
                ax2.bar(range(len(hist)), hist['Volume'], color=vol_colors, alpha=0.7, width=0.6)
                ax2.plot(range(len(hist)), hist['VolMA5'], color='yellow', linewidth=1, label='Vol MA5')
                ax2.set_ylabel('出来高', color='#AAAAAA')
                ax2.tick_params(colors='#AAAAAA')
                for spine in ax2.spines.values():
                    spine.set_edgecolor('#333333')
                ax2.grid(axis='y', color='#222222', linewidth=0.5)

                tick_step = max(1, len(hist) // 8)
                ticks = list(range(0, len(hist), tick_step))
                ax2.set_xticks(ticks)
                ax2.set_xticklabels(
                    [str(hist['Date'].iloc[t])[:10] for t in ticks],
                    rotation=30, ha='right', color='#AAAAAA', fontsize=7
                )

                plt.tight_layout(h_pad=0.5)
                return fig
            except Exception as e:
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
                            tk2 = yf.Ticker(f"{code}.T")
                            h2  = tk2.history(period="6mo").reset_index()
                            if len(h2) > 0:
                                h2['MA25']  = h2['Close'].rolling(25).mean()
                                h2['MA200'] = h2['Close'].rolling(200).mean()
                                h2 = h2.set_index('Date')
                                st.line_chart(h2[['Close','MA25','MA200']], use_container_width=True)
                                st.caption("※ matplotlib未インストールのため簡易表示")
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
                    st.markdown(f"陰線日数: {row_data['陰線日数']}日")
                    st.markdown(f"反発サイン: {row_data['反発サイン']}")
                    st.markdown(f"BT勝率: {row_data['BT勝率']}")
                    st.link_button("🔗 TradingViewで開く", row_data['チャート'])

        if discord_webhook:
            msg = "【🔥押し目買いサイン点灯】\n" + "\n".join(
                [f"・{r['コード']} {r['会社名']} スコア{r['スコア']} BT勝率{r['BT勝率']} 反発:{r['反発サイン']}"
                 for _, r in buy_only.iterrows()])
            requests.post(discord_webhook, json={"content": msg})
            st.balloons()
    else:
        st.info("現在、押し目買い条件を満たす銘柄はありません。サイドバーの条件を調整してみてください。")

    # 監視銘柄
    st.markdown("---")
    st.subheader("👀 監視銘柄（押し目形成中）")
    watch = res_df[res_df['判定'] == "👀 監視（押し目形成中）"].sort_values('スコア', ascending=False)
    if not watch.empty:
        st.dataframe(watch, use_container_width=True,
                     column_config={"チャート": st.column_config.LinkColumn("チャート")})
    else:
        st.write("監視銘柄もありません。")

    # 統計
    st.markdown("---")
    st.subheader("📊 スキャン統計")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("解析銘柄数",           len(res_df))
    c2.metric("🔥 買い候補",          len(res_df[res_df['判定'] == "🔥 買い候補"]))
    c3.metric("👀 監視",              len(res_df[res_df['判定'] == "👀 監視（押し目形成中）"]))
    c4.metric("⏳ 様子見",            len(res_df[res_df['判定'] == "⏳ 様子見"]))
    c5.metric("⛔ 除外（弱い銘柄）",  len(res_df[res_df['判定'] == "⛔ 除外（弱い銘柄）"]))

    # 全銘柄
    st.markdown("---")
    st.subheader("📋 全銘柄一覧")
    st.dataframe(res_df, use_container_width=True,
                 column_config={"チャート": st.column_config.LinkColumn("チャート")})

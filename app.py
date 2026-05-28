import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
import google.generativeai as genai
from datetime import datetime

# --- 初期設定 ---
st.set_page_config(page_title="アンチグラビティ・コア Pro+", layout="wide")

if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# --- サイドバー ---
with st.sidebar:
    st.header("⚙️ システム設定")
    gemini_key      = st.text_input("Gemini API Key", type="password")
    discord_webhook = st.text_input("Discord Webhook URL", type="password")

    st.markdown("---")
    st.subheader("🎯 スイング条件設定")
    rsi_max     = st.slider("RSI 上限（過熱排除）",      50, 75,  65)
    rsi_min     = st.slider("RSI 下限（下落排除）",      30, 55,  45)
    ma200_range = st.slider("200日線乖離 上限 (%)",       1, 15,   7)
    vol_mult    = st.slider("出来高急増 倍率（5日平均比）", 1.0, 5.0, 1.5, step=0.1)
    stop_pct    = st.slider("損切りライン (%)",           1, 10,   4)
    target_pct  = st.slider("利確ライン (%)",             2, 20,  10)

    st.markdown("---")
    st.subheader("💡 買い候補の条件")
    st.info(f"""
1. 株価 > **200日線**
2. 200日線乖離 < **{ma200_range}%**
3. RSI **{rsi_min}〜{rsi_max}**
4. MACD **上向き or GC**
5. 出来高 **{vol_mult}倍以上**急増
6. ボリンジャーバンド **中央〜上部**
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

# ================================================================
# バックテスト
# ================================================================
def backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, ma200_range, vol_mult):
    hist     = hist.copy().reset_index()
    trades   = []
    in_trade = False
    entry_price = 0.0

    for i in range(201, len(hist) - 1):
        r         = hist.iloc[i]
        prev      = hist.iloc[i - 1]
        price     = r['Close']
        ma200     = r['MA200']
        rsi       = r['RSI']
        macd_val  = r['MACD']
        sig_val   = r['Signal']
        vol_ratio = r['VolRatio']

        if pd.isna(ma200) or pd.isna(rsi):
            continue
        diff_pct = (price - ma200) / ma200 * 100

        if not in_trade:
            # 直近3日以内のGC
            gc = False
            for _gi in range(1, 4):
                if i >= _gi:
                    _gp = hist.iloc[i - _gi]
                    _gc = hist.iloc[i - _gi + 1]
                    if _gp['MACD'] < _gp['Signal'] and _gc['MACD'] >= _gc['Signal']:
                        gc = True
                        break
            if (price > ma200
                    and 0 <= diff_pct < ma200_range
                    and rsi_min <= rsi <= rsi_max
                    and gc
                    and vol_ratio >= vol_mult):
                entry_price = hist.iloc[i + 1]['Open']
                in_trade    = True
        else:
            exit_price = None
            if r['Low']  <= entry_price * (1 - stop_pct  / 100):
                exit_price = entry_price * (1 - stop_pct  / 100)
            elif r['High'] >= entry_price * (1 + target_pct / 100):
                exit_price = entry_price * (1 + target_pct / 100)
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
# メイン解析
# ================================================================
def analyze_stock(ticker_code, company_name,
                  stop_pct, target_pct, rsi_min, rsi_max, ma200_range, vol_mult):
    import time
    last_err = ""
    try:
        # リトライ付きデータ取得（最大3回）
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
        hist['RSI']      = calculate_rsi(hist)
        macd, sig        = calculate_macd(hist)
        hist['MACD']     = macd
        hist['Signal']   = sig
        bb_up, _, bb_lo  = calculate_bb(hist)
        hist['BB_upper'] = bb_up
        hist['BB_lower'] = bb_lo
        hist['VolMA5']   = hist['Volume'].rolling(5).mean()
        hist['VolRatio'] = hist['Volume'] / hist['VolMA5']

        latest = hist.iloc[-1]
        prev   = hist.iloc[-2]

        current_price = float(latest['Close'])
        ma200         = float(latest['MA200'])
        rsi           = float(latest['RSI'])
        macd_val      = float(latest['MACD'])
        sig_val       = float(latest['Signal'])
        vol_ratio     = float(latest['VolRatio'])
        bb_pos        = (current_price - float(latest['BB_lower'])) / \
                        (float(latest['BB_upper']) - float(latest['BB_lower'])) * 100

        if pd.isna(ma200):
            return None
        diff_pct = (current_price - ma200) / ma200 * 100

        # 直近3日以内にGCが発生していればOK
        gc_today = False
        for _i in range(1, 4):
            if len(hist) > _i:
                _p = hist.iloc[-(_i+1)]
                _c = hist.iloc[-_i]
                if float(_p['MACD']) < float(_p['Signal']) and float(_c['MACD']) >= float(_c['Signal']):
                    gc_today = True
                    break
        macd_label = "🟢 GC3日内" if gc_today else ("↑上" if macd_val > sig_val else "↓下")

        # スコアリング
        score   = 0
        reasons = []
        if current_price > ma200 and 0 <= diff_pct < ma200_range:
            score += 2; reasons.append("200日線上")
        if rsi_min <= rsi <= rsi_max:
            score += 2; reasons.append(f"RSI適正({rsi:.0f})")
        if macd_val > sig_val:
            score += 2; reasons.append("MACD上向き")
        if gc_today:
            score += 1; reasons.append("GC本日")
        if vol_ratio >= vol_mult:
            score += 2; reasons.append(f"出来高急増({vol_ratio:.1f}x)")
        if 40 <= bb_pos <= 80:
            score += 1; reasons.append("BB中央帯")

        if score >= 8:
            status = "🔥 買い候補"
        elif score >= 5:
            status = "👀 監視"
        elif current_price < ma200:
            status = "⏳ 潜伏中"
        else:
            status = "➖ 対象外"

        stop_price   = round(current_price * (1 - stop_pct  / 100), 1)
        target_price = round(current_price * (1 + target_pct / 100), 1)
        rr_ratio     = round(target_pct / stop_pct, 1)

        bt = backtest(hist, stop_pct, target_pct, rsi_min, rsi_max, ma200_range, vol_mult)

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
            "コード":     ticker_code,
            "会社名":     company_name,
            "判定":       status,
            "スコア":     score,
            "現在値":     round(current_price, 1),
            "200日乖離":  f"{round(diff_pct, 2)}%",
            "RSI(14)":    round(rsi, 1),
            "MACD":       macd_label,
            "BB位置":     f"{round(bb_pos, 0)}%",
            "出来高倍率": f"{vol_ratio:.1f}x",
            "損切り価格": stop_price,
            "利確価格":   target_price,
            "RRレシオ":   f"1:{rr_ratio}",
            "配当利回り": f"{round(div_yield * 100, 2)}%",
            "次期決算":   earnings_date,
            "PER":        per,
            "根拠":       " / ".join(reasons) if reasons else "-",
            "チャート":   f"https://jp.tradingview.com/chart/?symbol=TSE:{ticker_code}",
            "BT勝率":     bt["勝率"]     if bt else "-",
            "BT平均損益": bt["平均損益"] if bt else "-",
            "BT取引数":   bt["取引回数"] if bt else 0,
            "BT最大DD":   bt["最大DD"]   if bt else "-",
        }
    except:
        return None

# ================================================================
# メイン画面
# ================================================================
st.title("🚀 アンチグラビティ・コア Pro+")
st.caption("スイング特化版 ｜ MACD・BB・出来高・損切り利確・バックテスト搭載")

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
                            "あなたはプロのスイングトレーダーです。\n"
                            "以下のニュースを読み、\n"
                            "①相場への影響（強気/中立/弱気）\n"
                            "②恩恵を受けるセクター・銘柄\n"
                            "③スイングトレードとして注目すべきポイント\n"
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

        # SBIのCSV形式に対応（複数パターン）
        # コード列: 「コード」含む → なければ「銘柄」完全一致
        c_col = [c for c in df.columns if 'コード' in c]
        if not c_col:
            c_col = [c for c in df.columns if c.strip() == '銘柄']
        # 銘柄名列: 「銘柄名」含む → なければ「銘柄.1」
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
                    if not good:
                        st.warning("⚠️ データ取得はできましたが判定列がありません。")
                else:
                    if errors:
                        st.error(f"⚠️ 全{len(errors)}銘柄の取得に失敗しました。\n主な原因: Yahoo Financeへの接続制限\n→ 少し時間をおいて再実行してください。")
                    else:
                        st.warning("⚠️ 解析できた銘柄が0件でした（データ不足）。")
                    st.session_state.analysis_results = None

# ================================================================
# 結果表示
# ================================================================
if st.session_state.analysis_results is not None:
    res_df = st.session_state.analysis_results

    # ✅ 修正: 空DataFrameや'判定'列なしに対するガード
    if res_df.empty or '判定' not in res_df.columns:
        st.warning("⚠️ 表示できる解析結果がありません。再度スキャンしてください。")
        st.stop()

    st.markdown("---")

    # 買い候補
    st.header("🔥 【厳選】買い候補")
    buy_only = res_df[res_df['判定'] == "🔥 買い候補"].copy()

    def parse_pct(x):
        try:
            return float(str(x).replace('%', ''))
        except:
            return 0.0

    if not buy_only.empty:
        buy_only['_win'] = buy_only['BT勝率'].apply(parse_pct)
        buy_only = buy_only.sort_values(['_win', 'スコア'], ascending=False).drop(columns=['_win'])
        st.dataframe(
            buy_only,
            use_container_width=True,
            column_config={
                "チャート":   st.column_config.LinkColumn("チャート"),
                "損切り価格": st.column_config.NumberColumn("損切り💀", format="%.1f"),
                "利確価格":   st.column_config.NumberColumn("利確🎯",   format="%.1f"),
                "スコア":     st.column_config.ProgressColumn("スコア", min_value=0, max_value=10),
            }
        )

        st.subheader("📈 バックテスト結果（買い候補）")
        bt_cols = ['コード', '会社名', 'BT勝率', 'BT平均損益', 'BT取引数', 'BT最大DD', 'RRレシオ', '根拠']
        st.dataframe(buy_only[[c for c in bt_cols if c in buy_only.columns]], use_container_width=True)

        # ========== インラインチャート ==========
        st.markdown("---")
        st.subheader("📊 買い候補チャート（ローソク足）")
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        def plot_stock_chart(code, name, stop_price, target_price):
            try:
                tk   = yf.Ticker(f"{code}.T")
                hist = tk.history(period="6mo")
                if len(hist) < 30:
                    return None
                hist = hist.reset_index()
                hist['MA25']  = hist['Close'].rolling(25).mean()
                hist['MA75']  = hist['Close'].rolling(75).mean()
                hist['VolMA5'] = hist['Volume'].rolling(5).mean()

                fig = make_subplots(
                    rows=2, cols=1,
                    shared_xaxes=True,
                    row_heights=[0.7, 0.3],
                    vertical_spacing=0.05
                )

                # ローソク足
                fig.add_trace(go.Candlestick(
                    x=hist['Date'],
                    open=hist['Open'], high=hist['High'],
                    low=hist['Low'],   close=hist['Close'],
                    name="株価",
                    increasing_line_color='#FF4B4B',
                    decreasing_line_color='#1F77B4',
                ), row=1, col=1)

                # 移動平均線
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MA25'],
                    line=dict(color='orange', width=1.2), name="MA25"), row=1, col=1)
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist['MA75'],
                    line=dict(color='purple', width=1.2), name="MA75"), row=1, col=1)

                # 損切り・利確ライン
                fig.add_hline(y=stop_price,   line_dash="dash", line_color="red",
                              annotation_text=f"損切 {stop_price}", row=1, col=1)
                fig.add_hline(y=target_price, line_dash="dash", line_color="green",
                              annotation_text=f"利確 {target_price}", row=1, col=1)

                # 出来高
                colors = ['#FF4B4B' if c >= o else '#1F77B4'
                          for c, o in zip(hist['Close'], hist['Open'])]
                fig.add_trace(go.Bar(
                    x=hist['Date'], y=hist['Volume'],
                    marker_color=colors, name="出来高", opacity=0.7
                ), row=2, col=1)
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist['VolMA5'],
                    line=dict(color='yellow', width=1), name="出来高MA5"), row=2, col=1)

                fig.update_layout(
                    title=f"{code} {name}",
                    xaxis_rangeslider_visible=False,
                    template="plotly_dark",
                    height=500,
                    margin=dict(l=40, r=40, t=50, b=20),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig.update_yaxes(title_text="株価(円)", row=1, col=1)
                fig.update_yaxes(title_text="出来高",   row=2, col=1)
                return fig
            except Exception as e:
                return None

        chart_codes = buy_only[['コード', '会社名', '損切り価格', '利確価格']].values.tolist()
        cols_per_row = 1
        for i, (code, name, stop_p, tgt_p) in enumerate(chart_codes):
            with st.expander(f"📈 {code} {name}　損切:{stop_p}円 / 利確:{tgt_p}円", expanded=(i == 0)):
                col_left, col_right = st.columns([4, 1])
                with col_left:
                    with st.spinner(f"{code} チャート読み込み中..."):
                        fig = plot_stock_chart(code, name, float(stop_p), float(tgt_p))
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.warning("チャートデータを取得できませんでした")
                with col_right:
                    st.markdown(f"**{name}**")
                    st.markdown(f"🔴 損切: **{stop_p}円**")
                    st.markdown(f"🟢 利確: **{tgt_p}円**")
                    row = buy_only[buy_only['コード'] == code].iloc[0]
                    st.markdown(f"📊 スコア: **{row['スコア']}**")
                    st.markdown(f"RSI: {row['RSI(14)']}")
                    st.markdown(f"MACD: {row['MACD']}")
                    st.markdown(f"出来高倍率: {row['出来高倍率']}")
                    st.markdown(f"BT勝率: {row['BT勝率']}")
                    st.link_button("🔗 TradingViewで開く", row['チャート'])

        if discord_webhook:
            msg = "【🔥買いサイン点灯】\n" + "\n".join(
                [f"・{r['コード']} {r['会社名']} スコア{r['スコア']} BT勝率{r['BT勝率']}"
                 for _, r in buy_only.iterrows()])
            requests.post(discord_webhook, json={"content": msg})
            st.balloons()
    else:
        st.info("現在、条件を満たす買い候補はありません。サイドバーの条件を緩めてみてください。")

    # 監視銘柄
    st.markdown("---")
    st.subheader("👀 監視銘柄（条件もう一歩）")
    watch = res_df[res_df['判定'] == "👀 監視"].sort_values('スコア', ascending=False)
    if not watch.empty:
        st.dataframe(watch, use_container_width=True,
                     column_config={"チャート": st.column_config.LinkColumn("チャート")})
    else:
        st.write("監視銘柄もありません。")

    # 統計
    st.markdown("---")
    st.subheader("📊 スキャン統計")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("解析銘柄数",    len(res_df))
    c2.metric("🔥 買い候補",  len(res_df[res_df['判定'] == "🔥 買い候補"]))
    c3.metric("👀 監視",       len(res_df[res_df['判定'] == "👀 監視"]))
    c4.metric("⏳ 潜伏中",     len(res_df[res_df['判定'] == "⏳ 潜伏中"]))

    # 全銘柄
    st.markdown("---")
    st.subheader("📋 全銘柄一覧")
    st.dataframe(res_df, use_container_width=True,
                 column_config={"チャート": st.column_config.LinkColumn("チャート")})

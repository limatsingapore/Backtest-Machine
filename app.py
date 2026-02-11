import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import io
import xlsxwriter # 엑셀 다운로드를 위해 필요 (설치 확인)

# --- [페이지 설정] ---
st.set_page_config(page_title="Pension Stock Backtester Pro", layout="wide", page_icon="📈")

# --- [스타일링] ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] { background-color: #262730; padding: 15px; border-radius: 10px; border: 1px solid #444; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# [Helper Functions]
# ==============================================================================

@st.cache_data(ttl=3600*24)
def get_stock_listing():
    """KRX 전체 종목(주식+ETF) 리스트를 가져와 결합 및 캐싱"""
    try:
        # 1. 일반 주식 (KOSPI, KOSDAQ, KONEX)
        df_krx = fdr.StockListing('KRX')
        if 'Symbol' in df_krx.columns:
             df_krx.rename(columns={'Symbol': 'Code'}, inplace=True)
        df_krx = df_krx[['Code', 'Name']]

        # 2. 한국 ETF
        df_etf = fdr.StockListing('ETF/KR')
        if 'Symbol' in df_etf.columns:
             df_etf.rename(columns={'Symbol': 'Code'}, inplace=True)
        df_etf = df_etf[['Code', 'Name']]

        # 3. 리스트 병합 및 중복 제거
        df_combined = pd.concat([df_krx, df_etf], ignore_index=True)
        # 혹시 모를 중복 코드 제거 (코드가 같으면 이름을 덮어씀)
        df_combined.drop_duplicates(subset=['Code'], keep='first', inplace=True)

        return dict(zip(df_combined['Code'], df_combined['Name']))
    except Exception as e:
        # 에러 발생 시 빈 딕셔너리 반환 (로그는 서버 콘솔에 찍힘)
        print(f"Error fetching stock listing: {e}")
        return {}

def get_stock_name(ticker, listing_dict):
    """종목명 찾기 (1차: 합본 리스트 -> 2차: YFinance -> 3차: Ticker)"""
    # 1. FDR 합본 리스트에서 검색 (가장 정확함)
    if ticker in listing_dict:
        return listing_dict[ticker]
    
    # 2. 실패 시 YFinance 시도 (해외 종목 등)
    try:
        ticker_yf = yf.Ticker(f"{ticker}.KS")
        name = ticker_yf.info.get('shortName')
        if not name:
             ticker_yf = yf.Ticker(f"{ticker}.KQ")
             name = ticker_yf.info.get('shortName')
        if name: return name
    except:
        pass
        
    # 3. 모두 실패하면 티커 반환
    return ticker

@st.cache_data(ttl=3600*12)
def get_stock_data(ticker, start_date, end_date):
    """주가(FDR) + 배당(YFinance) 데이터 병합"""
    # 1. 주가 데이터
    df_price = fdr.DataReader(ticker, start_date, end_date)
    if df_price.empty: return None
    df_price = df_price[['Close']]
    
    # 2. 배당 데이터
    dividends = pd.Series(dtype=float)
    suffixes = ['.KS', '.KQ']
    for suffix in suffixes:
        try:
            yf_obj = yf.Ticker(f"{ticker}{suffix}")
            div_temp = yf_obj.dividends
            if not div_temp.empty:
                div_temp.index = div_temp.index.tz_localize(None)
                dividends = div_temp[(div_temp.index >= pd.to_datetime(start_date)) & 
                                     (div_temp.index <= pd.to_datetime(end_date))]
                break
        except: continue

    # 3. 병합
    df = df_price.copy()
    df['Dividend'] = 0.0
    if not dividends.empty:
        common_dates = df.index.intersection(dividends.index)
        if not common_dates.empty:
            df.loc[common_dates, 'Dividend'] = dividends.loc[common_dates]
    
    return df

def run_simulation(df, initial_capital, payment_amt, mode, interval="매월"):
    """[Core Logic] 시뮬레이션 엔진"""
    df = df.copy()
    
    # 변수 초기화
    shares = 0.0
    principal = 0.0
    share_history = []
    principal_history = []
    
    # 거치식 초기 매수
    if mode == "거치식":
        price = df.iloc[0]['Close']
        if price > 0:
            shares = initial_capital / price
            principal = initial_capital
    
    prev_month = df.index[0].month
    prev_year = df.index[0].year
    is_first_period = True 

    for date, row in df.iterrows():
        price = row['Close']
        div = row['Dividend']
        curr_year = date.year
        curr_month = date.month
        
        # 1. 적립식 매수 로직
        if mode == "적립식" and price > 0:
            should_buy = False
            if is_first_period:
                should_buy = True
                is_first_period = False 
            else:
                if interval == "매월":
                    if curr_month != prev_month: should_buy = True
                elif interval == "매년":
                    if curr_year != prev_year: should_buy = True

            if should_buy:
                shares += payment_amt / price
                principal += payment_amt
                prev_month = curr_month
                prev_year = curr_year
        
        # 2. 배당 재투자
        if div > 0 and shares > 0 and price > 0:
            dividend_amount = shares * div
            shares += dividend_amount / price
            
        share_history.append(shares)
        principal_history.append(principal)
        
    df['Shares'] = share_history
    df['Principal'] = principal_history
    df['Total_Value'] = df['Shares'] * df['Close']
    
    return df

def calculate_monthly_returns(df):
    """월별 수익률 히트맵용 데이터 생성"""
    df_m = df['Total_Value'].resample('ME').last()
    df_ret = df_m.pct_change()
    
    pivot_df = pd.DataFrame({
        'Year': df_ret.index.year,
        'Month': df_ret.index.month,
        'Return': df_ret.values
    })
    return pivot_df.pivot(index='Year', columns='Month', values='Return')

# ==============================================================================
# [UI: Sidebar]
# ==============================================================================
st.sidebar.header("🔧 시뮬레이션 설정")

# [중요] 주식 + ETF 합본 리스트 로드
KRX_TICKERS = get_stock_listing()

# 1. 투자 방식
sim_mode_raw = st.sidebar.radio("투자 방식", ["거치식 (Lump-sum)", "적립식 (DCA)"])
sim_mode = sim_mode_raw.split()[0]
dca_interval = "매월"

# 2. 금액 및 주기
if sim_mode == "거치식":
    input_amt = st.sidebar.number_input("초기 거치 금액 (원)", value=10000000, step=1000000, min_value=1, format="%d")
    payment_amt = 0
    st.sidebar.caption(f"💰 시작 원금: **{input_amt:,}원**")
else:
    c_opt1, c_opt2 = st.sidebar.columns(2)
    with c_opt1:
        dca_interval = st.radio("적립 주기", ["매월", "매년"], index=0)
    with c_opt2:
        payment_amt = st.number_input("회당 적립금 (원)", value=1000000, step=10000, min_value=1, format="%d")
    input_amt = 0
    st.sidebar.caption(f"📅 {dca_interval} **{payment_amt:,}원** 투자")

# 3. 기간 설정
start_date = st.sidebar.date_input("시작일", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("종료일", datetime.now())

if start_date >= end_date:
    st.sidebar.error("🚨 시작일은 종료일보다 앞서야 합니다.")

# 4. 종목 선택
st.sidebar.divider()
st.sidebar.subheader("📌 종목 코드 입력")
# 기본값 ETF로 변경
c1, c2 = st.sidebar.columns(2)
with c1: t1 = st.text_input("종목 1", value="360750", max_chars=6) # TIGER 미국S&P500
with c2: t2 = st.text_input("종목 2", value="279530", max_chars=6) # KODEX 고배당
with c1: t3 = st.text_input("종목 3", value="", max_chars=6)
with c2: t4 = st.text_input("종목 4", value="", max_chars=6)

raw_tickers = [t1, t2, t3, t4]
tickers = []
for t in raw_tickers:
    t_clean = t.strip()
    if t_clean:
        if len(t_clean) == 6 and t_clean.isdigit():
            tickers.append(t_clean)
        else:
            st.sidebar.warning(f"⚠️ '{t_clean}'은(는) 유효한 6자리 코드가 아닙니다. 제외됩니다.")

# ==============================================================================
# [Main Logic]
# ==============================================================================
st.title("💸 내 연금 계좌 백테스트 Pro")
st.markdown(f"##### 💡 **{sim_mode} ({dca_interval if sim_mode=='적립식' else '일시불'})** + **배당 재투자**")

if st.sidebar.button("🚀 시뮬레이션 시작", type="primary"):
    if start_date >= end_date:
        st.error("시작일과 종료일을 확인해주세요.")
    elif not tickers:
        st.error("유효한 종목 코드를 하나 이상 입력해주세요.")
    else:
        with st.spinner('데이터 분석 및 시뮬레이션 중...'):
            data_frames = {}
            temp_start_dates = []
            name_map = {}
            
            # --- 데이터 수집 ---
            for t in tickers:
                df = get_stock_data(t, start_date, end_date)
                if df is not None and not df.empty:
                    data_frames[t] = df
                    temp_start_dates.append(df.index.min())
                    # [중요] 종목명 매핑 실행
                    name_map[t] = get_stock_name(t, KRX_TICKERS)
            
            if not data_frames:
                st.error("데이터를 가져올 수 없습니다. 코드를 확인해주세요.")
                st.stop()
                
            common_start = max(temp_start_dates)
            st.success(f"✅ 분석 기간: **{common_start.strftime('%Y-%m-%d')} ~ {end_date.strftime('%Y-%m-%d')}**")
            
            # --- 시뮬레이션 실행 ---
            results = {}
            summary_stats = []
            
            for t, df in data_frames.items():
                df_trimmed = df[df.index >= common_start]
                res_df = run_simulation(df_trimmed, input_amt, payment_amt, sim_mode, dca_interval)
                results[t] = res_df
                
                # 통계 계산
                final_val = res_df['Total_Value'].iloc[-1]
                total_principal = res_df['Principal'].iloc[-1]
                total_return = (final_val - total_principal) / total_principal if total_principal > 0 else 0
                
                days = (res_df.index[-1] - res_df.index[0]).days
                cagr = (final_val / total_principal) ** (365/days) - 1 if days > 0 and total_principal > 0 else 0
                mdd = (res_df['Total_Value'] / res_df['Total_Value'].cummax() - 1).min()
                
                summary_stats.append({
                    "종목명": name_map.get(t, t), # 한글 종목명
                    "티커": t, # 티커
                    "최종 평가액": final_val,
                    "총 투자원금": total_principal,
                    "수익금": final_val - total_principal,
                    "총 수익률": total_return,
                    "CAGR": cagr,
                    "MDD": mdd
                })

            # ==================================================================
            # [UI: Tabs 구성]
            # ==================================================================
            tab1, tab2, tab3 = st.tabs(["📈 종합 비교", "🔍 종목별 상세 (히트맵)", "📥 데이터 다운로드"])
            
            # --- Tab 1: 종합 차트 ---
            with tab1:
                fig = go.Figure()
                
                # 1. 각 종목 자산 성장 그래프
                for t, res in results.items():
                    stock_name = name_map.get(t, t)
                    # 수익률 계산 (원금이 0 이상일 때만)
                    if res['Principal'].iloc[-1] > 0:
                        roi = (res['Total_Value'].iloc[-1] / res['Principal'].iloc[-1]) - 1
                    else:
                        roi = 0

                    fig.add_trace(go.Scatter(
                        x=res.index, y=res['Total_Value'], 
                        name=f"{stock_name} ({roi:+.1%})", # 범례에 한글 이름 표시
                        line=dict(width=2)
                    ))
                
                # 2. 투자 원금 라인
                first_t = list(results.keys())[0]
                fig.add_trace(go.Scatter(
                    x=results[first_t].index, y=results[first_t]['Principal'],
                    name="투자 원금", line=dict(color='gray', dash='dash'), opacity=0.6
                ))

                fig.update_layout(
                    title=f"자산 성장 추이",
                    xaxis_title="날짜",
                    yaxis_title="평가 금액 (원)",
                    hovermode="x unified",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # 요약 테이블
                st.subheader("📊 성과 요약")
                df_stats = pd.DataFrame(summary_stats)
                
                # 포맷팅
                df_disp = df_stats.copy()
                df_disp['최종 평가액'] = df_disp['최종 평가액'].apply(lambda x: f"{int(x):,}원")
                df_disp['총 투자원금'] = df_disp['총 투자원금'].apply(lambda x: f"{int(x):,}원")
                df_disp['수익금'] = df_disp['수익금'].apply(lambda x: f"{int(x):,}원")
                df_disp['총 수익률'] = df_disp['총 수익률'].apply(lambda x: f"{x:.2%}")
                df_disp['CAGR'] = df_disp['CAGR'].apply(lambda x: f"{x:.2%}")
                df_disp['MDD'] = df_disp['MDD'].apply(lambda x: f"{x:.2%}")
                
                # 인덱스를 숨기고 종목명, 티커 컬럼을 모두 보여줌
                st.dataframe(df_disp, use_container_width=True, hide_index=True)

            # --- Tab 2: 종목별 상세 (Heatmap) ---
            with tab2:
                st.caption("📅 월별 수익률 히트맵을 통해 계절성과 변동성을 확인하세요.")
                selected_ticker = st.selectbox("분석할 종목 선택", list(results.keys()), format_func=lambda x: name_map.get(x, x))
                
                if selected_ticker:
                    target_df = results[selected_ticker]
                    stock_name = name_map[selected_ticker]
                    monthly_ret = calculate_monthly_returns(target_df)
                    
                    # Heatmap
                    fig_map = px.imshow(
                        monthly_ret,
                        labels=dict(x="월", y="연도", color="수익률"),
                        x=monthly_ret.columns,
                        y=monthly_ret.index,
                        color_continuous_scale="RdBu",
                        color_continuous_midpoint=0,
                        text_auto='.1%'
                    )
                    fig_map.update_layout(title=f"{stock_name} 월별 수익률")
                    st.plotly_chart(fig_map, use_container_width=True)
                    
                    # MDD Chart
                    dd = (target_df['Total_Value'] / target_df['Total_Value'].cummax() - 1)
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(
                        x=dd.index, y=dd, fill='tozeroy', line=dict(color='red', width=1), name='MDD'
                    ))
                    fig_dd.update_layout(title=f"{stock_name} 전고점 대비 하락률 (Drawdown)", yaxis_tickformat=".1%")
                    st.plotly_chart(fig_dd, use_container_width=True)

            # --- Tab 3: 다운로드 ---
            with tab3:
                st.subheader("📥 시뮬레이션 결과 다운로드")
                
                output = io.BytesIO()
                # xlsxwriter 엔진 사용 (requirements.txt에 추가 필수)
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                    df_stats.to_excel(writer, index=False, sheet_name='Summary')
                    
                    for t, res in results.items():
                        # 시트 이름에 한글 종목명 사용 (특수문자 제거 및 길이 제한)
                        safe_name = "".join(c for c in name_map.get(t, t) if c.isalnum() or c in (' ', '_', '-'))
                        sheet_name = safe_name[:30] # 엑셀 시트 이름 길이 제한
                        res.to_excel(writer, sheet_name=sheet_name)
                        
                processed_data = output.getvalue()
                
                st.download_button(
                    label="📊 엑셀 파일 다운로드 (Excel)",
                    data=processed_data,
                    file_name=f'backtest_results_{datetime.now().strftime("%Y%m%d")}.xlsx',
                    mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
                )

else:
    st.info("👈 사이드바에서 조건을 설정하고 '시뮬레이션 시작'을 눌러주세요.")

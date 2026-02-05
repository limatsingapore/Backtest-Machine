import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime

# --- [페이지 설정] ---
st.set_page_config(page_title="Pension Stock Backtester", layout="wide", page_icon="📈")

# --- [스타일링] ---
st.markdown("""
    <style>
    .main { background-color: #0e1117; color: #fafafa; }
    div[data-testid="stMetric"] { background-color: #262730; padding: 15px; border-radius: 10px; }
    </style>
    """, unsafe_allow_html=True)

# ==============================================================================
# [Helper Functions]
# ==============================================================================

@st.cache_data(ttl=3600*24)
def get_stock_listing():
    """KRX 전체 종목 리스트를 가져와서 캐싱합니다 (이름 매핑용)."""
    try:
        df = fdr.StockListing('KRX')
        # 딕셔너리로 변환 {Symbol: Name}
        return dict(zip(df['Code'], df['Name']))
    except:
        return {}

@st.cache_data(ttl=3600*12)
def get_stock_data(ticker, start_date, end_date):
    """주가 데이터(FDR)와 배당 데이터(YFinance)를 가져와 병합합니다."""
    # 1. 주가 데이터
    df_price = fdr.DataReader(ticker, start_date, end_date)
    if df_price.empty:
        return None
    df_price = df_price[['Close']]
    
    # 2. 배당 데이터 (yfinance)
    yf_ticker = f"{ticker}.KS" 
    try:
        yf_obj = yf.Ticker(yf_ticker)
        dividends = yf_obj.dividends
        dividends.index = dividends.index.tz_localize(None)
        dividends = dividends[(dividends.index >= pd.to_datetime(start_date)) & 
                              (dividends.index <= pd.to_datetime(end_date))]
    except:
        dividends = pd.Series(dtype=float)

    # 3. 병합
    df = df_price.copy()
    df['Dividend'] = 0.0
    common_dates = df.index.intersection(dividends.index)
    if not common_dates.empty:
        df.loc[common_dates, 'Dividend'] = dividends.loc[common_dates]
    
    return df

def run_simulation(df, initial_capital, payment_amt, mode, interval="매월"):
    """
    시뮬레이션 엔진 (연단위 적립 로직 추가)
    """
    df = df.copy()
    df['Shares'] = 0.0
    df['Principal'] = 0.0
    df['Total_Value'] = 0.0
    
    shares = 0.0
    principal = 0.0
    
    # 거치식 초기 매수
    if mode == "거치식":
        price = df.iloc[0]['Close']
        if price > 0:
            shares = initial_capital / price
            principal = initial_capital
    
    share_history = []
    principal_history = []
    
    prev_month = df.index[0].month
    prev_year = df.index[0].year
    
    # 첫 해/첫 달 적립 여부 플래그
    is_first_period = True

    for date, row in df.iterrows():
        price = row['Close']
        div = row['Dividend']
        curr_year = date.year
        curr_month = date.month
        
        # 1. 적립식 매수 로직
        if mode == "적립식" and price > 0:
            should_buy = False
            
            # (1) 첫 데이터 날짜에 즉시 1회 적립
            if is_first_period:
                should_buy = True
                is_first_period = False
                
            # (2) 이후 주기별 적립
            else:
                if interval == "매월":
                    if curr_month != prev_month: # 월이 바뀔 때
                        should_buy = True
                elif interval == "매년":
                    if curr_year != prev_year: # 해가 바뀔 때 (연초)
                        should_buy = True

            if should_buy:
                shares += payment_amt / price
                principal += payment_amt
                
                # 상태 업데이트
                prev_month = curr_month
                prev_year = curr_year
        
        # 2. 배당 재투자 (공통)
        if div > 0 and shares > 0 and price > 0:
            dividend_amount = shares * div
            shares += dividend_amount / price # 세전 재투자 가정
            
        share_history.append(shares)
        principal_history.append(principal)
        
    df['Shares'] = share_history
    df['Principal'] = principal_history
    df['Total_Value'] = df['Shares'] * df['Close']
    
    return df

# ==============================================================================
# [UI: Sidebar]
# ==============================================================================
st.sidebar.header("🔧 시뮬레이션 설정")

# 종목명 매핑 데이터 로드 (캐시됨)
KRX_TICKERS = get_stock_listing()

# 1. 투자 방식
sim_mode_raw = st.sidebar.radio("투자 방식", ["거치식 (Lump-sum)", "적립식 (DCA)"])
sim_mode = sim_mode_raw.split()[0]

dca_interval = "매월" # 기본값

# 2. 금액 및 주기 설정
if sim_mode == "거치식":
    input_amt = st.sidebar.number_input("초기 거치 금액 (원)", value=10000000, step=1000000, format="%d")
    payment_amt = 0
    st.sidebar.caption(f"💰 시작 원금: **{input_amt:,}원**")
else:
    # 적립식일 때 주기 선택 옵션 표시
    c_opt1, c_opt2 = st.sidebar.columns(2)
    with c_opt1:
        dca_interval = st.radio("적립 주기", ["매월", "매년"], index=0)
    with c_opt2:
        payment_amt = st.number_input("회당 적립금 (원)", value=1000000, step=100000, format="%d")
        
    input_amt = 0
    st.sidebar.caption(f"📅 {dca_interval} **{payment_amt:,}원** 투자")

# 3. 기간 설정
start_date = st.sidebar.date_input("시작일", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("종료일", datetime.now())

# 4. 종목 선택 (자유 입력 4칸)
st.sidebar.divider()
st.sidebar.subheader("📌 종목 코드 입력")
st.sidebar.caption("종목코드 6자리를 입력하세요.")

c1, c2 = st.sidebar.columns(2)
with c1: t1 = st.text_input("종목 1", value="069500") # KODEX 200
with c2: t2 = st.text_input("종목 2", value="360750") # TIGER 미국S&P500
with c1: t3 = st.text_input("종목 3", value="")
with c2: t4 = st.text_input("종목 4", value="")

input_list = [t1, t2, t3, t4]
tickers = [t.strip() for t in input_list if t.strip() != ""]

# ==============================================================================
# [Main Logic]
# ==============================================================================
st.title("💸 내 연금 계좌 백테스트")
st.markdown(f"##### 💡 **{sim_mode} ({dca_interval if sim_mode=='적립식' else '일시불'})** + **배당 재투자** 성과 비교")

if st.sidebar.button("🚀 시뮬레이션 시작", type="primary"):
    if not tickers:
        st.error("종목을 하나 이상 입력해주세요.")
    else:
        with st.spinner('데이터 분석 및 시뮬레이션 중...'):
            data_frames = {}
            temp_start_dates = []
            
            # 데이터 수집
            for t in tickers:
                df = get_stock_data(t, start_date, end_date)
                if df is not None and not df.empty:
                    data_frames[t] = df
                    temp_start_dates.append(df.index.min())
            
            if not data_frames:
                st.error("데이터를 가져올 수 없습니다. 코드를 확인해주세요.")
                st.stop()
                
            # 공통 시작일
            common_start = max(temp_start_dates)
            st.info(f"⏳ 공통 분석 시작일: **{common_start.strftime('%Y-%m-%d')}**")
            
            results = {}
            name_map = {} # {티커: 종목명} 저장용
            
            for t, df in data_frames.items():
                # 종목명 찾기 (없으면 티커 그대로)
                name_map[t] = KRX_TICKERS.get(t, t)
                
                df_trimmed = df[df.index >= common_start]
                res_df = run_simulation(df_trimmed, input_amt, payment_amt, sim_mode, dca_interval)
                results[t] = res_df

            # --- 차트 시각화 ---
            fig = go.Figure()
            summary_stats = []
            
            for t, res in results.items():
                final_val = res['Total_Value'].iloc[-1]
                total_principal = res['Principal'].iloc[-1]
                
                # 종목명 표시 (예: 삼성전자 (005930))
                display_name = f"{name_map[t]} ({t})"
                
                if total_principal > 0:
                    total_return = (final_val - total_principal) / total_principal
                else:
                    total_return = 0.0
                
                days = (res.index[-1] - res.index[0]).days
                if days > 0 and final_val > 0 and total_principal > 0:
                    cagr = (final_val / total_principal) ** (365/days) - 1
                else:
                    cagr = 0.0
                    
                mdd = (res['Total_Value'] / res['Total_Value'].cummax() - 1).min()
                
                fig.add_trace(go.Scatter(
                    x=res.index, 
                    y=res['Total_Value'], 
                    name=f"{name_map[t]} ({total_return:+.1%})", # 범례에 이름 표시
                    mode='lines',
                    line=dict(width=2)
                ))
                
                summary_stats.append({
                    "종목명": display_name,
                    "최종 평가액": f"{int(final_val):,}원",
                    "총 투자원금": f"{int(total_principal):,}원",
                    "수익금": f"{int(final_val - total_principal):,}원",
                    "총 수익률": f"{total_return:.2%}",
                    "CAGR": f"{cagr:.2%}",
                    "MDD": f"{mdd:.2%}"
                })
            
            # 원금 라인 (첫번째 종목 기준)
            if results:
                first_t = list(results.keys())[0]
                fig.add_trace(go.Scatter(
                    x=results[first_t].index,
                    y=results[first_t]['Principal'],
                    name="투자 원금",
                    line=dict(color='gray', dash='dash'),
                    opacity=0.6
                ))

            fig.update_layout(
                title=f"자산 성장 추이 ({sim_mode} - {dca_interval if sim_mode=='적립식' else '일시불'})",
                xaxis_title="날짜",
                yaxis_title="평가 금액 (원)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # --- 결과 테이블 ---
            st.subheader("📊 성과 상세 분석")
            df_stats = pd.DataFrame(summary_stats).set_index("종목명") # 인덱스를 종목명으로
            st.dataframe(df_stats, use_container_width=True)
            
            st.warning("⚠️ 참고사항")
            st.caption("""
            1. **종목명**: KRX 상장 종목 리스트를 기반으로 자동 변환됩니다. (해외 직구 종목 등은 티커로 표시될 수 있습니다)
            2. **연단위 적립**: 선택 시 매년 1월(또는 데이터가 있는 첫 거래일)에 적립합니다.
            3. **소수점 매수**: 배당 재투자 및 적립 시 소수점 단위 주식까지 매수했다고 가정합니다.
            """)

else:
    st.info("👈 사이드바에서 조건을 설정하고 '시뮬레이션 시작'을 눌러주세요.")

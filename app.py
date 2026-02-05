import streamlit as st
import pandas as pd
import numpy as np
import FinanceDataReader as fdr
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime, timedelta
import pytz

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

@st.cache_data(ttl=3600*12)
def get_stock_data(ticker, start_date, end_date):
    """
    주가 데이터(FDR)와 배당 데이터(YFinance)를 가져와 병합합니다.
    """
    # 1. 주가 데이터 (FinanceDataReader)
    df_price = fdr.DataReader(ticker, start_date, end_date)
    if df_price.empty:
        return None
    df_price = df_price[['Close']]
    
    # 2. 배당 데이터 (yfinance)
    # 한국 주식은 티커 뒤에 .KS(코스피) 또는 .KQ(코스닥) 필요
    yf_ticker = f"{ticker}.KS" 
    
    try:
        yf_obj = yf.Ticker(yf_ticker)
        dividends = yf_obj.dividends
        # Timezone 제거 (FDR 데이터와 맞추기 위함)
        dividends.index = dividends.index.tz_localize(None)
        
        # 기간 필터링
        dividends = dividends[(dividends.index >= pd.to_datetime(start_date)) & 
                              (dividends.index <= pd.to_datetime(end_date))]
    except:
        dividends = pd.Series(dtype=float)

    # 3. 데이터 병합
    df = df_price.copy()
    df['Dividend'] = 0.0
    
    # 배당금이 있는 날짜에 값 매핑
    common_dates = df.index.intersection(dividends.index)
    if not common_dates.empty:
        df.loc[common_dates, 'Dividend'] = dividends.loc[common_dates]
    
    return df

def run_simulation(df, initial_capital, monthly_payment, mode):
    """
    거치식/적립식 및 배당 재투자 시뮬레이션 엔진
    """
    df = df.copy()
    df['Shares'] = 0.0       # 보유 주식 수
    df['Principal'] = 0.0    # 총 투입 원금
    df['Total_Value'] = 0.0  # 총 평가액
    
    shares = 0.0
    principal = 0.0
    
    # 거치식일 경우 첫날 매수
    if mode == "거치식":
        price = df.iloc[0]['Close']
        if price > 0:
            shares = initial_capital / price
            principal = initial_capital
    
    share_history = []
    principal_history = []
    
    prev_month = df.index[0].month
    
    for date, row in df.iterrows():
        price = row['Close']
        div = row['Dividend']
        
        # 1. 적립식 매수 (매월 첫 거래일)
        if mode == "적립식":
            curr_month = date.month
            if curr_month != prev_month: # 달이 바뀌면 투자
                if price > 0:
                    added_shares = monthly_payment / price
                    shares += added_shares
                    principal += monthly_payment
                prev_month = curr_month
        
        # 첫 달(적립식 시작일) 처리
        if mode == "적립식" and principal == 0 and price > 0:
             shares += monthly_payment / price
             principal += monthly_payment
        
        # 2. 배당 재투자
        if div > 0 and shares > 0 and price > 0:
            # 세전 배당금 전액 재투자 가정 (연금계좌/ISA)
            dividend_amount = shares * div
            reinvested_shares = dividend_amount / price
            shares += reinvested_shares
            
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

# 1. 투자 방식
sim_mode = st.sidebar.radio("투자 방식", ["거치식 (Lump-sum)", "적립식 (DCA)"])

# 2. 금액 입력
if sim_mode.startswith("거치식"):
    input_amt = st.sidebar.number_input("초기 거치 금액 (원)", value=10000000, step=1000000, format="%d")
    monthly_amt = 0
    st.sidebar.caption(f"💰 시작 원금: **{input_amt:,}원**")
else:
    input_amt = 0
    monthly_amt = st.sidebar.number_input("월 적립 금액 (원)", value=1000000, step=100000, format="%d")
    st.sidebar.caption(f"📅 매월 **{monthly_amt:,}원** 투자")

# 3. 기간 설정
start_date = st.sidebar.date_input("시작일", datetime(2018, 1, 1))
end_date = st.sidebar.date_input("종료일", datetime.now())

# 4. 종목 선택 (자유 입력 4칸)
st.sidebar.divider()
st.sidebar.subheader("📌 종목 코드 입력 (최대 4개)")
st.sidebar.caption("KOSPI/KOSDAQ 종목코드 6자리를 입력하세요. (비워두면 무시됩니다)")

tickers = []
# 2열 2행으로 배치하여 공간 효율화 (선택 사항, 그냥 1열로 해도 됨)
c1, c2 = st.sidebar.columns(2)

# 입력 필드 1 (기본값: KODEX 200)
with c1:
    t1 = st.text_input("종목 1", value="069500")
# 입력 필드 2 (기본값: TIGER 미국S&P500)
with c2:
    t2 = st.text_input("종목 2", value="360750")
# 입력 필드 3 (비워둠)
with c1:
    t3 = st.text_input("종목 3", value="")
# 입력 필드 4 (비워둠)
with c2:
    t4 = st.text_input("종목 4", value="")

# 입력된 값들만 모아서 리스트로 만들기
input_list = [t1, t2, t3, t4]
tickers = [t.strip() for t in input_list if t.strip() != ""]

# ------------------------------------------------------------------------------
# 이후 [Main Logic] 코드는 기존과 동일하게 tickers 리스트를 사용하므로 수정 불필요
# ------------------------------------------------------------------------------

# ==============================================================================
# [Main Logic]
# ==============================================================================
st.title("💸 내 연금 계좌 백테스트")
st.markdown("##### 💡 실제 배당금 데이터를 불러와 **배당 재투자(Total Return)** 성과를 비교합니다.")

# [중요] if문 시작 (들여쓰기 없음)
if st.sidebar.button("🚀 시뮬레이션 시작", type="primary"):
    if not tickers:
        st.error("종목을 하나 이상 선택해주세요.")
    else:
        with st.spinner('데이터 수집 및 배당 재투자 계산 중...'):
            data_frames = {}
            temp_start_dates = []
            
            # 데이터 수집
            for t in tickers:
                df = get_stock_data(t, start_date, end_date)
                if df is not None and not df.empty:
                    data_frames[t] = df
                    temp_start_dates.append(df.index.min())
            
            if not data_frames:
                st.error("데이터를 가져올 수 없습니다. 종목 코드나 기간을 확인해주세요.")
                st.stop()
                
            # 공통 시작일 찾기
            common_start = max(temp_start_dates)
            st.info(f"⏳ 공통 분석 시작일: **{common_start.strftime('%Y-%m-%d')}** (선택한 종목 중 데이터가 가장 짧은 종목 기준)")
            
            # 시뮬레이션 실행
            results = {}
            mode_str = sim_mode.split()[0] # "거치식" or "적립식"
            
            for t, df in data_frames.items():
                df_trimmed = df[df.index >= common_start]
                res_df = run_simulation(df_trimmed, input_amt, monthly_amt, mode_str)
                results[t] = res_df

            # 차트 시각화
            fig = go.Figure()
            summary_stats = []
            
            for t, res in results.items():
                final_val = res['Total_Value'].iloc[-1]
                total_principal = res['Principal'].iloc[-1]
                
                # 수익률 계산 (ZeroDivisionError 방지)
                if total_principal > 0:
                    total_return = (final_val - total_principal) / total_principal
                else:
                    total_return = 0.0
                    
                days = (res.index[-1] - res.index[0]).days
                if days > 0 and total_principal > 0:
                    cagr = (final_val / total_principal) ** (365/days) - 1
                else:
                    cagr = 0.0
                    
                # MDD 계산
                cum_max = res['Total_Value'].cummax()
                # cum_max가 0인 경우 방지
                with np.errstate(divide='ignore', invalid='ignore'):
                    dd = (res['Total_Value'] / cum_max) - 1
                mdd = dd.min() if not dd.empty else 0.0
                
                # 차트 추가
                fig.add_trace(go.Scatter(
                    x=res.index, 
                    y=res['Total_Value'], 
                    name=f"{t} ({total_return:+.1%})",
                    mode='lines',
                    line=dict(width=2)
                ))
                
                summary_stats.append({
                    "종목코드": t,
                    "최종 평가액": f"{int(final_val):,}원",
                    "총 투자원금": f"{int(total_principal):,}원",
                    "수익금": f"{int(final_val - total_principal):,}원",
                    "총 수익률": f"{total_return:.2%}",
                    "CAGR (연평균)": f"{cagr:.2%}",
                    "MDD (최대낙폭)": f"{mdd:.2%}"
                })
            
            # 원금 라인 추가
            if results:
                first_ticker = list(results.keys())[0]
                fig.add_trace(go.Scatter(
                    x=results[first_ticker].index,
                    y=results[first_ticker]['Principal'],
                    name="투자 원금",
                    line=dict(color='gray', dash='dash'),
                    opacity=0.6
                ))

            fig.update_layout(
                title=f"자산 성장 추이 ({mode_str})",
                xaxis_title="날짜",
                yaxis_title="평가 금액 (원)",
                hovermode="x unified",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # 결과 테이블
            st.subheader("📊 성과 상세 분석")
            df_stats = pd.DataFrame(summary_stats).set_index("종목코드")
            st.dataframe(df_stats, use_container_width=True)
            
            # [수정된 참고사항 문구]
            st.warning("⚠️ 시뮬레이션 해석 시 주의사항")
            st.caption("""
            1. **배당 재투자 가정**: 배당금 발생 시 세금(15.4%)을 차감하지 않고 전액 재투자하는 **연금계좌(과세이연)** 환경을 가정했습니다.
            2. **소수점 매수 적용**: 배당금 재투자 및 적립식 투자 시 잔돈을 남기지 않고 **소수점 단위(0.xxxx주)까지 주식을 매수**했다고 가정하여 계산했습니다. 
               (실제 매매 시에는 1주 단위 매수 및 잔여 현금 발생으로 인해 오차가 발생할 수 있습니다.)
            3. **데이터 출처**: 주가는 FinanceDataReader, 배당금은 Yahoo Finance 데이터를 사용했습니다.
            """)

# [중요] else문 위치 (if와 동일하게 들여쓰기 없음)
else:
    st.info("👈 사이드바에서 조건을 설정하고 '시뮬레이션 시작'을 눌러주세요.")

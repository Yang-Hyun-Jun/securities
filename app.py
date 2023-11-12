import streamlit as st
import pandas as pd

from time import sleep
from agent import RLSEARCH

# RL agent
factors = ['D3', 'D7', 'D14', 'NEWS']

config = {
    'Number': 30, 
    'Quantile': 1,
    'Balance': 1000,
    'Quarter': '1Q',
    'Factors': factors,
    'Dim': len(factors)
        }

RLsearch = RLSEARCH(config)

iter = 200
train_start = '2023-01'
train_end = '2023-03'
test_start = '2023-01'
test_end = '2023-09'

# 페이지 기본 설정
st.set_page_config(
    page_icon='😳',
    page_title='STREAM LIT',
    layout='wide',
)

# 페이지 헤더, 서브헤더 제목 설정
st.header('AI-Driven Portfolio 🤖')

prompt = st.text_input("What kind of strategy do you want to create?")

if len(prompt) > 0:
    # reward_func = set_reward_by_gpt(prompt)

    # Progress bar
    progress_text = 'Searching strategy ..'
    progress_bar = st.progress(0, text=progress_text)
    

    # Strategy search
    for i in range(1, iter):
        progress_bar.progress(int(i/(iter // 100)), text=progress_text)
        RLsearch.search(1, train_start, train_end)
    
    optimal = RLsearch.get_w(False)
    optimal = optimal.detach().numpy()
    RLsearch.init(optimal)
    PVs, PFs, TIs, POs, result = RLsearch.test(test_start, test_end)

    RLsearch.config['Factors'] = ['D3', 'D7', 'D14']
    RLsearch.config['Dim'] = len(['D3', 'D7', 'D14'])
    RLsearch.init(None)
    PVs_BM, _, _, _, _ = RLsearch.test(test_start, test_end)

    sr = round(RLsearch.get_sr(PVs), 2)
    er_d = round(RLsearch.get_er(PVs), 3)
    er_m = round(RLsearch.get_er(PVs) * 250/12 * 100, 3)
    er_y = round(RLsearch.get_er(PVs) * 250 * 100, 3)
    si = round(RLsearch.get_si(PVs) * 100, 3)
    md = round(RLsearch.get_mdd(PVs), 2)
    pl = round(100 * (PVs[-1]-PVs[0]) / PVs[0], 2)

    top_5 = TIs[-1][:5]
    df = pd.DataFrame(
    {'name': top_5,
     'date': ['2023-06-30' for _ in range(5)],
    'Views (past 30 days)': [RLsearch.price[ticker].to_numpy()[-30:] for ticker in top_5]}
    )

    sleep(1)
    progress_bar.empty()

    # 페이지 컬럼 분할 1
    st.subheader('Optimal Strategy')
    cols = st.columns((1, 1, 2))
    cols[0].metric("3일 주가 모멘텀", f"{round(optimal[0] * 100, 2)} %")
    cols[0].metric("7일 주가 모멘텀", f"{round(optimal[1] * 100, 2)} %")
    cols[1].metric("14일 주가 모멘텀", f"{round(optimal[2] * 100, 2)} %")
    cols[1].metric("뉴스 부정 점수", f"{round(optimal[3] * 100, 2)} %")

    cols[2].write("**Top 5 Stock**")
    cols[2].dataframe(
        df,

        column_config={
            'name': 'Stock Ticker',
            'Views (past 30 days)': st.column_config.LineChartColumn(
                'Close Price Views (past 30 days)')},

        hide_index=True)


    st.markdown("---")

    # 페이지 컬럼 분할 2 
    st.subheader('Search Result')
    cols = st.columns((1, 1, 2))

    st.markdown("---")

    cols[0].metric("Profitloss %", f"{pl}", "+ Higher is better")
    cols[0].metric("Sharpe Ratio", f"{sr}", "+")
    cols[0].metric("Expected Daily %", f'{er_d}', "+")
    cols[1].metric("Return Deviation %", f"{si}", "- Lower is better")
    cols[1].metric("Maximum Drawdown %", f"{md}", "-")
    cols[1].metric("Expected Monthly %", f"{er_m}", "+")

    chart_data = pd.DataFrame(
        {'AI strategy': PVs,
         'BM (not optimized)': PVs_BM},
        index=RLsearch.price[test_start:test_end].index
        )
    
    cols[2].line_chart(chart_data, color=['#005eff', '#c8c8c8'])

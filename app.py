import streamlit as st
import pandas as pd
import numpy as np

from time import sleep

# 페이지 기본 설정
st.set_page_config(
    page_icon='😳',
    page_title='STREAM LIT',
    layout='wide',
)

# 페이지 헤더, 서브헤더 제목 설정
st.header('AI-driven portfolio 🤖')
st.subheader('Prompt')
prompt = st.text_input("What kind of strategy do you want to create?")


if len(prompt) > 0:

    # Progress bar
    progress_text = 'Making strategy ..'
    progress_bar = st.progress(0, text=progress_text)

    for i in range(100):
        sleep(0.01)
        progress_bar.progress(i+1, text=progress_text)

    sleep(1)
    progress_bar.empty()

    # 페이지 컬럼 분할
    cols = st.columns((1, 1, 2))

    cols[0].metric("10/11", "15 °C", "2")
    cols[0].metric("10/12", "17 °C", "2 °F")
    cols[0].metric("10/13", "15 °C", "2")
    cols[1].metric("10/14", "17 °C", "2 °F")
    cols[1].metric("10/15", "14 °C", "-3 °F")
    cols[1].metric("10/16", "13 °C", "-1 °F")

    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['a', 'b', 'c'],
    )

    cols[2].line_chart(chart_data)

    # # 페이지 컬럼 다시 분할
    # cols2 = st.columns((1))
    # cols2[0].line_chart(chart_data)

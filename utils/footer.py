import streamlit as st

def render_footer():
    st.divider()
    st.markdown("""
<div style='text-align:center; color:#888; font-size:0.82rem; padding: 0.5rem 0 1rem 0;'>
본 서비스는 2026 문화 디지털혁신 및 데이터 활용 공모전 출품작으로,<br>
AI 모델을 활용한 게임 등급 재분류 가능성 사전 탐지 서비스입니다.<br><br>
© 2026 GUCC (게임이용자보호센터). All rights reserved.
</div>
""", unsafe_allow_html=True)

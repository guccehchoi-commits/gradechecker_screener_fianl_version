import streamlit as st

def render_footer():
    st.divider()
    st.markdown("""
<div style='text-align:center; color:#888; font-size:0.82rem; padding: 0.5rem 0 1rem 0;'>
본 서비스는 게임이용자보호센터에서 AI 모델을 활용하여 개발한<br>
게임 등급 재분류 가능성 사전 탐지 서비스입니다.<br><br>
© 2026 GUCC (게임이용자보호센터). All rights reserved.
</div>
""", unsafe_allow_html=True)

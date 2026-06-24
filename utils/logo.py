import base64
import streamlit as st
from pathlib import Path

def render_logo():
    logo_path = Path(__file__).parent.parent / 'gucc_logo.png'
    with open(logo_path, 'rb') as f:
        b64 = base64.b64encode(f.read()).decode()

    st.markdown(f"""
<style>
[data-testid="stLogoLink"] {{ display: none !important; }}
.gucc-logo-fixed {{
    position: fixed;
    top: 10px;
    left: 14px;
    z-index: 99999;
    width: 190px;
}}
.gucc-logo-fixed img {{
    width: 100%;
    height: auto;
    display: block;
}}
</style>
<div class="gucc-logo-fixed">
    <img src="data:image/png;base64,{b64}" alt="GUCC 로고">
</div>
""", unsafe_allow_html=True)

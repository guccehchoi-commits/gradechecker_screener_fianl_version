import streamlit as st

def render_logo():
    st.logo('gucc_logo.png', size='large', icon_image='gucc_logo.png')
    st.markdown("""
<style>
[data-testid="stLogoLink"] {
    overflow: visible !important;
    height: auto !important;
    padding: 0.6rem 0.4rem !important;
}
[data-testid="stHeaderLogo"] {
    height: 110px !important;
    width: auto !important;
    max-width: 210px !important;
    object-fit: contain !important;
}
</style>
""", unsafe_allow_html=True)

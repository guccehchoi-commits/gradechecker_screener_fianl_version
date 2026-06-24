import streamlit as st

def render_logo():
    st.logo('gucc_logo.png', size='large', icon_image='gucc_logo.png')
    st.markdown("""
<style>
[data-testid="stLogoLink"] {
    height: auto !important;
    overflow: visible !important;
    padding: 0.8rem 0.5rem 0.5rem 0.5rem !important;
}
[data-testid="stHeaderLogo"] {
    height: 110px !important;
    width: auto !important;
    max-width: 220px !important;
    object-fit: contain !important;
    display: block !important;
}
</style>
""", unsafe_allow_html=True)

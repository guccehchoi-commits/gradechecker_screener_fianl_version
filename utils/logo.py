import streamlit as st

def render_logo():
    st.logo('gucc_logo.png', size='large', icon_image='gucc_logo.png')
    st.markdown("""
<style>
[data-testid="stLogo"] {
    height: 160px !important;
    max-width: 300px !important;
    width: auto !important;
}
[data-testid="stLogoCollapsed"] {
    height: 90px !important;
    max-width: 200px !important;
    width: auto !important;
}
</style>
""", unsafe_allow_html=True)

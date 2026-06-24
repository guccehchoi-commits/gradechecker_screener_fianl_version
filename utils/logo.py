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
[data-testid="stHeaderLogo"],
[data-testid="stSidebarLogo"] {
    height: 110px !important;
    width: auto !important;
    max-width: 210px !important;
    object-fit: contain !important;
}
span[label="app"] p { font-size: 0 !important; }
span[label="app"] p::after { content: "HOME"; font-size: 0.9rem; font-weight: 400; }
</style>
""", unsafe_allow_html=True)

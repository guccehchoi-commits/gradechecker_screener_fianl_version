import streamlit as st

def render_logo():
    st.logo('gucc_logo.png', size='large', icon_image='gucc_logo.png')
    st.markdown("""
<style>
[data-testid="stSidebarHeader"] img,
[data-testid="stSidebarHeader"] > div > img,
[data-testid="stSidebarHeader"] > div > div > img {
    height: 160px !important;
    width: auto !important;
    max-width: 100% !important;
    object-fit: contain !important;
}
[data-testid="stSidebarHeader"] {
    padding-bottom: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

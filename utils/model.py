import pickle
import numpy as np
import streamlit as st
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / 'models'


@st.cache_resource(show_spinner='모델 로드 중...')
def load_model():
    with open(MODEL_DIR / 'xgb_model.pkl', 'rb') as f:
        clf = pickle.load(f)
    with open(MODEL_DIR / 'xgb_preprocessor.pkl', 'rb') as f:
        prep = pickle.load(f)
    return clf, prep


def predict(df_feat, clf, prep):
    from utils.preprocess import CAT_COLS, FEAT_COLS
    X = df_feat[FEAT_COLS].copy()
    for col in CAT_COLS:
        X[col] = X[col].fillna('미상').astype(str)
    prob = clf.predict_proba(prep.transform(X))[:, 1]
    return prob


def risk_label(p: float) -> str:
    if p >= 0.80: return '고위험'
    if p >= 0.50: return '중위험'
    return '저위험'


def risk_color(label: str) -> str:
    return {'고위험': '#C00000', '중위험': '#C55A11', '저위험': '#375C23'}.get(label, '#888')

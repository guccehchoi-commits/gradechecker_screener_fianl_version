import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
st.set_page_config(page_title='SHAP 분석', page_icon='🔍', layout='wide')
# 한글 폰트
try:
    _ko = fm.FontProperties(fname=r'C:\Windows\Fonts\malgun.ttf')
    plt.rcParams['font.family'] = _ko.get_name()
except:
    pass
plt.rcParams['axes.unicode_minus'] = False
st.title('🔍 SHAP 분석 — 탐지 근거')
st.caption('각 피처가 예측 확률을 얼마나 높이거나 낮추는지 시각적으로 확인합니다.')
# ── 데이터 확인 ────────────────────────────────────────────────
if 'result' not in st.session_state:
    st.warning('먼저 **① 파일 업로드** 페이지에서 분석을 실행해 주세요.')
    st.stop()
clf     = st.session_state['clf']
prep    = st.session_state['prep']
df_feat = st.session_state['df_feat']
result  = st.session_state['result']
from utils.preprocess import CAT_COLS, FEAT_COLS
X_raw = df_feat[FEAT_COLS].copy()
for col in CAT_COLS:
    X_raw[col] = X_raw[col].fillna('미상').astype(str)
X_trans = prep.transform(X_raw)
# SHAP explainer (캐시)
@st.cache_resource
def get_explainer(_clf):
    return shap.TreeExplainer(_clf)
with st.spinner('SHAP 계산 중... (처음 한 번만 시간이 걸립니다)'):
    explainer   = get_explainer(clf)
    shap_values = explainer.shap_values(X_trans)
# XGBoost binary: shap_values가 2D array
if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values
# 피처 이름
feature_names = prep.get_feature_names_out()
st.divider()
# ── Global SHAP ────────────────────────────────────────────────
st.subheader('① Global  전체 피처 중요도')
st.caption('SHAP mean |value| 기준 — 값이 클수록 예측에 더 많이 기여한 피처')
n_top = st.slider('상위 N개 피처', 5, 30, 15, key='global_n')
mean_abs = np.abs(sv).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1][:n_top]
top_names = [feature_names[i] for i in top_idx]
top_vals  = mean_abs[top_idx]
fig, ax = plt.subplots(figsize=(8, n_top*0.4 + 1))
colors = ['#C00000' if 'grade=미상' in n or 'grade__미상' in n
          else '#2E75B6' for n in top_names]
bars = ax.barh(range(n_top), top_vals[::-1], color=colors[::-1])
ax.set_yticks(range(n_top))
ax.set_yticklabels(top_names[::-1], fontsize=9)
ax.set_xlabel('mean |SHAP value|')
ax.set_title(f'피처 중요도 상위 {n_top}개', fontsize=11)
ax.spines[['top','right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()
st.divider()
# ── Local SHAP ─────────────────────────────────────────────────
st.subheader('② Local  개별 게임 탐지 근거')
st.caption('특정 게임이 왜 이 확률을 받았는지 피처별로 분해합니다.')
# 게임 선택
game_options = result['game_name'].tolist()
sel_game = st.selectbox('게임 선택', game_options,
                        format_func=lambda n: f'{n}  (확률: {result.loc[result["game_name"]==n, "재분류_확률"].values[0]:.4f})')
# 해당 게임 행 찾기
orig_idx = df_feat.index[df_feat.index.isin(
    result[result['game_name'] == sel_game].index
)]
if len(orig_idx) == 0:
    # game_name으로 매칭
    match = df_feat[df_feat.get('game_name', pd.Series()).eq(sel_game)] if 'game_name' in df_feat.columns else pd.DataFrame()
    if len(match) == 0:
        st.error('해당 게임을 찾을 수 없습니다.')
        st.stop()
    row_idx = match.index[0]
else:
    row_idx = orig_idx[0]
row_sv   = sv[row_idx]
row_prob = float(result.loc[result['game_name']==sel_game, '재분류_확률'].values[0])
# 위험도 뱃지
from utils.model import risk_label, risk_color
level = risk_label(row_prob)
color = risk_color(level)
st.markdown(
    f'<span style="background:{color};color:white;padding:4px 12px;'
    f'border-radius:4px;font-weight:bold">{level}</span>'
    f'&nbsp;&nbsp;재분류 확률: <b>{row_prob:.4f}</b>',
    unsafe_allow_html=True
)
st.markdown('')
# 상위 기여 피처 추출
top_n_local = 10
top_local_idx  = np.argsort(np.abs(row_sv))[::-1][:top_n_local]
local_names = [feature_names[i] for i in top_local_idx]
local_vals  = row_sv[top_local_idx]
fig2, ax2 = plt.subplots(figsize=(8, top_n_local*0.5 + 1))
bar_colors = ['#C00000' if v > 0 else '#2E75B6' for v in local_vals[::-1]]
ax2.barh(range(top_n_local), local_vals[::-1], color=bar_colors)
ax2.set_yticks(range(top_n_local))
ax2.set_yticklabels(local_names[::-1], fontsize=9)
ax2.axvline(0, color='#444', linewidth=0.8)
ax2.set_xlabel('SHAP value  (양수=이상 방향 ▲  /  음수=정상 방향 ▼)')
ax2.set_title(f'{sel_game} — 탐지 근거', fontsize=11)
ax2.spines[['top','right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)
plt.close()
# 탐지 사유 자동 요약
pos_feats = [(local_names[i], local_vals[i])
             for i in range(top_n_local) if local_vals[i] > 0.01]
if pos_feats:
    top3 = ', '.join([f[0] for f in pos_feats[:3]])
    if level in ('고위험', '중위험'):
        st.error(f'**탐지 사유 요약**: {top3}')
    else:
        st.info(f'**주요 피처**: {top3}')
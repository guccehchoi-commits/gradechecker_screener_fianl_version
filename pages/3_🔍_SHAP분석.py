import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go

st.set_page_config(page_title='SHAP 분석', page_icon='🔍', layout='wide')

# ── 공통 색상 팔레트 ───────────────────────────────────────────
C_RISK    = '#E63946'   # 위험 방향 (양수 SHAP)
C_SAFE    = '#457B9D'   # 안전 방향 (음수 SHAP)
C_MISSING = '#C1121F'   # 등급 미상
C_FT      = '#7B2D8B'   # FastText 텍스트
C_NEUTRAL = '#5C6D7E'   # 기타

def _plotly_base():
    return dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='sans-serif', size=12, color='#1C1C1E'),
        margin=dict(l=10, r=30, t=40, b=40),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )

# ── 상단 헤더 ─────────────────────────────────────────────────
st.title('🔍 SHAP 분석 — 탐지 근거')
st.caption('각 항목이 재분류 확률을 얼마나 높이거나 낮추는지 시각적으로 확인합니다.')

# ── 데이터 확인 ───────────────────────────────────────────────
if 'result' not in st.session_state:
    st.warning('먼저 **① 파일 업로드** 페이지에서 분석을 실행해 주세요.')
    st.stop()

clf     = st.session_state['clf']
prep    = st.session_state['prep']
df_feat = st.session_state['df_feat']
result  = st.session_state['result']
thr     = st.session_state.get('global_thr', 0.40)

from utils.preprocess import CAT_COLS, FEAT_COLS, GAMBLING_PATTERN, BETTING_GENRES
from utils.model import risk_label, risk_color

X_raw = df_feat[FEAT_COLS].copy()
for col in CAT_COLS:
    X_raw[col] = X_raw[col].fillna('미상').astype(str)
X_trans = prep.transform(X_raw)

# ── SHAP 계산 ────────────────────────────────────────────────
@st.cache_resource
def get_explainer(_clf):
    return shap.TreeExplainer(_clf)

with st.spinner('SHAP 계산 중... (처음 한 번만 시간이 걸립니다)'):
    explainer   = get_explainer(clf)
    shap_values = explainer.shap_values(X_trans)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values
raw_feature_names = list(prep.get_feature_names_out())

# ── 피처 이름 한국어 변환 ──────────────────────────────────────
_GRADE_LABEL = {
    '전체이용가': '전체이용가', '12세이용가': '12세이용가',
    '15세이용가': '15세이용가', '청소년이용불가': '청소년이용불가',
    '미상': '미상 ★',
}
_LANG_LABEL = {
    '한국어': '한국어', '영어': '영어', '혼합': '한+영 혼합', '기타': '기타 언어',
}

def pretty_name(raw: str) -> str:
    if raw.startswith('remainder__'):
        return '__ft__'
    name = raw.replace('cat__', '', 1)
    if name.startswith('grade__'):
        v = name[len('grade__'):]
        return f'등급: {_GRADE_LABEL.get(v, v)}'
    if name.startswith('language_type__'):
        v = name[len('language_type__'):]
        return f'게임명 언어: {_LANG_LABEL.get(v, v)}'
    if name.startswith('company__'):
        return f'제조사: {name[len("company__"):]}'
    if name.startswith('grade_company__'):
        v = name[len('grade_company__'):]
        parts = v.split('_', 1)
        if len(parts) == 2:
            g, c = parts
            return f'등급+제조사: {_GRADE_LABEL.get(g, g)} / {c}'
        return f'등급+제조사: {v}'
    return name

def aggregate_shap(sv_arr, raw_names):
    """ft_* 차원을 단일 항목으로 집계. 1D(단일 샘플) / 2D(전체) 모두 처리."""
    is_2d = sv_arr.ndim == 2
    ft_idx    = [i for i, n in enumerate(raw_names) if pretty_name(n) == '__ft__']
    other_idx = [i for i, n in enumerate(raw_names) if pretty_name(n) != '__ft__']

    if is_2d:
        other_sv = sv_arr[:, other_idx]
        ft_agg   = sv_arr[:, ft_idx].sum(axis=1, keepdims=True) if ft_idx else None
    else:
        other_sv = sv_arr[other_idx]
        ft_agg   = np.array([sv_arr[ft_idx].sum()]) if ft_idx else None

    labels = [pretty_name(raw_names[i]) for i in other_idx]
    if ft_agg is not None:
        agg_sv = np.hstack([other_sv, ft_agg]) if is_2d else np.append(other_sv, ft_agg)
        labels += ['게임명 텍스트 (FastText)']
    else:
        agg_sv = other_sv
    return agg_sv, labels

st.divider()

# ═══════════════════════════════════════════════════════════════
# ① Global SHAP
# ═══════════════════════════════════════════════════════════════
st.subheader('① 전체 피처 중요도')

col_desc, col_ctrl = st.columns([3, 1])
with col_desc:
    st.caption('모든 게임에 걸쳐 각 항목이 예측에 얼마나 영향을 미치는지 평균낸 값입니다. 막대가 길수록 중요한 항목입니다.')
with col_ctrl:
    n_top = st.slider('상위 N개', 5, 20, 12, key='global_n')

sv_agg, labels_agg = aggregate_shap(sv, raw_feature_names)
mean_abs = np.abs(sv_agg).mean(axis=0)
top_idx  = np.argsort(mean_abs)[::-1][:n_top]

# 내림차순 정렬 (Plotly horizontal bar는 아래→위 순서)
names_plot = [labels_agg[i] for i in top_idx][::-1]
vals_plot  = mean_abs[top_idx][::-1]

bar_colors = []
for n in names_plot:
    if '미상' in n:
        bar_colors.append(C_MISSING)
    elif 'FastText' in n:
        bar_colors.append(C_FT)
    else:
        bar_colors.append(C_NEUTRAL)

fig_global = go.Figure(go.Bar(
    x=vals_plot,
    y=names_plot,
    orientation='h',
    marker=dict(color=bar_colors, line=dict(width=0)),
    hovertemplate='<b>%{y}</b><br>중요도: %{x:.4f}<extra></extra>',
))
fig_global.update_layout(
    **_plotly_base(),
    height=max(280, n_top * 36 + 60),
    xaxis=dict(
        title='mean |SHAP value|',
        gridcolor='rgba(0,0,0,0.07)',
        zeroline=False,
    ),
    yaxis=dict(tickfont=dict(size=11)),
    bargap=0.35,
)
st.plotly_chart(fig_global, use_container_width=True, config={'displayModeBar': False})

# 범례
leg1, leg2, leg3 = st.columns(3)
leg1.markdown(f'<span style="color:{C_MISSING}">■</span> 등급 미상 관련', unsafe_allow_html=True)
leg2.markdown(f'<span style="color:{C_FT}">■</span> 게임명 텍스트 (FastText)', unsafe_allow_html=True)
leg3.markdown(f'<span style="color:{C_NEUTRAL}">■</span> 등급 · 제조사 · 언어 등', unsafe_allow_html=True)

st.divider()

# ═══════════════════════════════════════════════════════════════
# ② Local SHAP
# ═══════════════════════════════════════════════════════════════
st.subheader('② 개별 게임 탐지 근거')
st.caption('게임을 선택하면 해당 게임이 왜 이 확률을 받았는지 항목별로 분해합니다.')

game_options = result['game_name'].tolist()
sel_game = st.selectbox(
    '게임 선택',
    game_options,
    format_func=lambda n: (
        f'{n}  ·  확률 {result.loc[result["game_name"]==n, "재분류_확률"].values[0]:.4f}'
    ),
)

# 인덱스 버그 수정: pandas label → integer position
game_row = result[result['game_name'] == sel_game]
if len(game_row) == 0:
    st.error('해당 게임을 찾을 수 없습니다.')
    st.stop()

if 'game_name' in df_feat.columns:
    match = df_feat[df_feat['game_name'] == sel_game]
    feat_label_idx = match.index[0] if len(match) > 0 else game_row.index[0]
else:
    feat_label_idx = game_row.index[0]

try:
    row_pos = df_feat.index.get_loc(feat_label_idx)
except KeyError:
    row_pos = 0

row_sv   = sv[row_pos]
row_prob = float(game_row['재분류_확률'].values[0])
level    = risk_label(row_prob, thr)
color    = risk_color(level)

# ── 위험도 카드 ───────────────────────────────────────────────
m1, m2, m3 = st.columns(3)
m1.metric('재분류 확률', f'{row_prob:.4f}')
m2.metric('위험 등급', level)
m3.metric('현재 기준값', f'{thr}')

if row_prob >= thr:
    st.success(f'✅ 현재 기준값({thr}) 기준 — **검토 대상**입니다.')
else:
    st.info(f'ℹ️ 현재 기준값({thr}) 기준 — 검토 대상이 아닙니다. (기준값을 낮추면 포함됩니다)')

st.markdown('')

# ── Local SHAP 차트 ───────────────────────────────────────────
row_sv_agg, labels_local = aggregate_shap(row_sv, raw_feature_names)

top_n_local   = 12
top_local_idx = np.argsort(np.abs(row_sv_agg))[::-1][:top_n_local]
local_names   = [labels_local[i] for i in top_local_idx][::-1]
local_vals    = row_sv_agg[top_local_idx][::-1]

bar_colors_local = [C_RISK if v > 0 else C_SAFE for v in local_vals]

# 호버 텍스트: 방향 설명 포함
hover_texts = []
for n, v in zip(local_names, local_vals):
    direction = '위험도 ▲' if v > 0 else '위험도 ▼'
    hover_texts.append(f'<b>{n}</b><br>SHAP: {v:+.4f}  ({direction})')

fig_local = go.Figure()
fig_local.add_trace(go.Bar(
    x=local_vals,
    y=local_names,
    orientation='h',
    marker=dict(color=bar_colors_local, line=dict(width=0)),
    hovertemplate='%{customdata}<extra></extra>',
    customdata=hover_texts,
))
fig_local.add_vline(x=0, line=dict(color='#444', width=1))
fig_local.update_layout(
    **_plotly_base(),
    height=max(320, top_n_local * 38 + 80),
    title=dict(text=f'<b>{sel_game}</b> — 항목별 탐지 기여도', font=dict(size=14)),
    xaxis=dict(
        title='SHAP value',
        gridcolor='rgba(0,0,0,0.07)',
        zeroline=False,
        tickformat='+.3f',
    ),
    yaxis=dict(tickfont=dict(size=11)),
    bargap=0.3,
    annotations=[
        dict(x=max(local_vals) * 0.95, y=0.02, xref='x', yref='paper',
             text='위험도 ▲', showarrow=False,
             font=dict(color=C_RISK, size=11), xanchor='right'),
        dict(x=min(local_vals) * 0.95, y=0.02, xref='x', yref='paper',
             text='위험도 ▼', showarrow=False,
             font=dict(color=C_SAFE, size=11), xanchor='left'),
    ] if len(local_vals) > 0 and max(local_vals) > 0 and min(local_vals) < 0 else [],
)
st.plotly_chart(fig_local, use_container_width=True, config={'displayModeBar': False})

# ── 범례 ──────────────────────────────────────────────────────
lc1, lc2 = st.columns(2)
lc1.markdown(f'<span style="color:{C_RISK}">■</span> **빨간색** — 재분류 가능성을 높이는 항목', unsafe_allow_html=True)
lc2.markdown(f'<span style="color:{C_SAFE}">■</span> **파란색** — 재분류 가능성을 낮추는 항목', unsafe_allow_html=True)

# ── 탐지 사유 자동 요약 ───────────────────────────────────────
pos_feats = [(local_names[i], local_vals[i]) for i in range(len(local_vals)) if local_vals[i] > 0.01]
neg_feats = [(local_names[i], local_vals[i]) for i in range(len(local_vals)) if local_vals[i] < -0.01]

if pos_feats:
    top3 = ' · '.join([f[0] for f in pos_feats[:3]])
    if level == '고위험':
        st.error(f'**주요 탐지 사유**: {top3}')
    elif level == '검토 대상':
        st.warning(f'**주요 탐지 사유**: {top3}')
    else:
        st.info(f'**주요 영향 항목**: {top3}')
if neg_feats:
    neg2 = ' · '.join([f[0] for f in neg_feats[:2]])
    st.caption(f'위험도를 낮추는 항목: {neg2}')

st.divider()

# ═══════════════════════════════════════════════════════════════
# ③ 규칙 기반 보정 내역
# ═══════════════════════════════════════════════════════════════
st.subheader('③ 규칙 기반 보정 내역')
st.caption('모델 점수 외에 게임명·장르 규칙으로 추가 보정된 항목이 있으면 표시됩니다.')

genre_val = df_feat.loc[feat_label_idx, 'genre'] if 'genre' in df_feat.columns else ''
boosts = []

if GAMBLING_PATTERN.search(str(sel_game)):
    grade_val = df_feat.loc[feat_label_idx, 'grade'] if 'grade' in df_feat.columns else ''
    if str(grade_val) != '청소년이용불가':
        boosts.append(('도박 관련 키워드 감지', '게임명에 도박·사행 키워드가 포함되어 확률 +0.20 보정 적용'))

if str(genre_val) in BETTING_GENRES:
    boosts.append(('베팅성 장르 감지', '장르가 보드게임(베팅성)으로 확률 +0.05 보정 적용'))

if boosts:
    for label, desc in boosts:
        st.warning(f'**⚠️ {label}** — {desc}')
    st.caption('※ 위 보정은 SHAP 차트에 직접 반영되지 않으며, 최종 확률에 더해진 별도 규칙입니다.')
else:
    st.success('규칙 기반 보정 없음 — 위 확률은 모델 예측값 그대로입니다.')

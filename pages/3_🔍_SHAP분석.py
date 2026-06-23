import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

st.set_page_config(page_title='SHAP 분석', page_icon='🔍', layout='wide')

# 한글 폰트 (로컬 Windows용 — Cloud 환경에서는 영문 fallback)
_font_loaded = False
try:
    _ko = fm.FontProperties(fname=r'C:\Windows\Fonts\malgun.ttf')
    plt.rcParams['font.family'] = _ko.get_name()
    _font_loaded = True
except Exception:
    pass
plt.rcParams['axes.unicode_minus'] = False

st.title('🔍 SHAP 분석 — 탐지 근거')
st.caption('각 항목이 예측 확률을 얼마나 높이거나 낮추는지 시각적으로 확인합니다.')

# ── 데이터 확인 ────────────────────────────────────────────────
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

if isinstance(shap_values, list):
    sv = shap_values[1]
else:
    sv = shap_values

raw_feature_names = list(prep.get_feature_names_out())

# ── 피처 이름 한국어 변환 ──────────────────────────────────────
def pretty_name(raw: str) -> str:
    """sklearn ColumnTransformer 출력 이름을 사람이 읽기 쉬운 한국어로 변환."""
    # remainder__ft_N → FastText 차원 (집계 후 단일 표시)
    if raw.startswith('remainder__ft_') or raw.startswith('remainder__'):
        return '__ft__'  # 집계용 내부 마커

    # cat__ 접두어 제거
    name = raw.replace('cat__', '', 1) if raw.startswith('cat__') else raw

    _grade_map = {
        '전체이용가': '전체이용가', '12세이용가': '12세이용가',
        '15세이용가': '15세이용가', '청소년이용불가': '청소년이용불가',
        '미상': '미상 ★',
    }
    _lang_map = {
        '한국어': '한국어', '영어': '영어', '혼합': '한+영 혼합', '기타': '기타 언어',
    }

    # grade__ 접두어
    if name.startswith('grade__'):
        val = name[len('grade__'):]
        return f'등급: {_grade_map.get(val, val)}'

    # language_type__ 접두어
    if name.startswith('language_type__'):
        val = name[len('language_type__'):]
        return f'게임명 언어: {_lang_map.get(val, val)}'

    # company__ 접두어
    if name.startswith('company__'):
        val = name[len('company__'):]
        return f'제조사: {val}'

    # grade_company__ 접두어
    if name.startswith('grade_company__'):
        val = name[len('grade_company__'):]
        parts = val.split('_', 1)
        if len(parts) == 2:
            g, c = parts
            return f'등급+제조사: {_grade_map.get(g, g)} / {c}'
        return f'등급+제조사: {val}'

    return name  # fallback


# 피처를 '집계 포함' 형태로 변환
# ft_* 차원들을 하나의 '게임명 텍스트 (FastText)' 피처로 집계

def aggregate_shap(sv_row_or_matrix, raw_names):
    """
    sv_row_or_matrix: 1D (single sample) or 2D (all samples) SHAP array
    raw_names: list of raw feature names
    Returns: (aggregated_values, pretty_labels)
    """
    is_2d = sv_row_or_matrix.ndim == 2

    ft_indices   = [i for i, n in enumerate(raw_names) if pretty_name(n) == '__ft__']
    other_indices = [i for i, n in enumerate(raw_names) if pretty_name(n) != '__ft__']

    if is_2d:
        ft_agg   = np.abs(sv_row_or_matrix[:, ft_indices]).sum(axis=1, keepdims=True) if ft_indices else None
        other_sv = sv_row_or_matrix[:, other_indices]
    else:
        ft_agg   = np.array([sv_row_or_matrix[ft_indices].sum()]) if ft_indices else None
        other_sv = sv_row_or_matrix[other_indices]

    other_labels = [pretty_name(raw_names[i]) for i in other_indices]

    if ft_agg is not None:
        if is_2d:
            agg_sv = np.hstack([other_sv, ft_agg])
        else:
            agg_sv = np.append(other_sv, ft_agg)
        labels = other_labels + ['게임명 텍스트 (FastText)']
    else:
        agg_sv = other_sv
        labels = other_labels

    return agg_sv, labels


st.divider()

# ── ① Global SHAP ─────────────────────────────────────────────
st.subheader('① Global  전체 피처 중요도')
st.caption('SHAP mean |value| 기준 — 값이 클수록 예측에 더 많이 기여한 항목')

n_top = st.slider('상위 N개 항목', 5, 20, 12, key='global_n')

sv_agg, labels_agg = aggregate_shap(sv, raw_feature_names)
mean_abs  = np.abs(sv_agg).mean(axis=0)
top_idx   = np.argsort(mean_abs)[::-1][:n_top]
top_names = [labels_agg[i] for i in top_idx]
top_vals  = mean_abs[top_idx]

fig, ax = plt.subplots(figsize=(8, n_top * 0.45 + 1))
colors = []
for n in top_names:
    if '미상' in n:
        colors.append('#C00000')
    elif 'FastText' in n:
        colors.append('#7030A0')
    else:
        colors.append('#2E75B6')
ax.barh(range(n_top), top_vals[::-1], color=colors[::-1])
ax.set_yticks(range(n_top))
ax.set_yticklabels(top_names[::-1], fontsize=9)
ax.set_xlabel('mean |SHAP value|')
ax.set_title(f'항목 중요도 상위 {n_top}개', fontsize=11)
ax.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig)
plt.close()

with st.expander('📖 이 차트 읽는 법'):
    st.markdown('''
- **막대가 길수록** 해당 항목이 전체 예측에 더 큰 영향을 줍니다.
- **빨간색**: 등급 미상 항목 — 가장 강하게 위험도를 높이는 요소입니다.
- **보라색**: 게임명 텍스트(FastText) — 게임명의 언어·의미 패턴 전체를 합산한 값입니다.
- **파란색**: 나머지 항목 (등급·제조사·언어 등).
    ''')

st.divider()

# ── ② Local SHAP ──────────────────────────────────────────────
st.subheader('② Local  개별 게임 탐지 근거')
st.caption('특정 게임이 왜 이 확률을 받았는지 항목별로 분해합니다.')

game_options = result['game_name'].tolist()
sel_game = st.selectbox(
    '게임 선택',
    game_options,
    format_func=lambda n: (
        f'{n}  (확률: {result.loc[result["game_name"]==n, "재분류_확률"].values[0]:.4f})'
    )
)

# ── 인덱스 버그 수정: pandas label → integer position ──────────
game_row = result[result['game_name'] == sel_game]
if len(game_row) == 0:
    st.error('해당 게임을 찾을 수 없습니다.')
    st.stop()

game_label_idx = game_row.index[0]

# df_feat에서 같은 게임명으로 위치 탐색
if 'game_name' in df_feat.columns:
    match = df_feat[df_feat['game_name'] == sel_game]
    if len(match) > 0:
        feat_label_idx = match.index[0]
    else:
        feat_label_idx = game_label_idx
else:
    feat_label_idx = game_label_idx

# label → integer position 변환 (버그 수정 핵심)
try:
    row_pos = df_feat.index.get_loc(feat_label_idx)
except KeyError:
    row_pos = 0

row_sv   = sv[row_pos]
row_prob = float(game_row['재분류_확률'].values[0])

# ── 위험도 뱃지 + Threshold 맥락 표시 ─────────────────────────
level = risk_label(row_prob, thr)
color = risk_color(level)

col_badge, col_thr = st.columns([2, 3])
with col_badge:
    st.markdown(
        f'<span style="background:{color};color:white;padding:4px 14px;'
        f'border-radius:4px;font-weight:bold;font-size:15px">{level}</span>'
        f'&nbsp;&nbsp;재분류 확률: <b>{row_prob:.4f}</b>',
        unsafe_allow_html=True
    )
with col_thr:
    if row_prob >= thr:
        st.success(f'✅ 현재 기준값({thr}) 기준 → **검토 대상**입니다.')
    else:
        st.info(f'ℹ️ 현재 기준값({thr}) 기준 → **검토 대상이 아닙니다.** (기준값을 낮추면 포함됩니다)')

st.markdown('')

# ── Local SHAP 차트 ───────────────────────────────────────────
row_sv_agg, labels_local = aggregate_shap(row_sv, raw_feature_names)

top_n_local   = 10
top_local_idx = np.argsort(np.abs(row_sv_agg))[::-1][:top_n_local]
local_names   = [labels_local[i] for i in top_local_idx]
local_vals    = row_sv_agg[top_local_idx]

fig2, ax2 = plt.subplots(figsize=(8, top_n_local * 0.5 + 1))
bar_colors = ['#C00000' if v > 0 else '#2E75B6' for v in local_vals[::-1]]
ax2.barh(range(top_n_local), local_vals[::-1], color=bar_colors)
ax2.set_yticks(range(top_n_local))
ax2.set_yticklabels(local_names[::-1], fontsize=9)
ax2.axvline(0, color='#444', linewidth=0.8)
ax2.set_xlabel('SHAP value  (빨간색=위험 방향 ▲  /  파란색=정상 방향 ▼)')
ax2.set_title(f'{sel_game} — 탐지 근거', fontsize=11)
ax2.spines[['top', 'right']].set_visible(False)
plt.tight_layout()
st.pyplot(fig2)
plt.close()

with st.expander('📖 이 차트 읽는 법'):
    st.markdown('''
- **빨간색 막대**: 재분류 가능성을 **높이는** 항목
- **파란색 막대**: 재분류 가능성을 **낮추는** 항목
- 막대가 길수록 해당 항목의 영향이 큽니다.
    ''')

# ── 탐지 사유 자동 요약 (한국어 이름 사용) ───────────────────
pos_feats = [(local_names[i], local_vals[i])
             for i in range(len(local_vals)) if local_vals[i] > 0.01]
neg_feats = [(local_names[i], local_vals[i])
             for i in range(len(local_vals)) if local_vals[i] < -0.01]

if pos_feats:
    top3 = ', '.join([f[0] for f in pos_feats[:3]])
    if level == '고위험':
        st.error(f'**주요 탐지 사유**: {top3}')
    elif level == '검토 대상':
        st.warning(f'**주요 탐지 사유**: {top3}')
    else:
        st.info(f'**주요 영향 항목**: {top3}')

if neg_feats:
    neg3 = ', '.join([f[0] for f in neg_feats[:2]])
    st.caption(f'위험도를 낮추는 항목: {neg3}')

st.divider()

# ── 규칙 기반 보정 정보 ────────────────────────────────────────
st.subheader('③ 규칙 기반 보정 내역')
st.caption('모델 점수 외에 게임명·장르 규칙으로 추가 보정된 항목이 있으면 표시됩니다.')

game_name_str = sel_game
genre_val = df_feat.loc[feat_label_idx, 'genre'] if 'genre' in df_feat.columns else ''

boosts = []

# 도박 키워드 감지
if GAMBLING_PATTERN.search(game_name_str):
    grade_val = df_feat.loc[feat_label_idx, 'grade'] if 'grade' in df_feat.columns else ''
    if str(grade_val) != '청소년이용불가':
        boosts.append(('⚠️ 도박 관련 키워드 감지', f'게임명에 도박·사행 키워드가 포함되어 확률 +0.20 보정 적용'))

# 장르 베팅성
if str(genre_val) in BETTING_GENRES:
    boosts.append(('⚠️ 베팅성 장르 감지', f'장르가 보드게임(베팅성)으로 확률 +0.05 보정 적용'))

if boosts:
    for icon_label, desc in boosts:
        st.warning(f'**{icon_label}**\n\n{desc}')
    st.caption('※ 위 보정은 SHAP 차트에 직접 반영되지 않으며, 최종 확률에 더해진 별도 규칙입니다.')
else:
    st.success('규칙 기반 보정 없음 — 위 확률은 모델 예측값 그대로입니다.')

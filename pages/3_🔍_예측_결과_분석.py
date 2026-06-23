import streamlit as st
import pandas as pd
import numpy as np
import shap
import plotly.graph_objects as go

st.set_page_config(page_title='탐지 이유 분석', page_icon='🔍', layout='wide')

# ── 색상 팔레트 ───────────────────────────────────────────────
C_RISK    = '#E63946'
C_SAFE    = '#457B9D'
C_MISSING = '#C1121F'
C_FT      = '#7B2D8B'
C_NEUTRAL = '#5C6D7E'

def _base_layout(**kwargs):
    d = dict(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(family='sans-serif', size=12, color='#1C1C1E'),
        margin=dict(l=10, r=30, t=44, b=40),
        hoverlabel=dict(bgcolor='white', font_size=12),
    )
    d.update(kwargs)
    return d

_NO_TOOLBAR = {'displayModeBar': False}

# ── 헤더 ──────────────────────────────────────────────────────
st.title('🔍 예측 결과 분석')
st.caption('모델이 각 게임에 확률을 부여한 근거를 다양한 시각 자료로 보여줍니다.')

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
def _get_explainer(_clf):
    return shap.TreeExplainer(_clf)

with st.spinner('분석 데이터 준비 중... (처음 한 번만 시간이 걸립니다)'):
    explainer   = _get_explainer(clf)
    shap_values = explainer.shap_values(X_trans)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values
raw_names = list(prep.get_feature_names_out())

_ev = explainer.expected_value
base_val = float(
    _ev[1] if (hasattr(_ev, '__len__') and len(_ev) > 1)
    else (_ev[0] if hasattr(_ev, '__len__') else _ev)
)

# ── 피처 이름 변환 ───────────────────────────────────────────
_GL = {'전체이용가': '전체이용가', '12세이용가': '12세이용가',
       '15세이용가': '15세이용가', '청소년이용불가': '청소년이용불가', '미상': '미상 ★'}
_LL = {'한국어': '한국어', '영어': '영어', '혼합': '한+영 혼합', '기타': '기타 언어'}

def _pretty(raw: str) -> str:
    if raw.startswith('remainder__') or raw.startswith('num__'): return '__ft__'
    n = raw.replace('cat__', '', 1)
    if n.startswith('grade__'):
        return f'등급: {_GL.get(n[7:], n[7:])}'
    if n.startswith('language_type__'):
        return f'게임명 언어: {_LL.get(n[15:], n[15:])}'
    if n.startswith('company__'):
        return f'제조사: {n[9:]}'
    if n.startswith('grade_company__'):
        parts = n[15:].split('_', 1)
        if len(parts) == 2:
            return f'등급+제조사: {_GL.get(parts[0], parts[0])} / {parts[1]}'
        return f'등급+제조사: {n[15:]}'
    return n

def _short(n: str) -> str:
    return (n.replace('등급: ', '').replace('제조사: ', '')
             .replace('게임명 언어: ', '').replace('등급+제조사: ', '')
             .replace('게임명 텍스트 (FastText)', '게임명').replace(' ★', '★'))

def _color(name: str) -> str:
    if '미상' in name: return C_MISSING
    if 'FastText' in name or '게임명 텍스트' in name: return C_FT
    return C_NEUTRAL

# ── ft 차원 집계 ─────────────────────────────────────────────
def _aggregate(arr, rnames):
    is2d = arr.ndim == 2
    ft  = [i for i, n in enumerate(rnames) if _pretty(n) == '__ft__']
    oth = [i for i, n in enumerate(rnames) if _pretty(n) != '__ft__']
    sv_oth = arr[:, oth] if is2d else arr[oth]
    ft_agg = (arr[:, ft].sum(axis=1, keepdims=True) if is2d else np.array([arr[ft].sum()])) if ft else None
    labels = [_pretty(rnames[i]) for i in oth]
    if ft_agg is not None:
        out = np.hstack([sv_oth, ft_agg]) if is2d else np.append(sv_oth, ft_agg)
        labels += ['게임명 텍스트 (FastText)']
    else:
        out = sv_oth
    return out, labels

# 전체 집계 행렬 (한 번만 계산)
sv2d, labels = _aggregate(sv, raw_names)
mean_abs = np.abs(sv2d).mean(axis=0)
feat_rank = np.argsort(mean_abs)[::-1]

# ── 게임 → 행 위치 변환 헬퍼 ─────────────────────────────────
def _row_pos(game_name: str) -> int:
    if 'game_name' in df_feat.columns:
        m = df_feat[df_feat['game_name'] == game_name]
        if len(m) > 0:
            try:
                return df_feat.index.get_loc(m.index[0])
            except KeyError:
                pass
    # fallback: result 기준 순서로 추정
    idx = result[result['game_name'] == game_name].index
    return int(idx[0]) if len(idx) > 0 else 0

st.divider()

# ═══════════════════════════════════════════════════════════════
# ① 전체 게임 패턴
# ═══════════════════════════════════════════════════════════════
st.subheader('① 전체 게임에서 탐지에 영향을 준 항목')
st.caption('모든 게임을 분석한 결과, 어떤 항목들이 재분류 가능성 판단에 가장 큰 영향을 줬는지 보여줍니다.')

tab_rank, tab_dot = st.tabs(['📊 중요도 순위', '🔵 영향 분포'])

with tab_rank:
    n_top_g = st.slider('상위 N개 항목', 5, 20, 12, key='g_n')
    idx_g  = feat_rank[:n_top_g]
    names_g = [labels[i] for i in idx_g][::-1]
    vals_g  = mean_abs[idx_g][::-1]

    fig_rank = go.Figure(go.Bar(
        x=vals_g, y=names_g, orientation='h',
        marker=dict(color=[_color(n) for n in names_g], line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>평균 영향력: %{x:.4f}<extra></extra>',
    ))
    fig_rank.update_layout(
        **_base_layout(height=max(280, n_top_g * 36 + 60)),
        xaxis=dict(title='평균 영향력 (높을수록 탐지에 많이 기여한 항목)', gridcolor='rgba(0,0,0,0.07)', zeroline=False),
        yaxis=dict(tickfont=dict(size=11)), bargap=0.35,
    )
    st.plotly_chart(fig_rank, use_container_width=True, config=_NO_TOOLBAR)
    c1, c2, c3 = st.columns(3)
    c1.markdown(f'<span style="color:{C_MISSING}">■</span> 등급 미상 관련', unsafe_allow_html=True)
    c2.markdown(f'<span style="color:{C_FT}">■</span> 게임명 텍스트 (FastText)', unsafe_allow_html=True)
    c3.markdown(f'<span style="color:{C_NEUTRAL}">■</span> 등급 · 제조사 · 언어 등', unsafe_allow_html=True)

with tab_dot:
    st.caption(
        '각 **점 하나가 게임 하나**입니다. '
        '점이 오른쪽에 있을수록 그 항목이 해당 게임의 위험도를 높인 것이고, '
        '왼쪽은 반대입니다. 점이 촘촘히 몰릴수록 많은 게임에 그 항목이 영향을 줬습니다.'
    )
    n_dot = st.slider('표시할 항목 수', 4, 12, 8, key='dot_n')
    rng = np.random.default_rng(42)
    n_games = len(sv2d)
    samp = rng.choice(n_games, min(n_games, 300), replace=False)

    fig_dot = go.Figure()
    for rank_i, feat_i in enumerate(feat_rank[:n_dot][::-1]):
        fname = labels[feat_i]
        xs = sv2d[samp, feat_i]
        ys = rank_i + rng.uniform(-0.28, 0.28, len(samp))
        fig_dot.add_trace(go.Scatter(
            x=xs, y=ys, mode='markers',
            marker=dict(color=[C_RISK if x > 0 else C_SAFE for x in xs],
                        size=5, opacity=0.55, line=dict(width=0)),
            hovertemplate=f'<b>{fname}</b><br>영향도: %{{x:+.4f}}<extra></extra>',
            showlegend=False,
        ))

    fig_dot.add_vline(x=0, line=dict(color='#555', width=1, dash='dot'))
    fig_dot.update_layout(
        **_base_layout(height=max(300, n_dot * 46 + 80)),
        xaxis=dict(title='영향도  (오른쪽 = 위험도 ▲ / 왼쪽 = 위험도 ▼)',
                   gridcolor='rgba(0,0,0,0.07)', zeroline=False),
        yaxis=dict(
            tickvals=list(range(n_dot)),
            ticktext=[labels[feat_rank[i]] for i in range(n_dot)][::-1],
            tickfont=dict(size=11),
        ),
        annotations=[
            dict(x=0.98, y=0.01, xref='paper', yref='paper',
                 text='● 위험 방향', showarrow=False, font=dict(color=C_RISK, size=11), xanchor='right'),
            dict(x=0.02, y=0.01, xref='paper', yref='paper',
                 text='● 안전 방향', showarrow=False, font=dict(color=C_SAFE, size=11), xanchor='left'),
        ],
    )
    st.plotly_chart(fig_dot, use_container_width=True, config=_NO_TOOLBAR)

st.divider()

# ═══════════════════════════════════════════════════════════════
# ② 히트맵
# ═══════════════════════════════════════════════════════════════
st.subheader('② 게임별 탐지 패턴 히트맵')
st.caption(
    '상위 게임들을 항목별로 한눈에 비교합니다. '
    '**빨간색이 짙을수록** 그 항목이 해당 게임의 위험도를 강하게 높인 것이고, '
    '**파란색이 짙을수록** 반대 방향으로 작용한 것입니다.'
)

n_hm = st.slider('표시할 게임 수 (확률 상위순)', 5, 30, 15, key='hm_n')
N_HM_FEAT = 10

# 상위 게임 행 위치 수집
hm_names_found, hm_rows = [], []
for gn in result['game_name'].head(n_hm).tolist():
    pos = _row_pos(gn)
    hm_rows.append(pos)
    hm_names_found.append(gn[:16] + '…' if len(gn) > 16 else gn)

if hm_rows:
    hm_feat_idx = feat_rank[:N_HM_FEAT]
    hm_xlabels  = [_short(labels[i]) for i in hm_feat_idx]
    hm_matrix   = sv2d[hm_rows][:, hm_feat_idx]

    fig_hm = go.Figure(go.Heatmap(
        z=hm_matrix[::-1],
        x=hm_xlabels,
        y=hm_names_found[::-1],
        colorscale=[[0, C_SAFE], [0.5, '#F5F5F5'], [1, C_RISK]],
        zmid=0,
        hovertemplate='<b>%{y}</b><br>항목: %{x}<br>영향도: %{z:+.4f}<extra></extra>',
        colorbar=dict(title='영향도', tickformat='+.2f', len=0.8, thickness=14),
    ))
    fig_hm.update_layout(
        **_base_layout(
            height=max(300, len(hm_rows) * 30 + 110),
            margin=dict(l=130, r=80, t=60, b=30),
        ),
        xaxis=dict(side='top', tickangle=-30, tickfont=dict(size=10)),
        yaxis=dict(tickfont=dict(size=10)),
    )
    st.plotly_chart(fig_hm, use_container_width=True, config=_NO_TOOLBAR)
else:
    st.info('히트맵을 표시하려면 게임이 충분히 있어야 합니다.')

st.divider()

# ═══════════════════════════════════════════════════════════════
# ③ 개별 게임 분석
# ═══════════════════════════════════════════════════════════════
st.subheader('③ 이 게임의 탐지 이유')
st.caption('게임을 선택하면, 모델이 왜 이 확률을 줬는지 항목별로 분해해서 보여줍니다.')

sel_game = st.selectbox(
    '게임 선택',
    result['game_name'].tolist(),
    format_func=lambda n: f'{n}  ·  확률 {result.loc[result["game_name"]==n,"재분류_확률"].values[0]:.4f}',
)

game_row = result[result['game_name'] == sel_game]
if len(game_row) == 0:
    st.error('해당 게임을 찾을 수 없습니다.')
    st.stop()

pos      = _row_pos(sel_game)
row_sv   = sv2d[pos]
row_prob = float(game_row['재분류_확률'].values[0])
level    = risk_label(row_prob, thr)
color    = risk_color(level)

feat_label_idx = (df_feat[df_feat['game_name'] == sel_game].index[0]
                  if 'game_name' in df_feat.columns and len(df_feat[df_feat['game_name'] == sel_game]) > 0
                  else game_row.index[0])
grade_val = str(df_feat.loc[feat_label_idx, 'grade']) if 'grade' in df_feat.columns else '미상'
genre_val = str(df_feat.loc[feat_label_idx, 'genre']) if 'genre' in df_feat.columns else ''

# 위험도 카드
m1, m2, m3 = st.columns(3)
m1.metric('재분류 확률', f'{row_prob:.4f}')
m2.metric('위험 등급', level)
m3.metric('현재 기준값', f'{thr}')

if row_prob >= thr:
    st.success(f'✅ 현재 기준값({thr}) 기준 — **검토 대상**입니다.')
else:
    st.info(f'ℹ️ 현재 기준값({thr}) 기준 — 검토 대상이 아닙니다. (기준값을 낮추면 포함됩니다)')

st.markdown('')

# Local SHAP 준비
top_n  = 10
top_idx = np.argsort(np.abs(row_sv))[::-1][:top_n]
loc_names = [labels[i] for i in top_idx]
loc_vals  = row_sv[top_idx]
loc_names_r = loc_names[::-1]
loc_vals_r  = loc_vals[::-1]

pos_feats = [(loc_names[i], loc_vals[i]) for i in range(len(loc_vals)) if loc_vals[i] > 0.01]
neg_feats = [(loc_names[i], loc_vals[i]) for i in range(len(loc_vals)) if loc_vals[i] < -0.01]

# 규칙 기반 보정 체크 (차트보다 먼저)
boosts = []
if GAMBLING_PATTERN.search(str(sel_game)) and grade_val != '청소년이용불가':
    boosts.append(('도박 관련 키워드 감지', '게임명에 도박·사행 키워드가 포함되어 확률 +0.20 보정 적용'))
if genre_val in BETTING_GENRES:
    boosts.append(('베팅성 장르 감지', '장르가 보드게임(베팅성)으로 확률 +0.05 보정 적용'))

# ── 차트 탭 ──────────────────────────────────────────────────
tab_wf, tab_bar = st.tabs(['📈 단계별 누적 분석', '📊 항목별 영향도'])

with tab_wf:
    st.caption(
        '게임의 기본 점수(전체 평균)에서 시작해서 각 항목이 점수를 얼마씩 더하거나 뺐는지 순서대로 보여줍니다. '
        '최종 탐지 점수가 높을수록 재분류 확률이 높습니다.'
    )
    wf_n   = 7
    wf_idx = np.argsort(np.abs(row_sv))[::-1][:wf_n]
    wf_names = [labels[i] for i in wf_idx]
    wf_vals  = row_sv[wf_idx]
    remainder = row_sv.sum() - wf_vals.sum()

    fig_wf = go.Figure(go.Waterfall(
        orientation='h',
        measure=['absolute'] + ['relative'] * wf_n + ['relative', 'total'],
        y=['기본값 (전체 평균)'] + wf_names + ['기타 요인 합산', '최종 탐지 점수'],
        x=[base_val] + list(wf_vals) + [remainder, 0],
        connector=dict(line=dict(color='rgba(120,120,120,0.25)', width=1)),
        increasing=dict(marker=dict(color=C_RISK)),
        decreasing=dict(marker=dict(color=C_SAFE)),
        totals=dict(marker=dict(color='#444444', line=dict(color='#222', width=1))),
        hovertemplate='<b>%{y}</b><br>영향도: %{x:+.4f}<extra></extra>',
    ))
    fig_wf.update_layout(
        **_base_layout(height=max(320, (wf_n + 3) * 46 + 80)),
        title=dict(text=f'<b>{sel_game}</b> — 단계별 탐지 점수 누적', font=dict(size=13)),
        xaxis=dict(title='탐지 점수', gridcolor='rgba(0,0,0,0.07)', zeroline=False),
        yaxis=dict(tickfont=dict(size=11), autorange='reversed'),
    )
    st.plotly_chart(fig_wf, use_container_width=True, config=_NO_TOOLBAR)
    st.caption('※ 탐지 점수는 내부 계산값(로그 오즈)입니다. 최종 재분류 확률은 이를 0~1 사이로 변환한 값입니다.')

with tab_bar:
    st.caption(
        '**빨간 막대**가 길수록 그 항목이 재분류 가능성을 크게 높인 것입니다. '
        '**파란 막대**는 반대로 가능성을 낮추는 방향으로 작용한 것입니다.'
    )
    fig_bar = go.Figure(go.Bar(
        x=loc_vals_r, y=loc_names_r, orientation='h',
        marker=dict(color=[C_RISK if v > 0 else C_SAFE for v in loc_vals_r], line=dict(width=0)),
        hovertemplate='<b>%{y}</b><br>영향도: %{x:+.4f}<br>%{customdata}<extra></extra>',
        customdata=['위험도 ▲' if v > 0 else '위험도 ▼' for v in loc_vals_r],
    ))
    fig_bar.add_vline(x=0, line=dict(color='#555', width=1))
    fig_bar.update_layout(
        **_base_layout(height=max(320, top_n * 40 + 80)),
        title=dict(text=f'<b>{sel_game}</b> — 항목별 영향도', font=dict(size=13)),
        xaxis=dict(title='영향도', gridcolor='rgba(0,0,0,0.07)', zeroline=False, tickformat='+.3f'),
        yaxis=dict(tickfont=dict(size=11)), bargap=0.3,
    )
    st.plotly_chart(fig_bar, use_container_width=True, config=_NO_TOOLBAR)
    lc1, lc2 = st.columns(2)
    lc1.markdown(f'<span style="color:{C_RISK}">■</span> **빨간색** — 재분류 가능성을 높이는 항목', unsafe_allow_html=True)
    lc2.markdown(f'<span style="color:{C_SAFE}">■</span> **파란색** — 재분류 가능성을 낮추는 항목', unsafe_allow_html=True)

st.divider()

# ── 자동 분석 요약 (Layer 1, 항상 무료) ──────────────────────
def _rule_insight(game, pos_f, neg_f, prob, thr_, boosts_):
    lines = []
    if pos_f:
        n0, v0 = pos_f[0]
        if '미상' in n0:
            lines.append(f"**{game}** 은(는) 등급이 '미상'으로 분류되어 재분류 가능성을 가장 크게 높이는 요인이 됐습니다.")
        elif 'FastText' in n0 or '텍스트' in n0:
            lines.append(f"**{game}** 의 게임명 패턴이 과거 재분류 사례와 유사하게 분석됐습니다.")
        elif '제조사' in n0:
            co = n0.split(': ')[-1] if ': ' in n0 else ''
            lines.append(f"**{game}** 은(는) {'제조사(' + co + ') 이력이' if co else '제조사 이력이'} 재분류 가능성을 높이는 방향으로 작용했습니다.")
        else:
            lines.append(f"**{game}** 은(는) '{n0}' 항목이 재분류 가능성을 가장 크게 높이는 요인으로 분석됩니다.")
    if len(pos_f) >= 2:
        n1 = pos_f[1][0]
        if n1 != (pos_f[0][0] if pos_f else ''):
            lines.append(f"'{n1}' 항목도 위험도를 높이는 방향으로 추가 작용했습니다.")
    if boosts_:
        lines.append("게임명에 도박·사행 관련 키워드가 감지되어 규칙 기반 보정이 추가 적용됐습니다.")
    if neg_f and prob < 0.80:
        lines.append(f"다만 '{neg_f[0][0]}' 항목은 위험도를 낮추는 방향으로 작용했습니다.")
    if not lines:
        lines.append(f"탐지에 영향을 준 항목들이 복합적으로 작용하여 최종 확률 {prob:.0%}가 산출됐습니다.")
    return ' '.join(lines[:3])

st.markdown('**💡 자동 분석 요약**')
auto_txt = _rule_insight(sel_game, pos_feats, neg_feats, row_prob, thr, boosts)
if level == '고위험':
    st.error(auto_txt)
elif level == '검토 대상':
    st.warning(auto_txt)
else:
    st.info(auto_txt)

# ── AI 심층 분석 (Layer 2 — Gemini) ─────────────────────────
gemini_key = st.secrets.get('GEMINI_API_KEY', '')

if not gemini_key:
    st.caption('🔑 AI 심층 분석 기능을 사용하려면 Streamlit Secrets에 GEMINI_API_KEY를 등록해 주세요.')
elif row_prob >= thr:
    cache_key = f'gem_{sel_game}'

    if cache_key not in st.session_state:
        if st.button('✨ AI 심층 분석', type='secondary'):
            with st.spinner('분석 중...'):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=gemini_key)
                    mdl = genai.GenerativeModel(
                        'gemini-1.5-flash',
                        generation_config=genai.GenerationConfig(
                            max_output_tokens=100,
                            temperature=0.2,
                        ),
                    )
                    top2 = ','.join(f'{n}({v:+.2f})' for n, v in pos_feats[:2])
                    boost_str = boosts[0][0] if boosts else ''
                    prompt = (
                        f"게임:{sel_game} 등급:{grade_val} 확률:{row_prob:.0%}\n"
                        f"탐지주요인:{top2}"
                        + (f" 추가보정:{boost_str}" if boost_str else '')
                        + "\n쉬운 한국어 2문장. 전문용어 금지."
                    )
                    resp = mdl.generate_content(prompt)
                    st.session_state[cache_key] = ('ok', resp.text.strip())
                except Exception as e:
                    # 오류 종류를 저장해 진단에 활용, 사용자에게는 부드럽게 표시
                    st.session_state[cache_key] = ('err', str(e))

    cached = st.session_state.get(cache_key)
    if cached:
        status, content = cached
        if status == 'ok' and content:
            st.markdown('**🤖 AI 심층 분석**')
            st.success(content)
        elif status == 'err':
            # API 키 미등록 / 할당량 초과 / 네트워크 오류 등 구분 안내
            if 'API_KEY' in content or 'api_key' in content or 'invalid' in content.lower():
                st.caption('🔑 API 키가 올바르지 않습니다. Streamlit Secrets의 GEMINI_API_KEY를 확인해 주세요.')
            elif '429' in content or 'quota' in content.lower() or 'exhausted' in content.lower():
                st.caption('📊 오늘 AI 분석 횟수를 모두 사용했습니다. 내일 다시 이용하거나 위 자동 요약을 참고해 주세요.')
            else:
                st.caption(f'⚠️ 오류 내용 (임시 진단용): {content}')

st.divider()

# ═══════════════════════════════════════════════════════════════
# ④ 규칙 기반 보정 내역
# ═══════════════════════════════════════════════════════════════
st.subheader('④ 규칙 기반 보정 내역')
st.caption('모델 점수 외에 게임명·장르 규칙으로 추가 보정된 항목이 있으면 표시됩니다.')

if boosts:
    for label, desc in boosts:
        st.warning(f'**⚠️ {label}** — {desc}')
    st.caption('※ 위 보정은 차트에 직접 반영되지 않으며, 최종 확률에 더해진 별도 규칙입니다.')
else:
    st.success('규칙 기반 보정 없음 — 위 확률은 모델 예측값 그대로입니다.')

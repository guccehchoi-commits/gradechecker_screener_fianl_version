import streamlit as st
import pandas as pd
import numpy as np
from utils.model import risk_label, risk_color
from utils.preprocess import is_fortnite
st.set_page_config(page_title='예측 결과', page_icon='📊', layout='wide')
from utils.logo import render_logo
render_logo()
st.title('📊 예측 결과')
# ── 데이터 확인 ────────────────────────────────────────────────
if 'result' not in st.session_state:
    st.warning('먼저 **① 파일 업로드** 페이지에서 파일을 업로드하고 분석을 실행해 주세요.')
    st.stop()
result   = st.session_state['result'].copy()
filename = st.session_state.get('filename', '')
thr      = st.session_state.get('global_thr', 0.40)

st.caption(f'파일: {filename}  |  전체 {len(result):,}건  |  현재 기준값: {thr}')
st.divider()

# ── 탐지 건수 실시간 요약 ──────────────────────────────────────
n_total  = len(result)
n_high   = (result['재분류_확률'] >= 0.80).sum()
n_target = ((result['재분류_확률'] >= thr) & (result['재분류_확률'] < 0.80)).sum()
n_safe   = (result['재분류_확률'] < thr).sum()
pct      = (n_high + n_target) / n_total * 100 if n_total else 0

st.info(
    f'**기준값 {thr} 적용 시 → 검토 대상 {n_high + n_target:,}건** '
    f'(전체의 {pct:.1f}%)  |  '
    f'고위험 {n_high:,}건  +  검토 대상 {n_target:,}건  |  이상 없음 {n_safe:,}건'
)

# ── KPI 카드 ──────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
c1.metric('전체 게임',                   f'{n_total:,}건')
c2.metric(f'고위험  ≥0.80',             f'{n_high:,}건',   delta='즉시 검토 필요', delta_color='inverse')
c3.metric(f'검토 대상  {thr}~0.80',     f'{n_target:,}건', delta='우선 검토',      delta_color='off')
c4.metric(f'이상 없음  <{thr}',          f'{n_safe:,}건',   delta='모니터링 유지',  delta_color='off')
st.divider()

# ── 위험도 컬럼 (Threshold 연동) ───────────────────────────────
result['위험도'] = result['재분류_확률'].apply(lambda p: risk_label(p, thr))

# ── 필터 ──────────────────────────────────────────────────────
col_f1, col_f2 = st.columns([1, 1])
with col_f1:
    view_mode = st.radio(
        '표시 범위',
        ['전체', '고위험만', '검토 대상만', '이상 없음만'],
        horizontal=True
    )
with col_f2:
    if 'grade' in result.columns:
        grades = ['전체'] + sorted(result['grade'].dropna().unique().tolist())
        sel_grade = st.selectbox('등급 필터', grades)
    else:
        sel_grade = '전체'

# 필터 적용
df_view = result.copy()
if view_mode == '고위험만':
    df_view = df_view[df_view['재분류_확률'] >= 0.80]
elif view_mode == '검토 대상만':
    df_view = df_view[(df_view['재분류_확률'] >= thr) & (df_view['재분류_확률'] < 0.80)]
elif view_mode == '이상 없음만':
    df_view = df_view[df_view['재분류_확률'] < thr]
if sel_grade != '전체' and 'grade' in df_view.columns:
    df_view = df_view[df_view['grade'] == sel_grade]

st.caption(f'현재 표시: {len(df_view):,}건')

# ── 결과 테이블 ────────────────────────────────────────────────
def highlight_risk(row):
    p = row['재분류_확률']
    if p >= 0.80:  return ['background-color: #FFEBEB'] * len(row)
    elif p >= thr: return ['background-color: #FFF0E0'] * len(row)
    else:          return [''] * len(row)

def fmt_prob(v):
    return f'{v:.4f}'

display_cols = ['game_name', 'grade', 'company', 'genre', '재분류_확률', '위험도']
display_cols = [c for c in display_cols if c in df_view.columns or c in ['재분류_확률', '위험도']]
col_rename = {
    'game_name': '게임명', 'grade': '등급',
    'company': '플랫폼', 'genre': '장르',
}
df_display = df_view[display_cols].rename(columns=col_rename)
styled = (
    df_display.style
    .apply(highlight_risk, axis=1)
    .format({'재분류_확률': fmt_prob})
)
st.dataframe(styled, use_container_width=True, height=420)

# ── 포트나이트 주의 문구 ───────────────────────────────────────
if 'game_name' in df_view.columns:
    ft_rows = df_view[df_view['game_name'].apply(is_fortnite)]
    if len(ft_rows) > 0:
        st.warning(
            f'⚠️  **포트나이트 관련 게임 {len(ft_rows)}건** — 학습 데이터에 이상치 이력이 없어 예측 신뢰도가 낮습니다.',
            icon='⚠️'
        )
st.divider()

# ── 다운로드 ──────────────────────────────────────────────────
df_dl = df_view.rename(columns=col_rename)
stem  = filename.rsplit('.', 1)[0] if '.' in filename else filename
col_dl1, col_dl2 = st.columns(2)
with col_dl1:
    csv = df_dl.to_csv(index=True, encoding='utf-8-sig')
    st.download_button(
        label='⬇  CSV 다운로드',
        data=csv,
        file_name=f'이상탐지_결과_{stem}.csv',
        mime='text/csv',
        type='primary',
        use_container_width=True,
    )
with col_dl2:
    import io
    buf = io.BytesIO()
    df_dl.to_excel(buf, index=True, engine='openpyxl')
    st.download_button(
        label='⬇  Excel 다운로드',
        data=buf.getvalue(),
        file_name=f'이상탐지_결과_{stem}.xlsx',
        mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        type='primary',
        use_container_width=True,
    )

from utils.footer import render_footer
render_footer()

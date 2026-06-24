import streamlit as st

st.set_page_config(
    page_title='GUCC GRADE CHECKER - 게임물 등급재분류 가능성 예측 서비스',
    page_icon='🎯',
    layout='wide',
    initial_sidebar_state='expanded',
)

from utils.logo import render_logo
render_logo()

# ── 사이드바 ───────────────────────────────────────────────────
st.sidebar.info(
    '**사용 순서**\n\n'
    '1️⃣  파일 업로드\n\n'
    '2️⃣  예측 결과 확인\n\n'
    '3️⃣  예측 결과 분석 (탐지 근거)'
)
st.sidebar.divider()

# ── 탐지 민감도 설정 (전역 Threshold) ─────────────────────────
st.sidebar.subheader('⚙️ 탐지 민감도 설정')
st.sidebar.caption('값을 낮추면 더 많은 게임을 검토 대상으로 잡아냅니다.')

thr = st.sidebar.slider(
    '기준값 (Threshold)',
    min_value=0.20, max_value=0.80, value=0.40, step=0.05,
    key='global_thr',
)

# 민감도에 따른 설명 텍스트
if thr <= 0.30:
    st.sidebar.info(
        '**폭넓게 탐지**\n\n'
        '의심스러운 게임을 최대한 잡아냅니다. '
        '검토할 게임 수가 많아집니다.'
    )
elif thr <= 0.45:
    st.sidebar.success(
        '**권장 설정 ★**\n\n'
        '탐지 범위와 정확도의 균형입니다.\n'
        '실제 재분류 게임의 약 **73%** 를 잡아냅니다.'
    )
elif thr <= 0.65:
    st.sidebar.warning(
        '**정밀 탐지**\n\n'
        '확실한 게임만 포함합니다. '
        '경계선 게임을 놓칠 수 있습니다.'
    )
else:
    st.sidebar.error(
        '**고정밀 모드**\n\n'
        '매우 확실한 게임만 탐지합니다. '
        '탐지율이 크게 낮아집니다.'
    )

# Threshold별 성능 힌트 (사전 계산값)
_perf = {
    0.20: ('~85%', '~55%'),
    0.25: ('~82%', '~60%'),
    0.30: ('~80%', '~65%'),
    0.35: ('~77%', '~70%'),
    0.40: ('72.9%', '73.8%'),
    0.45: ('~70%', '~77%'),
    0.50: ('66.7%', '83.9%'),
    0.55: ('~62%', '~87%'),
    0.60: ('~57%', '~90%'),
    0.65: ('~52%', '~92%'),
    0.70: ('~47%', '~94%'),
    0.75: ('~44%', '~95%'),
    0.80: ('41.7%', '95.7%'),
}
_rec, _pre = _perf.get(round(thr, 2), ('—', '—'))
st.sidebar.caption(f'예상 재현율 {_rec}  /  정밀도 {_pre}')

st.sidebar.divider()
st.sidebar.caption('※ 예측 결과는 우선 재검토 순위 제공 목적\n최종 판정은 전문가 검토 필수')

# ── 홈 화면 ───────────────────────────────────────────────────
st.title('🎯 GRADE CHECKER')
st.markdown('<p style="font-size:1.15rem;">게임물 데이터를 업로드하면 등급재분류 확률이 높은 게임을 자동으로 판별하는 <strong>게임물 등급재분류 가능성 AI 예측 서비스</strong>입니다.</p>', unsafe_allow_html=True)
# st.divider()

# col1, col2, col3 = st.columns(3)
# with col1:
#     st.metric('모델 AUC', '0.9906', help='A그룹 외부 검증 기준')
# with col2:
#     st.metric('F1 이상 탐지', '0.9261', help='threshold 0.40 채택 (FN 최소화)')
# with col3:
#     st.metric('채택 Threshold', '0.40', help='신규 1,420건 검증, 재현율 72.9%')

st.divider()

st.subheader('시작하려면 왼쪽 메뉴에서 **① 파일 업로드**를 클릭하세요.')

with st.expander('💡 입력 파일 형식 안내'):
    st.markdown('''
| 컬럼명 | 필수 여부 | 설명 |
|---|---|---|
| `game_name` | ✅ 필수 | 게임 타이틀 (텍스트 분석에 사용) |
| `grade` | ✅ 필수 | 등급 표기 — 미입력 시 최고위험 자동 분류 |
| `company` | 선택 | 플랫폼명 — 예측 보조 반영 |
| `genre` | 선택 | 장르 — 결제 유도 요소 포함 장르 가중치 적용 |
    ''')

with st.expander('❓ 탐지 민감도(Threshold)란?'):
    st.markdown('''
모델은 각 게임에 **0~1 사이의 재분류 확률**을 부여합니다.
Threshold는 이 확률 중 어디서부터 "검토 대상"으로 볼지 정하는 기준선입니다.

| 기준값 | 효과 |
|---|---|
| 낮게 설정 (예: 0.30) | 더 많은 게임을 검토 대상에 포함 — 놓치는 게임이 적지만 검토 부담 증가 |
| 권장값 0.40 ★ | 탐지율과 정확도의 균형점 |
| 높게 설정 (예: 0.60 이상) | 확실한 게임만 포함 — 검토 부담은 줄지만 경계선 게임을 놓칠 수 있음 |

왼쪽 사이드바의 슬라이더로 언제든지 조정할 수 있으며,
변경 사항은 **예측 결과**와 **예측 결과 분석** 전체에 즉시 반영됩니다.
    ''')

from utils.footer import render_footer
render_footer()

import streamlit as st

st.set_page_config(
    page_title='GUCC 이상탐지 대시보드',
    page_icon='🎮',
    layout='wide',
    initial_sidebar_state='expanded',
)

# ── 사이드바 ───────────────────────────────────────────────────
st.sidebar.title('🎮 GUCC')
st.sidebar.caption('게임 등급재분류 이상탐지 대시보드')
st.sidebar.divider()
st.sidebar.info(
    '**사용 순서**\n\n'
    '1️⃣  파일 업로드\n\n'
    '2️⃣  예측 결과 확인\n\n'
    '3️⃣  SHAP 분석 (탐지 근거)'
)
st.sidebar.divider()
st.sidebar.caption('※ 예측 결과는 우선 재검토 순위 제공 목적\n최종 판정은 전문가 검토 필수')

# ── 홈 화면 ───────────────────────────────────────────────────
st.title('🎮 게임 등급재분류 이상탐지 대시보드')
st.markdown('게임물 데이터를 업로드하면 **재분류 가능성이 높은 게임을 자동으로 순위화**합니다.')
st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric('모델 AUC', '0.9906', help='A그룹 외부 검증 기준')
with col2:
    st.metric('F1 이상 탐지', '0.9261', help='threshold 0.40 채택 (FN 최소화)')
with col3:
    st.metric('채택 Threshold', '0.40', help='신규 1,420건 검증, 재현율 72.9%')

st.divider()

st.subheader('시작하려면 왼쪽 메뉴에서 **① 파일 업로드**를 클릭하세요.')

with st.expander('💡 입력 파일 형식 안내'):
    st.markdown('''
| 컬럼명 | 필수 여부 | 설명 |
|---|---|---|
| `game_name` | ✅ 필수 | 게임 타이틀 (텍스트 분석에 사용) |
| `grade` | ✅ 필수 | 등급 표기 — 미입력 시 최고위험 자동 분류 |
| `company` | 선택 | 플랫폼명 — 예측 보조 반영 |
| `genre` | 선택 | 장르 — 보드게임(베팅성) 위험 가중치 적용 |
    ''')

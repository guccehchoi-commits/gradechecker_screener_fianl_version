import streamlit as st
import pandas as pd
from utils.preprocess import run_preprocess, load_fasttext, apply_genre_boost, apply_gambling_keyword_boost
from utils.model import load_model, predict
st.set_page_config(page_title='파일 업로드', page_icon='📂', layout='wide')
st.logo('gucc_logo.png', size='large', icon_image='gucc_logo.png')
st.title('📂 파일 업로드')
st.caption('분석할 게임물 파일을 업로드하세요. Excel·CSV 모두 지원하며 컬럼명을 자동으로 감지합니다.')
st.divider()
# ── 컬럼 별칭 (자동 탐지용) ────────────────────────────────────
COLUMN_ALIASES = {
    'game_name': ['game_name', 'gamename', 'game_title', '게임명', '게임이름', '타이틀', 'title', 'name'],
    'grade':     ['grade', '등급', '이용등급', '게임등급', 'rating', '자체등급', 'game_grade', '자체등급분류'],
    'company':   ['company', '유통사', '회사', '플랫폼', 'publisher', 'distributor', 'platform', '신청사', '사업자명'],
    'genre':     ['genre', '장르', 'category', '게임장르', 'game_genre', '게임유형'],
}
LABELS = {
    'game_name': '게임명 (game_name)  ★필수',
    'grade':     '등급 (grade)  ★필수',
    'company':   '유통사/플랫폼 (company)  선택',
    'genre':     '장르 (genre)  선택',
}
def auto_detect(df_cols: list[str], aliases: list[str]) -> str | None:
    lower_map = {c.lower().strip(): c for c in df_cols}
    for alias in aliases:
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]
    return None
# ── 파일 업로드 ────────────────────────────────────────────────
uploaded = st.file_uploader(
    '파일 선택 (.xlsx / .xls / .csv)',
    type=['xlsx', 'xls', 'csv'],
)
if uploaded is None:
    st.info('파일을 업로드하면 컬럼을 자동으로 감지한 뒤 전처리 → 예측이 실행됩니다.')
    st.stop()
# ── 파일 읽기 ──────────────────────────────────────────────────
try:
    if uploaded.name.lower().endswith('.csv'):
        try:
            df_raw = pd.read_csv(uploaded, encoding='utf-8-sig')
        except UnicodeDecodeError:
            uploaded.seek(0)
            df_raw = pd.read_csv(uploaded, encoding='cp949')
    else:
        df_raw = pd.read_excel(uploaded)
except Exception as e:
    st.error(f'파일을 읽을 수 없습니다: {e}')
    st.stop()
st.success(f'업로드 완료 — **{len(df_raw):,}건** / 컬럼 {len(df_raw.columns)}개')
with st.expander('📄 업로드 데이터 미리보기 (상위 5건)'):
    st.dataframe(df_raw.head(), use_container_width=True)
st.divider()
# ── 컬럼 자동 감지 + 매핑 UI ────────────────────────────────────
st.subheader('🔗 컬럼 매핑')
st.caption('자동으로 감지한 컬럼입니다. 잘못 감지됐으면 직접 선택해 주세요. 없는 컬럼은 **"— 없음 —"** 으로 두면 됩니다.')
cols_all = list(df_raw.columns)
none_opt = '— 없음 —'
options  = [none_opt] + cols_all
detected = {k: auto_detect(cols_all, v) for k, v in COLUMN_ALIASES.items()}
mapping_ui = st.columns(4)
col_mapping: dict[str, str | None] = {}
for i, (key, label) in enumerate(LABELS.items()):
    default = detected[key]
    default_idx = options.index(default) if default in options else 0
    with mapping_ui[i]:
        chosen = st.selectbox(label, options=options, index=default_idx, key=f'map_{key}')
        col_mapping[key] = None if chosen == none_opt else chosen
# 감지 요약
auto_found = sum(1 for v in col_mapping.values() if v is not None)
st.caption(f'감지된 컬럼: {auto_found} / 4개')
st.divider()
# ── 분석 시작 ──────────────────────────────────────────────────
if st.button('🔍 분석 시작', type='primary', use_container_width=True):
    df_work = pd.DataFrame(index=df_raw.index)
    # game_name — 없으면 행 번호로 대체
    gn = col_mapping['game_name']
    if gn:
        df_work['game_name'] = df_raw[gn].astype(str)
    else:
        df_work['game_name'] = [f'게임_{i+1}' for i in range(len(df_raw))]
        st.warning('게임명 컬럼이 지정되지 않아 행 번호(게임_1, 게임_2 …)로 대체합니다.')
    # grade — 없으면 전부 NaN → run_preprocess 에서 '미상' 처리
    gr = col_mapping['grade']
    if gr:
        df_work['grade'] = df_raw[gr]
    else:
        df_work['grade'] = None
        st.warning('⚠️ 등급(grade) 컬럼이 없어 전체를 **미상(최고위험군)**으로 처리합니다.')
    # 선택 컬럼
    for key in ('company', 'genre'):
        mapped = col_mapping[key]
        if mapped:
            df_work[key] = df_raw[mapped].values
    # 나머지 원본 컬럼 병합 (결과 표시용)
    for c in cols_all:
        if c not in df_work.columns:
            df_work[c] = df_raw[c].values
    # 등급 미입력 부분 경고
    n_miss = df_work['grade'].isna().sum()
    if 0 < n_miss < len(df_work):
        st.warning(f'⚠️ 등급 미입력 {n_miss}건 → **미상(최고위험군)** 자동 처리')
    with st.spinner('FastText 모델 로드 중...'):
        ft_model = load_fasttext()
    with st.spinner('전처리 중...'):
        df_feat = run_preprocess(df_work, ft_model)
    with st.spinner('예측 중...'):
        clf, prep = load_model()
        prob = predict(df_feat, clf, prep)
    prob = apply_genre_boost(prob, df_work)
    prob = apply_gambling_keyword_boost(prob, df_feat)
    # 결과 데이터프레임 구성
    result = pd.DataFrame({'game_name': df_work['game_name']})
    if 'grade'   in df_work.columns: result['grade']   = df_feat['grade']
    if 'company' in df_work.columns: result['company'] = df_feat['company']
    if 'genre'   in df_work.columns: result['genre']   = df_work['genre']
    result['재분류_확률'] = prob.round(4)
    result = result.sort_values('재분류_확률', ascending=False).reset_index(drop=True)
    result.index += 1
    st.session_state.update({
        'result':   result,
        'prob':     prob,
        'df_feat':  df_feat,
        'clf':      clf,
        'prep':     prep,
        'filename': uploaded.name,
    })
    st.success('✅ 분석 완료! 왼쪽 메뉴에서 **② 예측 결과**를 확인하세요.')
    st.balloons()

    # ── Google Sheets 저장 ─────────────────────────────────────
    try:
        import requests, datetime
        url = st.secrets.get('SHEETS_URL', '')
        if url:
            from utils.model import risk_label
            now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            payload = [
                {
                    'game_name': str(row['game_name']),
                    'grade':     str(row.get('grade', '')),
                    'company':   str(row.get('company', '')),
                    'prob':      float(row['재분류_확률']),
                    'risk':      risk_label(float(row['재분류_확률'])),
                    'uploaded_at': now,
                }
                for _, row in result.iterrows()
            ]
            requests.post(url, json=payload, timeout=10)
    except Exception:
        pass
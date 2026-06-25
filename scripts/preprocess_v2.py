"""
[STEP 2] 2차 전처리 — B그룹 피처 생성 (모델링용 최종 데이터셋)
=================================================================
입력:
  - B그룹_데이터셋_260618_v2.xlsx  (grade_missing 제거 버전)

처리 내용:
  - language_type 파생: game_name에서 한국어/영어/혼합/기타 분류
  - grade×company 교차 피처: 50건 이상 조합만 유효 처리 (과적합 방지)
  - FastText 임베딩: game_name을 50차원 벡터로 변환
    (유사 키워드·신규 표현 대응 — slot≈slots, C4sh≈cash 등)

출력:
  - B그룹_데이터셋_260619_v3.xlsx  (15,567건, 피처 57개)
  - fasttext_gamename.model         (임베딩 모델, STEP 7·9에서 재사용)
"""
import re
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import FastText

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
MODEL_DIR = Path(r'C:\Users\User\python')

IN_B     = DATA_DIR / 'B그룹_데이터셋_260618_v2.xlsx'
OUT_B    = DATA_DIR / 'B그룹_데이터셋_260619_v3.xlsx'
MODEL_PATH = MODEL_DIR / 'fasttext_gamename.model'

# grade×company 교차 피처에서 50건 미만 조합은 '기타'로 처리
# → 희소한 조합을 별도 피처로 쓰면 과적합 위험, 일반화 성능 하락
CROSS_THRESHOLD = 50
# FastText 임베딩 벡터 차원 수 (50차원 = 정보량과 계산비용의 균형점)
FT_DIM = 50

# ════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════
print('=' * 60)
print('B그룹 v2 로드')
df = pd.read_excel(IN_B)
print(f'로드: {len(df):,}건  컬럼: {list(df.columns)}')

# ════════════════════════════════════════════════════════════════
# 2. language_type 생성
# ════════════════════════════════════════════════════════════════
KO_RE = re.compile(r'[ᄀ-ᇿ㄰-㆏가-힯]')
EN_RE = re.compile(r'[a-zA-Z]')

def detect_lang(name: str) -> str:
    has_ko = bool(KO_RE.search(name))
    has_en = bool(EN_RE.search(name))
    if has_ko and has_en:
        return '혼합'
    if has_ko:
        return '한국어'
    if has_en:
        return '영어'
    return '기타'

df['language_type'] = df['game_name'].fillna('').apply(detect_lang)

print('\n[language_type 분포]')
print(df['language_type'].value_counts().to_string())

# ════════════════════════════════════════════════════════════════
# 3. grade × company 교차 피처 (≥50건 조합만 유지)
# ════════════════════════════════════════════════════════════════
df['grade_company'] = df['grade'].fillna('미상') + '_' + df['company'].fillna('')
counts = df['grade_company'].value_counts()
valid_combos = set(counts[counts >= CROSS_THRESHOLD].index)
df['grade_company'] = df['grade_company'].apply(
    lambda x: x if x in valid_combos else '기타'
)

print(f'\n[grade×company 교차 피처]  임계값: {CROSS_THRESHOLD}건')
print(f'유효 조합 수: {len(valid_combos)}개')
print(df['grade_company'].value_counts().to_string())

# ════════════════════════════════════════════════════════════════
# 4. FastText 학습
# ════════════════════════════════════════════════════════════════
def tokenize(name: str) -> list:
    name = str(name).lower().strip()
    tokens = re.findall(r'[가-힣]+|[a-z0-9]+', name)
    return tokens if tokens else ['<unk>']

print('\n[FastText 학습]')
sentences = df['game_name'].fillna('').apply(tokenize).tolist()
print(f'학습 샘플: {len(sentences):,}건')

ft_model = FastText(
    sentences=sentences,
    vector_size=FT_DIM,  # 임베딩 차원
    window=3,            # 앞뒤 3토큰 문맥 학습 (게임명은 짧으므로 3이면 충분)
    min_count=1,         # 1회 등장 단어도 학습 (신조어·오타 대응)
    epochs=30,           # 30회 반복 학습
    sg=1,                # Skip-gram 방식: 중심 단어로 주변 단어 예측 → 희소 단어 표현력 우수
    workers=4,
    seed=42,
)
ft_model.save(str(MODEL_PATH))
print(f'모델 저장: {MODEL_PATH}')

# ════════════════════════════════════════════════════════════════
# 5. 임베딩 벡터 생성 (토큰 평균)
# ════════════════════════════════════════════════════════════════
def get_vector(name: str) -> np.ndarray:
    tokens = tokenize(name)
    vecs = [ft_model.wv[t] for t in tokens if t in ft_model.wv]
    # 토큰별 벡터를 평균 → 게임명 전체를 하나의 벡터로 표현
    return np.mean(vecs, axis=0) if vecs else np.zeros(FT_DIM)

print('임베딩 벡터 생성 중...')
embeddings = np.vstack(df['game_name'].fillna('').apply(get_vector).tolist())
ft_cols = [f'ft_{i}' for i in range(FT_DIM)]
df_ft = pd.DataFrame(embeddings, columns=ft_cols, index=df.index)
df = pd.concat([df, df_ft], axis=1)
print(f'FastText 컬럼 추가: ft_0 ~ ft_{FT_DIM - 1}')

# ════════════════════════════════════════════════════════════════
# 6. 저장
# ════════════════════════════════════════════════════════════════
df.to_excel(OUT_B, index=False)
print(f'\n저장 완료: {OUT_B}')
print(f'최종 컬럼 수: {len(df.columns)}개')
print(f'컬럼 목록: {list(df.columns)}')
lbl = df['label'].value_counts().to_dict()
print(f'label: 정상(0)={lbl.get(0, 0):,}건  이상(1)={lbl.get(1, 0):,}건')

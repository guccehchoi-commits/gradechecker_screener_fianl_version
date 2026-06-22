import re
import numpy as np
import pandas as pd
import streamlit as st
from gensim.models import KeyedVectors
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / 'models'
FT_DIM = 50

GRADE_MAP = {
    '전체이용가': '전체이용가', '12세이용가': '12세이용가',
    '15세이용가': '15세이용가', '청소년이용불가': '청소년이용불가',
    '전체': '전체이용가', '전체이용': '전체이용가',
    '3세': '전체이용가', '4+': '전체이용가', '9+': '전체이용가',
    '12세': '12세이용가', '12+': '12세이용가',
    '15세': '15세이용가',
    '16세': '청소년이용불가', '17+': '청소년이용불가',
    '18세': '청소년이용불가', '청불': '청소년이용불가', '18+': '청소년이용불가',
    '등급취소': '미상', '등급거부': '미상',
}

KO_RE = re.compile(r'[가-힣]')
EN_RE = re.compile(r'[a-zA-Z]')
FORTNITE_KEYWORDS = ['fortnite', '포트나이트']
CAT_COLS  = ['grade', 'company', 'language_type', 'grade_company']
FT_COLS   = [f'ft_{i}' for i in range(FT_DIM)]
FEAT_COLS = CAT_COLS + FT_COLS
BETTING_GENRES = ['보드게임(베팅성)']

GAMBLING_KEYWORDS = [
    'slot', 'slots', '슬롯', 'poker', '포커',
    '고스톱', '고스탑', 'gostop', 'go-stop',
    'blackjack', 'black jack', '블랙잭',
    'casino', '카지노', 'roulette', '룰렛',
    'baccarat', '바카라', 'jackpot', '잭팟', '도박',
]
GAMBLING_PATTERN = re.compile('|'.join(re.escape(k) for k in GAMBLING_KEYWORDS), re.IGNORECASE)


@st.cache_resource(show_spinner='FastText 모델 로드 중...')
def load_fasttext():
    return KeyedVectors.load(str(MODEL_DIR / 'fasttext_kv.model'))


def detect_lang(name):
    s = str(name)
    has_ko = bool(KO_RE.search(s))
    has_en = bool(EN_RE.search(s))
    if has_ko and has_en: return '혼합'
    if has_ko:            return '한국어'
    if has_en:            return '영어'
    return '기타'


def normalize_grade(g):
    return GRADE_MAP.get(str(g).strip(), '미상')


def tokenize(name):
    tokens = re.findall(r'[가-힣]+|[a-z0-9]+', str(name).lower().strip())
    return tokens if tokens else ['<unk>']


def get_vector(name, kv):
    tokens = tokenize(name)
    vecs = [kv[t] for t in tokens if t in kv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(FT_DIM)


def run_preprocess(df: pd.DataFrame, kv) -> pd.DataFrame:
    df = df.copy()
    df['grade'] = df['grade'].fillna('미상').apply(normalize_grade)
    if 'company' in df.columns:
        df['company'] = df['company'].fillna('기타').astype(str).str.strip()
    else:
        df['company'] = '기타'
    df['language_type'] = df['game_name'].fillna('').apply(detect_lang)
    df['grade_company'] = df['grade'] + '_' + df['company']
    vecs = np.vstack(
        df['game_name'].fillna('').apply(lambda n: get_vector(n, kv)).tolist()
    )
    ft_df = pd.DataFrame(vecs, columns=FT_COLS, index=df.index)
    df = pd.concat([df, ft_df], axis=1)
    return df


def is_fortnite(game_name: str) -> bool:
    name_lower = str(game_name).lower()
    return any(kw in name_lower for kw in FORTNITE_KEYWORDS)


def apply_genre_boost(prob: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    if 'genre' not in df.columns:
        return prob
    boosted = prob.copy()
    mask = df['genre'].isin(BETTING_GENRES)
    boosted[mask] = np.clip(prob[mask] + 0.05, 0, 1.0)
    return boosted


def apply_gambling_keyword_boost(prob: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    boosted = prob.copy()
    name_series = df['game_name'].fillna('').astype(str)
    grade_series = df['grade'].fillna('') if 'grade' in df.columns else pd.Series([''] * len(df))
    kw_mask    = name_series.str.contains(GAMBLING_PATTERN, na=False)
    grade_mask = grade_series != '청소년이용불가'
    mask = kw_mask & grade_mask
    boosted[mask] = np.maximum(prob[mask] + 0.20, 0.80)
    boosted = np.clip(boosted, 0, 1.0)
    return boosted

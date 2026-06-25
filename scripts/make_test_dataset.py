"""
테스트용 데이터셋 생성
- 크롤링 데이터 (label=0, 2026년 신규)
+ B그룹 label=1 케이스 샘플 (실제 등급재분류/취소 게임)
→ 테스트_데이터셋_최종.xlsx
"""
import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전')
CRAWL_FILE  = BASE_DIR / '테스트_데이터셋_크롤링.xlsx'
B_FILE      = BASE_DIR / r'데이터 3종_260615\B그룹_데이터셋_260619_v3.xlsx'
OUT_FILE    = BASE_DIR / '테스트_데이터셋_최종.xlsx'

SAMPLE_N    = 50   # label=1 샘플 건수 (비율 약 4~5%)
RANDOM_SEED = 42

# ── 1. 크롤링 데이터 로드 ────────────────────────────────────────
print('크롤링 데이터 로드...')
df_crawl = pd.read_excel(CRAWL_FILE)
df_crawl = df_crawl[['game_name', 'grade', 'company', 'label']].copy()
df_crawl['source'] = 'crawl_2026'
print(f'  크롤링: {len(df_crawl)}건 (label=0 전체)')

# ── 2. B그룹 label=1 케이스 로드 ────────────────────────────────
print('B그룹 label=1 케이스 로드...')
df_b = pd.read_excel(B_FILE, usecols=['game_name', 'grade', 'company', 'label'])
df_b1 = df_b[df_b['label'] == 1].copy()
df_b1['source'] = 'b_group_actual'
print(f'  B그룹 label=1 전체: {len(df_b1)}건')

# ── 3. 크롤링 데이터와 중복 제거 ────────────────────────────────
crawl_names = set(df_crawl['game_name'].str.strip().str.lower())
df_b1 = df_b1[~df_b1['game_name'].str.strip().str.lower().isin(crawl_names)].copy()
print(f'  중복 제거 후: {len(df_b1)}건')

# ── 4. 등급 비례 층화 샘플링 ────────────────────────────────────
grade_dist = df_b1['grade'].value_counts(normalize=True)
print('\n  label=1 등급 분포:')
for g, r in grade_dist.items():
    print(f'    {g}: {r:.1%}')

samples = []
for grade, ratio in grade_dist.items():
    n = max(1, round(SAMPLE_N * ratio))
    pool = df_b1[df_b1['grade'] == grade]
    n = min(n, len(pool))
    samples.append(pool.sample(n=n, random_state=RANDOM_SEED))

df_sampled = pd.concat(samples).reset_index(drop=True)
# 반올림 오차로 SAMPLE_N 초과 시 조정
if len(df_sampled) > SAMPLE_N:
    df_sampled = df_sampled.sample(n=SAMPLE_N, random_state=RANDOM_SEED).reset_index(drop=True)

print(f'\n  샘플링 완료: {len(df_sampled)}건')
print('  샘플 등급 분포:')
print(df_sampled['grade'].value_counts().to_string())

# ── 5. 합치기 + 셔플 ────────────────────────────────────────────
df_final = pd.concat([df_crawl, df_sampled], ignore_index=True)
df_final = df_final.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

# ── 6. 저장 (대시보드용: source 컬럼 제외) ───────────────────────
df_save = df_final[['game_name', 'grade', 'company', 'label']].copy()
df_save['company'] = df_save['company'].fillna('').astype(str).str.strip()

df_save.to_excel(OUT_FILE, index=False)

# ── 7. 결과 요약 ────────────────────────────────────────────────
n_total = len(df_save)
n0 = (df_save['label'] == 0).sum()
n1 = (df_save['label'] == 1).sum()

print('\n' + '='*50)
print('테스트 데이터셋 생성 완료')
print('='*50)
print(f'저장 위치: {OUT_FILE}')
print(f'전체 건수: {n_total:,}건')
print(f'  정상(0): {n0:,}건 ({n0/n_total:.1%})')
print(f'  이상(1): {n1:,}건 ({n1/n_total:.1%})')
print()
print('등급 분포:')
print(df_save['grade'].value_counts().to_string())

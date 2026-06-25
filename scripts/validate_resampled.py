"""
실제 운영환경 이상비율(2.3%)에 맞춘 리샘플링 외부 검증
- 정상 1,099건 고정 + 이상 ~26건 샘플링 × 30회 반복
- 결과: 평균 ± 표준편차
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

sys.path.insert(0, r'C:\Users\User\python\GUCC\dashboard')
from utils.preprocess import run_preprocess, load_fasttext, apply_gambling_keyword_boost
from utils.model import load_model, predict

DATA_DIR = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615\앱 테스트용')

# 데이터 로드
df0 = pd.read_excel(DATA_DIR / '테스트_데이터셋_크롤링.xlsx')
df0['label'] = 0
df1 = pd.read_excel(DATA_DIR / '2025-2026 등급 부적합 데이터_테스트.xlsx')
df1 = df1.rename(columns={'게임명': 'game_name', '등급': 'grade'})
df1['label'] = 1

# 목표 이상비율 → 이상 샘플 수 계산
TARGET_RATIO = 0.023          # A그룹과 동일한 실제 운영환경 비율
N_NORMAL     = len(df0)       # 정상 1,099건 고정
N_ABNORMAL   = round(N_NORMAL * TARGET_RATIO / (1 - TARGET_RATIO))  # ~26건
N_ITER       = 30             # 반복 횟수
THRESHOLD    = 0.80

print(f"정상: {N_NORMAL}건 고정 / 이상: 매 회 {N_ABNORMAL}건 무작위 샘플링 × {N_ITER}회")
print(f"이상비율: {N_ABNORMAL/(N_NORMAL+N_ABNORMAL)*100:.1f}%\n")

print("모델 로드 중...")
ft_model = load_fasttext()
clf, prep = load_model()

results = []
for seed in range(N_ITER):
    sample1 = df1.sample(n=N_ABNORMAL, random_state=seed)
    df_iter  = pd.concat([df0, sample1], ignore_index=True)

    df_feat = run_preprocess(df_iter, ft_model)
    prob    = predict(df_feat, clf, prep)
    prob    = apply_gambling_keyword_boost(prob, df_feat)

    y    = df_iter['label'].values
    yp   = (prob >= THRESHOLD).astype(int)
    auc  = roc_auc_score(y, prob)
    prec = precision_score(y, yp, zero_division=0)
    rec  = recall_score(y, yp, zero_division=0)
    f1   = f1_score(y, yp, zero_division=0)
    results.append({'AUC': auc, '정밀도': prec, '재현율': rec, 'F1': f1})

df_res = pd.DataFrame(results)

print("="*50)
print(f"  리샘플링 외부 검증 결과 (threshold={THRESHOLD})")
print("="*50)
for col in ['AUC', '정밀도', '재현율', 'F1']:
    m, s = df_res[col].mean(), df_res[col].std()
    print(f"  {col:6s}: {m:.4f} ± {s:.4f}")

print(f"\n  (30회 평균, 매 회 이상 {N_ABNORMAL}건 무작위 샘플링)")

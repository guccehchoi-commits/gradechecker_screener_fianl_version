"""
신규 외부 검증: 2026 자체등급분류(label=0) + 2025-2026 등급부적합(label=1)
Threshold 0.40 / 0.45 / 0.50 / 0.80 비교
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

sys.path.insert(0, r'C:\Users\User\python\GUCC\dashboard')
from utils.preprocess import run_preprocess, load_fasttext, apply_gambling_keyword_boost
from utils.model import load_model, predict, risk_label

DATA_DIR = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615\앱 테스트용\1차 테스트_260621')

# ── 데이터 로드 & 라벨 부여 ──────────────────────────────────
df0 = pd.read_excel(DATA_DIR / '추가 자체등급분류 데이터_테스트용.xlsx')
df0['label'] = 0

df1 = pd.read_excel(DATA_DIR / '2025-2026 등급 부적합 데이터_테스트용.xlsx')
df1 = df1.rename(columns={'게임명': 'game_name', '등급': 'grade'})
df1['label'] = 1

df = pd.concat([df0, df1], ignore_index=True)
print(f"검증셋: {len(df):,}건 (정상 {len(df0):,} / 부적합 {len(df1):,})")
print(f"이상 비율: {len(df1)/len(df)*100:.1f}%\n")

# ── 전처리 & 예측 (한 번만) ────────────────────────────────────
print("FastText 로드 중...")
ft_model = load_fasttext()

print("전처리 중...")
df_feat = run_preprocess(df, ft_model)

print("예측 중...")
clf, prep = load_model()
prob = predict(df_feat, clf, prep)
prob = apply_gambling_keyword_boost(prob, df_feat)

y_true = df['label'].values
auc = roc_auc_score(y_true, prob)

print(f"\nAUC (threshold 무관): {auc:.4f}")
print(f"실제 이상(label=1): {len(df1)}건 / 실제 정상(label=0): {len(df0)}건\n")

# ── Threshold별 성능 비교 ─────────────────────────────────────
THRESHOLDS = [0.80, 0.50, 0.45, 0.40]
results = []

print("="*70)
print(f"  {'Threshold':>10}  {'정밀도':>8}  {'재현율':>8}  {'F1':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'TN':>6}")
print("="*70)

for thr in THRESHOLDS:
    y_pred = (prob >= thr).astype(int)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)
    TP, FN = int(cm[1][1]), int(cm[1][0])
    FP, TN = int(cm[0][1]), int(cm[0][0])
    results.append({'threshold': thr, '정밀도': prec, '재현율': rec, 'F1': f1,
                    'TP': TP, 'FP': FP, 'FN': FN, 'TN': TN})
    print(f"  {thr:>10.2f}  {prec:>8.4f}  {rec:>8.4f}  {f1:>7.4f}  {TP:>5}  {FP:>5}  {FN:>5}  {TN:>6}")

print("="*70)

# ── 권장 Threshold 요약 ───────────────────────────────────────
print("\n[권장 Threshold]")
print("  공모전 기준: Threshold 0.50  (F1 최고, 정밀도·재현율 균형)")
print("  현장 운영:  Threshold 0.40  (재현율 최대화, FN 최소)")

# ── 결과 저장 ─────────────────────────────────────────────────
risk_labels_list = [risk_label(p) for p in prob]
df_res = df[['game_name', 'label']].copy()
df_res['확률'] = prob.round(4)
df_res['위험도'] = risk_labels_list
df_res['grade'] = df_feat['grade']
df_res['pred_050'] = (prob >= 0.50).astype(int)
df_res['pred_040'] = (prob >= 0.40).astype(int)

out_path = DATA_DIR / '신규외부검증_결과.xlsx'
df_res.sort_values('확률', ascending=False).to_excel(out_path, index=False)
print(f"\n결과 저장: {out_path}")

"""
도박성 키워드 보정 적용 후 성능 재평가
테스트_데이터셋_최종.xlsx → 전처리 → XGBoost 예측 → 보정 → 평가
"""
import sys
sys.path.insert(0, r'C:\Users\User\python\GUCC\dashboard')
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix
from utils.preprocess import run_preprocess, load_fasttext, apply_genre_boost, apply_gambling_keyword_boost
from utils.model import load_model, predict

TRUTH_FILE = (r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026'
              r'\6. 2026 문화 디지털혁신 및 데이터 활용 공모전'
              r'\데이터 3종_260615\앱 테스트용\테스트_데이터셋_최종.xlsx')
THR = 0.80

df = pd.read_excel(TRUTH_FILE)
print(f'테스트셋 로드: {len(df)}건')

ft_model = load_fasttext()
df_feat  = run_preprocess(df, ft_model)
clf, prep = load_model()
prob = predict(df_feat, clf, prep)
prob = apply_genre_boost(prob, df)
prob_boosted = apply_gambling_keyword_boost(prob.copy(), df_feat)

y_true = df['label'].fillna(0).astype(int)

def report(label, p):
    y_pred = (p >= THR).astype(int)
    auc = roc_auc_score(y_true, p)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    prec = tp/(tp+fp) if tp+fp else 0
    rec  = tp/(tp+fn) if tp+fn else 0
    f1   = 2*prec*rec/(prec+rec) if prec+rec else 0
    print(f'\n[{label}]')
    print(f'  AUC:       {auc:.4f}')
    print(f'  Precision: {prec:.4f}')
    print(f'  Recall:    {rec:.4f}')
    print(f'  F1:        {f1:.4f}')
    print(f'  TP:{tp}  FP:{fp}  FN:{fn}  TN:{tn}')
    df1 = df[y_true==1].copy(); df1['prob']=p[y_true==1]
    hi = (df1['prob']>=0.80).sum()
    mi = ((df1['prob']>=0.50)&(df1['prob']<0.80)).sum()
    lo = (df1['prob']<0.50).sum()
    print(f'  label=1 분류: 고위험={hi} / 중위험={mi} / 저위험={lo}')
    return {'auc':auc,'prec':prec,'rec':rec,'f1':f1,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'hi':hi,'mi':mi,'lo':lo}

r_before = report('보정 전', prob)
r_after  = report('보정 후', prob_boosted)

print('\n[변화량]')
print(f'  AUC:       {r_before["auc"]:.4f} → {r_after["auc"]:.4f}')
print(f'  Precision: {r_before["prec"]:.4f} → {r_after["prec"]:.4f}')
print(f'  Recall:    {r_before["rec"]:.4f} → {r_after["rec"]:.4f}')
print(f'  F1:        {r_before["f1"]:.4f} → {r_after["f1"]:.4f}')
print(f'  TP: {r_before["tp"]} → {r_after["tp"]} / FN: {r_before["fn"]} → {r_after["fn"]}')

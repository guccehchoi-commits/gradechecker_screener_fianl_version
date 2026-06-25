import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix

f = r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615\앱 테스트용\1차 테스트_260621\신규외부검증_결과.xlsx'
df = pd.read_excel(f)

y_true = df['label'].values
prob   = df['확률'].values

auc = roc_auc_score(y_true, prob)
n1 = int((y_true==1).sum())
n0 = int((y_true==0).sum())
print(f'AUC: {auc:.4f}')
print(f'label=1: {n1}건 / label=0: {n0}건 / 합계: {len(y_true)}건')
print()

header = 'Threshold  정밀도    재현율    F1       TP     FP     FN     TN'
print(header)
print('='*70)

for thr in [0.80, 0.50, 0.45, 0.40]:
    yp = (prob >= thr).astype(int)
    p  = precision_score(y_true, yp, zero_division=0)
    r  = recall_score(y_true, yp, zero_division=0)
    f1 = f1_score(y_true, yp, zero_division=0)
    cm = confusion_matrix(y_true, yp)
    TP, FN = int(cm[1][1]), int(cm[1][0])
    FP, TN = int(cm[0][1]), int(cm[0][0])
    print(f'{thr:.2f}       {p:.4f}    {r:.4f}    {f1:.4f}   {TP:4d}   {FP:4d}   {FN:4d}   {TN:4d}')

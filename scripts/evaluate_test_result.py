import sys
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix

TRUTH_FILE = r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615\앱 테스트용\테스트_데이터셋_최종.xlsx'
PRED_FILE  = r'C:\Users\User\Downloads\이상탐지_결과_테스트_데이터셋_최종.xlsx'
THR = 0.80

df_truth = pd.read_excel(TRUTH_FILE)
df_pred  = pd.read_excel(PRED_FILE, index_col=0).rename(columns={'게임명': 'game_name'})

df = df_pred.merge(df_truth[['game_name', 'label']], on='game_name', how='left')
print(f'병합 건수: {len(df)} / 매칭 실패: {df["label"].isna().sum()}건')

y_true = df['label'].fillna(0).astype(int)
y_prob = df['재분류_확률']
y_pred = (y_prob >= THR).astype(int)

auc = roc_auc_score(y_true, y_prob)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

prec = tp / (tp + fp) if (tp + fp) > 0 else 0
rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0

print(f'\nAUC: {auc:.4f}')
print(f'\n혼동행렬 (threshold={THR})')
print(f'  TP 정탐(이상->이상): {tp}건')
print(f'  FN 미탐(이상->정상): {fn}건')
print(f'  FP 오탐(정상->이상): {fp}건')
print(f'  TN 정탐(정상->정상): {tn}건')
print(f'\n  정밀도: {prec:.4f}')
print(f'  재현율: {rec:.4f}')
print(f'  F1:     {f1:.4f}')

df1 = df[df['label'] == 1].sort_values('재분류_확률', ascending=False)
hi = (df1['재분류_확률'] >= 0.80).sum()
mi = ((df1['재분류_확률'] >= 0.50) & (df1['재분류_확률'] < 0.80)).sum()
lo = (df1['재분류_확률'] < 0.50).sum()
print(f'\nlabel=1 게임 {len(df1)}건 중')
print(f'  고위험(>=0.80) 잡음: {hi}건')
print(f'  중위험(0.50~)  잡음: {mi}건')
print(f'  저위험(<0.50)  놓침: {lo}건')
print(f'\n상위 5건:')
print(df1[['game_name', '등급', '재분류_확률', '위험도']].head(5).to_string())
print(f'\n미탐지(놓친 게임):')
missed = df1[df1['재분류_확률'] < 0.50]
if len(missed):
    print(missed[['game_name', '등급', '재분류_확률', '위험도']].to_string())
else:
    print('  없음')

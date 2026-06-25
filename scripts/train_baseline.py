"""
[STEP 6] 베이스라인 모델 학습 — Logistic Regression
======================================================
목적:
  단순 선형 모델로도 예측 가능한 수준을 확인하고,
  XGBoost 선택의 근거(성능 향상폭)를 마련한다.

실험 구성:
  A. 전체 데이터 (미상 grade 포함) — hold-out 20%
  B. 미상 grade 제외 서브셋       — hold-out 20%
  C. 5-Fold Stratified CV (전체)

결과:
  실험 A: AUC=0.9124, F1(이상)=0.6898
  실험 B: AUC=0.8680, F1(이상)=0.5310
  실험 C: AUC=0.9145±0.006 (안정적)

입력: B그룹_데이터셋_260619_v3.xlsx
출력: baseline_lr.pkl, baseline_lr_report.txt
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR  = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
MODEL_DIR = Path(r'C:\Users\User\python')

IN_B      = DATA_DIR / 'B그룹_데이터셋_260619_v3.xlsx'
MODEL_OUT = MODEL_DIR / 'baseline_lr.pkl'
REPORT_OUT = MODEL_DIR / 'baseline_lr_report.txt'

CAT_COLS = ['grade', 'company', 'language_type', 'grade_company']
FT_COLS  = [f'ft_{i}' for i in range(50)]
FEAT_COLS = CAT_COLS + FT_COLS

SEED = 42

# ════════════════════════════════════════════════════════════════
# 1. 데이터 로드
# ════════════════════════════════════════════════════════════════
print('=' * 60)
print('데이터 로드')
df = pd.read_excel(IN_B)
print(f'전체: {len(df):,}건  이상률: {df["label"].mean()*100:.1f}%')

X = df[FEAT_COLS].copy()
y = df['label'].copy()

for col in CAT_COLS:
    X[col] = X[col].fillna('미상').astype(str)

# ════════════════════════════════════════════════════════════════
# 2. 파이프라인 구성
# ════════════════════════════════════════════════════════════════
# 범주형 피처: 원핫인코딩 (모르는 카테고리는 무시)
# 수치형 피처(FastText): 표준화 (평균 0, 분산 1)
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ('num', StandardScaler(), FT_COLS),
])

pipeline = Pipeline([
    ('prep', preprocessor),
    ('clf',  LogisticRegression(
        class_weight='balanced',  # 이상치(23.8%) vs 정상(76.2%) 불균형 보정: 이상치에 더 높은 가중치 부여
        max_iter=1000,            # 원핫인코딩 후 피처 수 증가로 기본값(100)으로는 수렴 안 됨
        C=1.0,                    # 정규화 강도 (1/C): 기본값 유지
        solver='lbfgs',           # 다차원 문제에 안정적인 최적화 알고리즘
        random_state=SEED,
        n_jobs=-1,
    )),
])

# ════════════════════════════════════════════════════════════════
# 3. 학습 / 평가 함수
# ════════════════════════════════════════════════════════════════
def evaluate(pipe, X_tr, X_te, y_tr, y_te, tag=''):
    pipe.fit(X_tr, y_tr)
    y_pred = pipe.predict(X_te)
    y_prob = pipe.predict_proba(X_te)[:, 1]

    f1_macro = f1_score(y_te, y_pred, average='macro')
    f1_pos   = f1_score(y_te, y_pred, average='binary')
    auc      = roc_auc_score(y_te, y_prob)

    lines = []
    lines.append(f'\n{"=" * 60}')
    lines.append(f'[{tag}]  test 건수: {len(y_te):,}  이상률: {y_te.mean()*100:.1f}%')
    lines.append(f'  F1(macro)={f1_macro:.4f}  F1(이상)={f1_pos:.4f}  AUC={auc:.4f}')
    lines.append(classification_report(y_te, y_pred, target_names=['정상(0)', '이상(1)']))
    lines.append(f'혼동행렬:\n{confusion_matrix(y_te, y_pred)}')
    return '\n'.join(lines), pipe, f1_macro, f1_pos, auc

# ════════════════════════════════════════════════════════════════
# 4-A. 전체 데이터 (미상 grade 포함)
# ════════════════════════════════════════════════════════════════
print('\n[실험 A] 전체 데이터 (미상 grade 포함)')
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
report_a, pipe_a, f1m_a, f1p_a, auc_a = evaluate(
    Pipeline([('prep', preprocessor), ('clf', LogisticRegression(
        class_weight='balanced', max_iter=1000, C=1.0,
        solver='lbfgs', random_state=SEED, n_jobs=-1))]),
    X_tr, X_te, y_tr, y_te, tag='전체(미상 포함)'
)
print(report_a)

# ════════════════════════════════════════════════════════════════
# 4-B. 미상 grade 제외 서브셋
# ════════════════════════════════════════════════════════════════
print('\n[실험 B] 미상 grade 제외 서브셋')
mask_no_missing = df['grade'] != '미상'
X_nm = X[mask_no_missing].reset_index(drop=True)
y_nm = y[mask_no_missing].reset_index(drop=True)
print(f'미상 제외 후: {len(y_nm):,}건  이상률: {y_nm.mean()*100:.1f}%')

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_nm, y_nm, test_size=0.2, random_state=SEED, stratify=y_nm
)
report_b, pipe_b, f1m_b, f1p_b, auc_b = evaluate(
    Pipeline([('prep', preprocessor), ('clf', LogisticRegression(
        class_weight='balanced', max_iter=1000, C=1.0,
        solver='lbfgs', random_state=SEED, n_jobs=-1))]),
    X_tr2, X_te2, y_tr2, y_te2, tag='미상 제외'
)
print(report_b)

# ════════════════════════════════════════════════════════════════
# 4-C. 전체 데이터 5-Fold CV (AUC)
# ════════════════════════════════════════════════════════════════
print('\n[실험 C] 5-Fold Stratified CV (전체 데이터)')
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_auc = cross_val_score(pipeline, X, y, cv=cv, scoring='roc_auc', n_jobs=-1)
cv_f1  = cross_val_score(pipeline, X, y, cv=cv, scoring='f1_macro', n_jobs=-1)
report_c = (
    f'\n{"=" * 60}\n'
    f'[5-Fold CV] AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}  '
    f'F1(macro)={cv_f1.mean():.4f}±{cv_f1.std():.4f}\n'
    f'  AUC per fold: {[round(v,4) for v in cv_auc]}\n'
    f'  F1  per fold: {[round(v,4) for v in cv_f1]}'
)
print(report_c)

# ════════════════════════════════════════════════════════════════
# 5. 최종 모델 전체 데이터로 재학습 + 저장
# ════════════════════════════════════════════════════════════════
print('\n최종 모델 전체 데이터 재학습 중...')
pipeline.fit(X, y)
with open(MODEL_OUT, 'wb') as f:
    pickle.dump(pipeline, f)
print(f'모델 저장: {MODEL_OUT}')

# ════════════════════════════════════════════════════════════════
# 6. 리포트 저장
# ════════════════════════════════════════════════════════════════
summary = (
    '=== Logistic Regression 베이스라인 성능 리포트 ===\n'
    f'데이터: {IN_B.name}\n'
    f'피처: {CAT_COLS} + ft_0~ft_49\n'
    '\n[실험 A] 전체(미상 포함)'
    + report_a
    + '\n\n[실험 B] 미상 grade 제외'
    + report_b
    + report_c
)
REPORT_OUT.write_text(summary, encoding='utf-8')
print(f'리포트 저장: {REPORT_OUT}')

# ════════════════════════════════════════════════════════════════
# 7. 피처 중요도 (계수 절댓값 상위 20)
# ════════════════════════════════════════════════════════════════
ohe = pipeline.named_steps['prep'].named_transformers_['cat']
cat_feat_names = ohe.get_feature_names_out(CAT_COLS).tolist()
all_feat_names = cat_feat_names + FT_COLS
coef = pipeline.named_steps['clf'].coef_[0]

top_idx = np.argsort(np.abs(coef))[::-1][:20]
print('\n[피처 중요도 상위 20 (계수 절댓값)]')
print('  coef 양수(+): 이상 확률 높임  /  음수(-): 이상 확률 낮춤')
for rank, i in enumerate(top_idx, 1):
    print(f'  {rank:>2}. {all_feat_names[i]:<35} coef={coef[i]:+.4f}')

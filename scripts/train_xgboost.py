"""
[STEP 7] 메인 모델 학습 — XGBoost + SHAP 분석
===============================================
목적:
  등급재분류 가능성이 높은 게임을 탐지하는 최종 예측 모델 학습.
  SHAP 분석으로 어떤 피처가 예측에 영향을 미치는지 해석한다.

실험 구성:
  A. 전체 데이터 (미상 grade 포함) — hold-out 20%  ★ 최종 채택
  B. 미상 grade 제외 서브셋       — hold-out 20%
  C. 5-Fold Stratified CV (전체)

결과:
  실험 A: AUC=0.9363, F1(macro)=0.8423, F1(이상)=0.7621  ← 베이스라인 대비 +0.072
  실험 B: AUC=0.8918, F1(이상)=0.6239
  실험 C: AUC=0.9335±0.005 (안정적)

SHAP 상위 피처:
  1위 grade=미상 (0.73)  — 이상 방향
  2위 company=포트나이트 (0.65) — 정상 방향 (편향)
  3위 company=구글 플레이 (0.56)
  4위~ FastText 임베딩 다수

입력: B그룹_데이터셋_260619_v3.xlsx
출력: xgb_model.pkl, xgb_preprocessor.pkl, xgb_report.txt, shap_importance.png
"""
import sys
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
import shap
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 한글 폰트 설정 (Windows Malgun Gothic)
_ko_font = fm.FontProperties(fname=r'C:\Windows\Fonts\malgun.ttf')
plt.rcParams['font.family'] = _ko_font.get_name()
plt.rcParams['axes.unicode_minus'] = False

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR  = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
MODEL_DIR = Path(r'C:\Users\User\python')

IN_B        = DATA_DIR / 'B그룹_데이터셋_260619_v3.xlsx'
MODEL_OUT   = MODEL_DIR / 'xgb_model.pkl'
PREP_OUT    = MODEL_DIR / 'xgb_preprocessor.pkl'
REPORT_OUT  = MODEL_DIR / 'xgb_report.txt'
SHAP_PNG    = MODEL_DIR / 'shap_importance.png'

CAT_COLS  = ['grade', 'company', 'language_type', 'grade_company']
FT_COLS   = [f'ft_{i}' for i in range(50)]
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

# scale_pos_weight: 클래스 불균형 보정값 = 정상 건수 / 이상 건수
# XGBoost가 이상치(소수 클래스)를 더 강하게 학습하도록 가중치 부여
n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
spw = round(n_neg / n_pos, 3)
print(f'scale_pos_weight: {n_neg}/{n_pos} = {spw}')

# ════════════════════════════════════════════════════════════════
# 2. 전처리기
# ════════════════════════════════════════════════════════════════
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ('num', StandardScaler(), FT_COLS),
])

# ════════════════════════════════════════════════════════════════
# 3. 학습 / 평가 함수
# ════════════════════════════════════════════════════════════════
def make_xgb(spw):
    return xgb.XGBClassifier(
        n_estimators=500,          # 최대 트리 수 (early stopping으로 실제 사용 트리 결정)
        max_depth=5,               # 트리 깊이 제한 → 과적합 방지
        learning_rate=0.05,        # 작은 학습률 + 많은 트리 = 안정적 수렴
        subsample=0.8,             # 각 트리 학습 시 80% 샘플만 사용 → 과적합 방지
        colsample_bytree=0.8,      # 각 트리 학습 시 80% 피처만 사용 → 다양성 확보
        min_child_weight=5,        # 리프 노드 최소 샘플 수 → 희소 패턴 학습 억제
        gamma=0.1,                 # 분할 최소 손실 감소량 → 불필요한 분할 방지
        reg_alpha=0.1,             # L1 정규화
        reg_lambda=1.0,            # L2 정규화
        scale_pos_weight=spw,      # 클래스 불균형 보정
        eval_metric='auc',
        early_stopping_rounds=30,  # 30회 연속 AUC 미개선 시 학습 조기 종료
        use_label_encoder=False,
        random_state=SEED,
        n_jobs=-1,
        verbosity=0,
    )

def evaluate(X_tr, X_te, y_tr, y_te, tag, spw_val):
    prep = ColumnTransformer([
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
        ('num', StandardScaler(), FT_COLS),
    ])
    X_tr_t = prep.fit_transform(X_tr)
    X_te_t = prep.transform(X_te)

    # 조기 종료용 검증셋 (train 내 10%)
    X_tr2, X_val, y_tr2, y_val = train_test_split(
        X_tr_t, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr
    )

    clf = make_xgb(spw_val)
    clf.fit(
        X_tr2, y_tr2,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    best_n = clf.best_iteration + 1

    y_pred = clf.predict(X_te_t)
    y_prob = clf.predict_proba(X_te_t)[:, 1]

    f1_macro = f1_score(y_te, y_pred, average='macro')
    f1_pos   = f1_score(y_te, y_pred, average='binary')
    auc      = roc_auc_score(y_te, y_prob)

    lines = [
        f'\n{"=" * 60}',
        f'[{tag}]  test: {len(y_te):,}건  이상률: {y_te.mean()*100:.1f}%  best_n={best_n}',
        f'  F1(macro)={f1_macro:.4f}  F1(이상)={f1_pos:.4f}  AUC={auc:.4f}',
        classification_report(y_te, y_pred, target_names=['정상(0)', '이상(1)']),
        f'혼동행렬:\n{confusion_matrix(y_te, y_pred)}',
    ]
    return '\n'.join(lines), clf, prep, f1_macro, f1_pos, auc

# ════════════════════════════════════════════════════════════════
# 4-A. 전체 데이터 (미상 grade 포함)
# ════════════════════════════════════════════════════════════════
print('\n[실험 A] 전체 데이터 (미상 grade 포함)')
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
report_a, clf_a, prep_a, f1m_a, f1p_a, auc_a = evaluate(
    X_tr, X_te, y_tr, y_te, '전체(미상 포함)', spw
)
print(report_a)

# ════════════════════════════════════════════════════════════════
# 4-B. 미상 grade 제외 서브셋
# ════════════════════════════════════════════════════════════════
print('\n[실험 B] 미상 grade 제외 서브셋')
mask = df['grade'] != '미상'
X_nm = X[mask].reset_index(drop=True)
y_nm = y[mask].reset_index(drop=True)
spw_nm = round((y_nm == 0).sum() / (y_nm == 1).sum(), 3)
print(f'미상 제외: {len(y_nm):,}건  이상률: {y_nm.mean()*100:.1f}%  spw={spw_nm}')

X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_nm, y_nm, test_size=0.2, random_state=SEED, stratify=y_nm
)
report_b, clf_b, prep_b, f1m_b, f1p_b, auc_b = evaluate(
    X_tr2, X_te2, y_tr2, y_te2, '미상 제외', spw_nm
)
print(report_b)

# ════════════════════════════════════════════════════════════════
# 4-C. 5-Fold CV (Pipeline, 전체)
# ════════════════════════════════════════════════════════════════
print('\n[실험 C] 5-Fold Stratified CV (전체 데이터)')
# CV용 파이프라인: early stopping 없이 best_n으로 고정
best_n_a = clf_a.best_iteration + 1
cv_clf = xgb.XGBClassifier(
    n_estimators=best_n_a,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=spw,
    use_label_encoder=False,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)
cv_pipe = Pipeline([('prep', preprocessor), ('clf', cv_clf)])
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
cv_auc = cross_val_score(cv_pipe, X, y, cv=cv, scoring='roc_auc', n_jobs=1)
cv_f1  = cross_val_score(cv_pipe, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
report_c = (
    f'\n{"=" * 60}\n'
    f'[5-Fold CV]  n_estimators={best_n_a}\n'
    f'  AUC={cv_auc.mean():.4f}±{cv_auc.std():.4f}  '
    f'F1(macro)={cv_f1.mean():.4f}±{cv_f1.std():.4f}\n'
    f'  AUC per fold: {[round(v,4) for v in cv_auc]}\n'
    f'  F1  per fold: {[round(v,4) for v in cv_f1]}'
)
print(report_c)

# ════════════════════════════════════════════════════════════════
# 5. 베이스라인 비교
# ════════════════════════════════════════════════════════════════
LR_AUC, LR_F1M, LR_F1P = 0.9124, 0.7852, 0.6898
report_cmp = (
    f'\n{"=" * 60}\n'
    f'[베이스라인 vs XGBoost (실험 A, 미상 포함)]\n'
    f'{"":20}  {"AUC":>8}  {"F1(macro)":>10}  {"F1(이상)":>9}\n'
    f'{"LR 베이스라인":20}  {LR_AUC:>8.4f}  {LR_F1M:>10.4f}  {LR_F1P:>9.4f}\n'
    f'{"XGBoost":20}  {auc_a:>8.4f}  {f1m_a:>10.4f}  {f1p_a:>9.4f}\n'
    f'{"향상":20}  {auc_a-LR_AUC:>+8.4f}  {f1m_a-LR_F1M:>+10.4f}  {f1p_a-LR_F1P:>+9.4f}'
)
print(report_cmp)

# ════════════════════════════════════════════════════════════════
# 6. SHAP 분석 (실험 A 모델)
# ════════════════════════════════════════════════════════════════
print('\n[SHAP 분석]')
ohe      = prep_a.named_transformers_['cat']
cat_names = ohe.get_feature_names_out(CAT_COLS).tolist()
feat_names = cat_names + FT_COLS

X_te_t = prep_a.transform(X_te)
X_te_df = pd.DataFrame(X_te_t, columns=feat_names)

# SHAP(SHapley Additive exPlanations): 각 피처가 개별 예측값에 기여한 정도를 수치화
# TreeExplainer는 트리 기반 모델에 특화된 빠른 SHAP 계산 방법
explainer  = shap.TreeExplainer(clf_a)
shap_vals  = explainer.shap_values(X_te_df)

# 피처별 평균 |SHAP| → 절댓값 평균이 클수록 예측에 큰 영향을 주는 피처
mean_shap = np.abs(shap_vals).mean(axis=0)
top20_idx  = np.argsort(mean_shap)[::-1][:20]
top20_names = [feat_names[i] for i in top20_idx]
top20_vals  = mean_shap[top20_idx]

print('SHAP 중요도 상위 20 (mean |SHAP value|)')
for rank, (name, val) in enumerate(zip(top20_names, top20_vals), 1):
    print(f'  {rank:>2}. {name:<40} {val:.4f}')

# SHAP 바 차트 저장
fig, ax = plt.subplots(figsize=(9, 7))
ax.barh(range(20), top20_vals[::-1], color='steelblue')
ax.set_yticks(range(20))
ax.set_yticklabels(top20_names[::-1], fontsize=9)
ax.set_xlabel('mean |SHAP value|')
ax.set_title('XGBoost SHAP Feature Importance (Top 20)')
plt.tight_layout()
plt.savefig(SHAP_PNG, dpi=150)
plt.close()
print(f'SHAP 차트 저장: {SHAP_PNG}')

# ════════════════════════════════════════════════════════════════
# 7. 최종 모델 전체 데이터 재학습 + 저장
# ════════════════════════════════════════════════════════════════
print('\n최종 모델 전체 재학습...')
final_prep = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ('num', StandardScaler(), FT_COLS),
])
X_all_t = final_prep.fit_transform(X)
final_clf = xgb.XGBClassifier(
    n_estimators=best_n_a,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=5,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=spw,
    use_label_encoder=False,
    random_state=SEED,
    n_jobs=-1,
    verbosity=0,
)
final_clf.fit(X_all_t, y)

with open(MODEL_OUT, 'wb') as f:
    pickle.dump(final_clf, f)
with open(PREP_OUT, 'wb') as f:
    pickle.dump(final_prep, f)
print(f'모델 저장: {MODEL_OUT}')
print(f'전처리기 저장: {PREP_OUT}')

# ════════════════════════════════════════════════════════════════
# 8. 리포트 저장
# ════════════════════════════════════════════════════════════════
full_report = (
    '=== XGBoost 메인 모델 성능 리포트 ===\n'
    f'데이터: {IN_B.name}\n'
    f'피처: {CAT_COLS} + ft_0~ft_49\n'
    f'scale_pos_weight={spw}  best_n_estimators={best_n_a}\n'
    + report_a + '\n\n' + report_b + '\n' + report_c + '\n' + report_cmp
)
REPORT_OUT.write_text(full_report, encoding='utf-8')
print(f'리포트 저장: {REPORT_OUT}')

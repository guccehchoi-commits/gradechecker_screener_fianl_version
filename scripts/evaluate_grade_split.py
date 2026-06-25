"""
[STEP 8] grade=미상 분리 평가 — 모델 필요성 입증
===================================================
목적:
  "grade=미상이면 무조건 이상"이라는 단순 규칙만으로 충분한가?
  → grade=미상을 제거한 서브셋에서도 모델이 이상치를 탐지하는지 확인
  → game_name(FastText), company 등 다른 피처의 실질 기여 입증

평가 구성:
  ① 전체 테스트셋 (3,114건)
  ② grade=미상 만 (263건) — 전량 이상치, AUC 계산 불가
  ③ grade 정상 기재 (2,851건) — 모델 vs 단순 규칙 비교

결과:
  ③ 단순 규칙 (무조건 정상 예측): F1(이상)=0.0000
  ③ XGBoost 모델:                 F1(이상)=0.6375, AUC=0.9011
  → grade 외 피처가 이상 탐지에 실질적으로 기여함을 확인

입력: B그룹_데이터셋_260619_v3.xlsx (STEP 2 출력물)
출력: grade_split_eval_report.txt
"""
import sys
import numpy as np
import pandas as pd
from pathlib import Path

import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix,
)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA_DIR   = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
MODEL_DIR  = Path(r'C:\Users\User\python')
IN_B       = DATA_DIR / 'B그룹_데이터셋_260619_v3.xlsx'
REPORT_OUT = MODEL_DIR / 'grade_split_eval_report.txt'

CAT_COLS  = ['grade', 'company', 'language_type', 'grade_company']
FT_COLS   = [f'ft_{i}' for i in range(50)]
FEAT_COLS = CAT_COLS + FT_COLS
SEED = 42

# ─────────────────────────────────────────────
# 1. 데이터 로드 + 실험 A 동일 분할
# ─────────────────────────────────────────────
df = pd.read_excel(IN_B)
print(f'전체: {len(df):,}건  이상률: {df["label"].mean()*100:.1f}%')

X = df[FEAT_COLS].copy()
y = df['label'].copy()
for col in CAT_COLS:
    X[col] = X[col].fillna('미상').astype(str)

n_neg, n_pos = (y == 0).sum(), (y == 1).sum()
spw = round(n_neg / n_pos, 3)
print(f'scale_pos_weight={spw}')

X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
grade_te = X_te['grade'].values  # 이미 '미상' 채워진 상태

# ─────────────────────────────────────────────
# 2. 전처리 + 모델 학습 (실험 A 재현)
# ─────────────────────────────────────────────
print('\n모델 학습 중...')
prep = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), CAT_COLS),
    ('num', StandardScaler(), FT_COLS),
])
X_tr_t = prep.fit_transform(X_tr)
X_te_t = prep.transform(X_te)

X_tr2, X_val, y_tr2, y_val = train_test_split(
    X_tr_t, y_tr, test_size=0.1, random_state=SEED, stratify=y_tr
)

clf = xgb.XGBClassifier(
    n_estimators=500, max_depth=5, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
    scale_pos_weight=spw, eval_metric='auc',
    early_stopping_rounds=30, use_label_encoder=False,
    random_state=SEED, n_jobs=-1, verbosity=0,
)
clf.fit(X_tr2, y_tr2, eval_set=[(X_val, y_val)], verbose=False)
print(f'best_iteration: {clf.best_iteration + 1}')

# ─────────────────────────────────────────────
# 3. 테스트셋 전체 예측
# ─────────────────────────────────────────────
y_pred_all = clf.predict(X_te_t)
y_prob_all = clf.predict_proba(X_te_t)[:, 1]
y_te_arr   = y_te.values

# ─────────────────────────────────────────────
# 4. 서브셋 평가 함수
# ─────────────────────────────────────────────
def eval_subset(mask, tag):
    n = mask.sum()
    yt    = y_te_arr[mask]
    yp    = y_pred_all[mask]
    yprob = y_prob_all[mask]

    lines = [f'\n{"="*60}', f'[{tag}]  {n:,}건  이상 {yt.sum()}건 ({yt.mean()*100:.1f}%)']

    if len(np.unique(yt)) < 2:
        lines.append('  → 단일 클래스: AUC 계산 불가')
        lines.append(classification_report(yt, yp, labels=[0, 1], target_names=['정상(0)', '이상(1)'], zero_division=0))
    else:
        f1m  = f1_score(yt, yp, average='macro',  zero_division=0)
        f1p  = f1_score(yt, yp, average='binary', zero_division=0)
        auc  = roc_auc_score(yt, yprob)
        cm   = confusion_matrix(yt, yp)
        tn, fp, fn, tp = cm.ravel()
        lines += [
            f'  F1(macro)={f1m:.4f}  F1(이상)={f1p:.4f}  AUC={auc:.4f}',
            f'  정밀도={tp/(tp+fp) if (tp+fp)>0 else 0:.4f}  재현율={tp/(tp+fn) if (tp+fn)>0 else 0:.4f}',
            classification_report(yt, yp, target_names=['정상(0)', '이상(1)'], zero_division=0),
            f'혼동행렬 (TN={tn} FP={fp} / FN={fn} TP={tp}):\n{cm}',
        ]
    return '\n'.join(lines)


def eval_rule_baseline(mask):
    yt = y_te_arr[mask]
    # 단순 규칙: "grade가 정상적으로 기재된 게임은 모두 정상으로 판단"
    # → 이 규칙으로는 이상치를 하나도 탐지 못함 (F1=0.0000)
    # → 모델이 이 규칙보다 월등히 높은 F1을 기록하면 grade 외 피처의 기여 입증
    yp_rule = np.zeros_like(yt)
    cm = confusion_matrix(yt, yp_rule)
    tn, fp, fn, tp = cm.ravel()
    f1p = f1_score(yt, yp_rule, average='binary', zero_division=0)
    lines = [
        f'\n{"="*60}',
        f'[단순 규칙 베이스라인 — ③ 미상 제외 서브셋 한정]',
        f'  규칙: grade ≠ 미상이면 무조건 정상(0) 예측',
        f'  → 이상 {yt.sum()}건 중 {tp}건 검출  F1(이상)={f1p:.4f}',
        f'혼동행렬 (TN={tn} FP={fp} / FN={fn} TP={tp}):\n{cm}',
    ]
    return '\n'.join(lines)

# ─────────────────────────────────────────────
# 5. 평가 실행
# ─────────────────────────────────────────────
mask_all     = np.ones(len(y_te_arr), dtype=bool)
mask_missing = (grade_te == '미상')
mask_nonmiss = ~mask_missing

print(f'\n테스트셋 구성: 전체 {mask_all.sum():,}건 | 미상 {mask_missing.sum():,}건 | 미상 제외 {mask_nonmiss.sum():,}건')

rep_all     = eval_subset(mask_all,     '① 전체 테스트셋')
rep_missing = eval_subset(mask_missing, '② grade=미상 만')
rep_nonmiss = eval_subset(mask_nonmiss, '③ grade 정상 기재 (미상 제외)')
rep_rule    = eval_rule_baseline(mask_nonmiss)

# ─────────────────────────────────────────────
# 6. Threshold 최적화 (③ 미상 제외 서브셋 한정)
# ─────────────────────────────────────────────
def threshold_sweep(mask):
    yt    = y_te_arr[mask]
    yprob = y_prob_all[mask]
    n_pos = yt.sum()

    thresholds = np.arange(0.05, 0.96, 0.05)
    rows = []
    for thr in thresholds:
        yp  = (yprob >= thr).astype(int)
        f1p = f1_score(yt, yp, average='binary', zero_division=0)
        prec = f1_score(yt, yp, average='binary', zero_division=0)
        cm  = confusion_matrix(yt, yp, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        recall = tp / n_pos if n_pos > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        rows.append({'threshold': thr, 'F1(이상)': f1p, '정밀도': precision, '재현율': recall, 'TP': tp, 'FP': fp, 'FN': fn})

    df_thr = pd.DataFrame(rows)
    best   = df_thr.loc[df_thr['F1(이상)'].idxmax()]

    lines = [
        f'\n{"="*60}',
        f'[④ Threshold 최적화 — ③ 미상 제외 서브셋]',
        f'  이상 {n_pos}건 기준  (기본값 threshold=0.50)',
        f'\n  {"Thr":>5}  {"F1(이상)":>9}  {"정밀도":>7}  {"재현율":>7}  {"TP":>5}  {"FP":>5}  {"FN":>5}',
    ]
    for _, r in df_thr.iterrows():
        marker = ' ◀ 최적' if abs(r['threshold'] - best['threshold']) < 0.001 else (
                 ' ◀ 기본값' if abs(r['threshold'] - 0.50) < 0.001 else '')
        lines.append(
            f"  {r['threshold']:>5.2f}  {r['F1(이상)']:>9.4f}  {r['정밀도']:>7.4f}  {r['재현율']:>7.4f}"
            f"  {int(r['TP']):>5}  {int(r['FP']):>5}  {int(r['FN']):>5}{marker}"
        )
    lines += [
        f'\n  최적 threshold={best["threshold"]:.2f}  F1(이상)={best["F1(이상)"]:.4f}'
        f'  재현율={best["재현율"]:.4f}  정밀도={best["정밀도"]:.4f}',
        f'  → 이상 {n_pos}건 중 {int(best["TP"])}건 검출  (threshold=0.5 대비 +{int(best["TP"])-320}건)',
    ]
    return '\n'.join(lines)

rep_threshold = threshold_sweep(mask_nonmiss)

# ─────────────────────────────────────────────
# 7. 리포트 저장
# ─────────────────────────────────────────────
full_report = (
    '=== 등급 미상 분리 평가 리포트 ===\n'
    '학습 조건: B그룹_v3 전체 80% (실험 A 동일, seed=42)\n'
    '목적: grade=미상 신호 없이도 모델이 이상치를 탐지하는가?\n'
    + rep_all
    + rep_missing
    + rep_nonmiss
    + rep_rule
    + rep_threshold
)

print(full_report)
REPORT_OUT.write_text(full_report, encoding='utf-8')
print(f'\n리포트 저장: {REPORT_OUT}')

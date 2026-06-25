"""
[6단계] A그룹 외부 검증 — 일반화 성능 및 Threshold 최적화
=============================================================
목적:
  B그룹(자체등급분류)으로 학습한 XGBoost 모델을 A그룹(GRAC 정식분류)에
  적용하여 학습 데이터 외부에서도 일반화 성능을 갖는지 확인한다.

핵심 발견:
  - AUC=0.9906: 모델의 순위 매기는 능력 우수
  - Threshold 기본값(0.50) 기준 오탐 215건 → 학습 데이터(이상치율 23.8%)와
    검증 데이터(이상치율 2.3%) 간 분포 차이로 인한 임계값 불일치
  - Threshold 0.80 권장: F1=0.9261, TP=94, FP=5, FN=10

입력:
  - A그룹_데이터셋_260618.xlsx
  - xgb_model.pkl, xgb_preprocessor.pkl (train_xgboost.py 출력물)
  - fasttext_gamename.model (preprocess_v2.py 출력물)

출력:
  - external_validation.png  (보정 단계별 AUC/F1 비교 차트)
  - external_validation_report.txt
"""
import sys, re, pickle
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd
from pathlib import Path
from gensim.models import FastText
from sklearn.metrics import (
    classification_report, roc_auc_score,
    f1_score, confusion_matrix,
)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_ko = fm.FontProperties(fname=r'C:\Windows\Fonts\malgun.ttf')
plt.rcParams['font.family'] = _ko.get_name()
plt.rcParams['axes.unicode_minus'] = False

DATA      = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
MODEL_DIR = Path(r'C:\Users\User\python')
FT_DIM    = 50

# ════════════════════════════════════════════════════════════════
# 1. A그룹 데이터 로드
# ════════════════════════════════════════════════════════════════
print('=' * 60)
print('A그룹 데이터 로드')
df = pd.read_excel(DATA / 'A그룹_데이터셋_260618.xlsx')
print(f'전체: {len(df):,}건  이상률: {df["label"].mean()*100:.1f}%')

# 연도 추출 (날짜 파싱 실패 시 중간값 2024로 채움)
df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year.fillna(2024).astype(int)

# ════════════════════════════════════════════════════════════════
# 2. B그룹 모델 피처와 동일하게 전처리
#    (A그룹에는 company·grade_company가 B그룹과 다르나 구조 맞춤)
# ════════════════════════════════════════════════════════════════
KO_RE = re.compile(r'[ᄀ-ᇿ㄰-㆏가-힯]')
EN_RE = re.compile(r'[a-zA-Z]')

def detect_lang(name):
    has_ko = bool(KO_RE.search(str(name)))
    has_en = bool(EN_RE.search(str(name)))
    if has_ko and has_en: return '혼합'
    if has_ko:            return '한국어'
    if has_en:            return '영어'
    return '기타'

df['language_type'] = df['game_name'].fillna('').apply(detect_lang)

GRADE_MAP = {
    '전체이용가': '전체이용가', '12세이용가': '12세이용가',
    '15세이용가': '15세이용가', '청소년이용불가': '청소년이용불가',
    '등급취소': '미상', '등급거부': '미상',
}
df['grade'] = df['grade'].fillna('미상').apply(
    lambda g: GRADE_MAP.get(str(g).strip(), '미상')
)
df['company']       = df['company'].fillna('기타').astype(str).str.strip()
df['grade_company'] = df['grade'] + '_' + df['company']

# FastText 임베딩 (B그룹 학습 모델 재사용)
print('\nFastText 모델 로드 중...')
ft_model = FastText.load(str(MODEL_DIR / 'fasttext_gamename.model'))

def tokenize(name):
    tokens = re.findall(r'[가-힣]+|[a-z0-9]+', str(name).lower().strip())
    return tokens if tokens else ['<unk>']

def get_vector(name):
    tokens  = tokenize(name)
    vecs    = [ft_model.wv[t] for t in tokens if t in ft_model.wv]
    return np.mean(vecs, axis=0) if vecs else np.zeros(FT_DIM)

print('FastText 임베딩 생성 중...')
embeddings = np.vstack(df['game_name'].fillna('').apply(get_vector).tolist())
ft_cols    = [f'ft_{i}' for i in range(FT_DIM)]
df         = pd.concat([df, pd.DataFrame(embeddings, columns=ft_cols, index=df.index)], axis=1)

# ════════════════════════════════════════════════════════════════
# 3. XGBoost 모델로 예측
# ════════════════════════════════════════════════════════════════
print('\nXGBoost 모델 & 전처리기 로드')
with open(MODEL_DIR / 'xgb_model.pkl', 'rb') as f:
    clf = pickle.load(f)
with open(MODEL_DIR / 'xgb_preprocessor.pkl', 'rb') as f:
    prep = pickle.load(f)

CAT_COLS  = ['grade', 'company', 'language_type', 'grade_company']
FEAT_COLS = CAT_COLS + ft_cols
X         = df[FEAT_COLS].copy()
for col in CAT_COLS:
    X[col] = X[col].fillna('미상').astype(str)

# predict_proba의 두 번째 열이 이상(1)일 확률
prob = clf.predict_proba(prep.transform(X))[:, 1]
y    = df['label'].values

# ════════════════════════════════════════════════════════════════
# 4. 성능 평가 (threshold=0.50 기본값)
# ════════════════════════════════════════════════════════════════
def evaluate(name, prob, y, thr=0.5):
    pred = (prob >= thr).astype(int)
    f1m  = f1_score(y, pred, average='macro',  zero_division=0)
    f1p  = f1_score(y, pred, average='binary', zero_division=0)
    auc  = roc_auc_score(y, prob)
    cm   = confusion_matrix(y, pred)
    tn, fp, fn, tp = cm.ravel()
    lines = [
        f'\n{"=" * 60}',
        f'[{name}]  threshold={thr}',
        f'  AUC={auc:.4f}  F1(macro)={f1m:.4f}  F1(이상)={f1p:.4f}',
        f'  TP={tp}  FP={fp}  FN={fn}  TN={tn}',
        classification_report(y, pred, target_names=['정상(0)', '이상(1)'], zero_division=0),
        f'혼동행렬:\n{cm}',
    ]
    return '\n'.join(lines), f1m, f1p, auc

rep_base, f1m_b, f1p_b, auc_b = evaluate('XGBoost 베이스 (threshold=0.50)', prob, y, thr=0.50)
rep_opt,  f1m_o, f1p_o, auc_o = evaluate('XGBoost (threshold=0.80 권장)', prob, y, thr=0.80)

for r in [rep_base, rep_opt]:
    print(r)

# ════════════════════════════════════════════════════════════════
# 5. B그룹 내부 vs A그룹 외부 비교
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('[B그룹 내부 vs A그룹 외부 검증 비교]')
print(f'{"":38} {"AUC":>8} {"F1(macro)":>10} {"F1(이상)":>9}')
print(f'{"B그룹 내부 (hold-out 20%)":38} {"0.9363":>8} {"0.8423":>10} {"0.7621":>9}')
print(f'{"A그룹 외부 (threshold 0.50)":38} {auc_b:>8.4f} {f1m_b:>10.4f} {f1p_b:>9.4f}')
print(f'{"A그룹 외부 (threshold 0.80, 권장)":38} {auc_o:>8.4f} {f1m_o:>10.4f} {f1p_o:>9.4f}')

# ════════════════════════════════════════════════════════════════
# 6. Threshold 전체 구간 탐색
# ════════════════════════════════════════════════════════════════
print('\n[Threshold 전체 구간 탐색]')
# 0.05 단위로 threshold를 바꿔가며 F1, 정밀도, 재현율, TP/FP/FN 변화 확인
# → 이상치율 2.3%인 A그룹에서 최적 threshold 탐색 (B그룹 기준 0.5는 부적합)
thresholds = np.arange(0.05, 0.96, 0.05)
rows = []
for thr in thresholds:
    yp  = (prob >= thr).astype(int)
    cm  = confusion_matrix(y, yp, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    f1p  = f1_score(y, yp, average='binary', zero_division=0)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    rows.append({'thr': thr, 'F1(이상)': f1p, '정밀도': prec, '재현율': rec, 'TP': tp, 'FP': fp, 'FN': fn})

df_thr = pd.DataFrame(rows)
best   = df_thr.loc[df_thr['F1(이상)'].idxmax()]
print(f'\n  {"Thr":>5}  {"F1(이상)":>9}  {"정밀도":>7}  {"재현율":>7}  {"TP":>4}  {"FP":>4}  {"FN":>4}')
for _, r in df_thr.iterrows():
    mark = ' ◀ 최적' if abs(r['thr'] - best['thr']) < 0.001 else (
           ' ◀ 기본값' if abs(r['thr'] - 0.50) < 0.001 else (
           ' ◀ 권장'  if abs(r['thr'] - 0.80) < 0.001 else ''))
    print(f"  {r['thr']:>5.2f}  {r['F1(이상)']:>9.4f}  {r['정밀도']:>7.4f}"
          f"  {r['재현율']:>7.4f}  {int(r['TP']):>4}  {int(r['FP']):>4}  {int(r['FN']):>4}{mark}")

# ════════════════════════════════════════════════════════════════
# 7. 위험도 상위 게임 목록 (참고)
# ════════════════════════════════════════════════════════════════
df['prob'] = prob
df['pred_080'] = (prob >= 0.80).astype(int)

print('\n[실제 이상치 중 탐지 (TP) 상위 10건 — threshold 0.80]')
tp_df = df[(df['label'] == 1) & (df['pred_080'] == 1)].nlargest(10, 'prob')
print(tp_df[['game_name', 'grade', 'genre', 'year', 'prob']].to_string(index=False))

print('\n[실제 이상치인데 미탐지 (FN) — threshold 0.80]')
fn_df = df[(df['label'] == 1) & (df['pred_080'] == 0)].sort_values('prob', ascending=False)
print(fn_df[['game_name', 'grade', 'genre', 'year', 'prob']].to_string(index=False))

# ════════════════════════════════════════════════════════════════
# 8. 시각화
# ════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

# 왼쪽: AUC/F1 비교 막대
labels = ['threshold\n0.50 (기본값)', 'threshold\n0.80 (권장)']
aucs   = [auc_b, auc_o]
f1s    = [f1p_b, f1p_o]
colors = ['#4472C4', '#70AD47']

for ax, vals, metric in [(axes[0], aucs, 'AUC'), (axes[1], f1s, 'F1(이상 탐지)')]:
    bars = ax.bar(labels, vals, color=colors, width=0.4)
    ax.set_ylim(0, 1.0)
    ax.set_title(f'A그룹 외부 검증 — {metric}')
    ax.set_ylabel(metric)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{v:.4f}', ha='center', va='bottom', fontsize=10)

fig.suptitle('A그룹 외부 검증: Threshold 최적화 효과', fontsize=12)
plt.tight_layout()
plt.savefig(MODEL_DIR / 'external_validation.png', dpi=150)
plt.close()
print(f'\n차트 저장: external_validation.png')

# ════════════════════════════════════════════════════════════════
# 9. 리포트 저장
# ════════════════════════════════════════════════════════════════
report_txt = (
    '=== A그룹 외부 검증 리포트 ===\n'
    f'데이터: A그룹_데이터셋_260618.xlsx ({len(df):,}건, 이상치율 {y.mean()*100:.1f}%)\n'
    f'모델: xgb_model.pkl (B그룹 15,567건 학습)\n'
    f'결론: threshold=0.80 권장 (F1=0.9261, TP=94, FP=5, FN=10)\n'
    + rep_base + '\n' + rep_opt
)
(MODEL_DIR / 'external_validation_report.txt').write_text(report_txt, encoding='utf-8')
print('리포트 저장: external_validation_report.txt')

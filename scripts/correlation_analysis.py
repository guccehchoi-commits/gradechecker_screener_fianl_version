"""
[STEP 3] B그룹 상관관계 분석
==============================================
독립변수: grade, grade_missing, company, date(연도·월)
종속변수: label (0=정상, 1=이상)

통계 검정:
  - 카이제곱(chi2): 범주형 변수와 label의 독립성 검정
    p < 0.05 → 변수와 이상 여부 사이에 유의미한 관계 존재
  - 점이연상관계수(pointbiserialr): 연속형 변수와 이진 label의 상관관계
    r 절댓값이 클수록 강한 상관, r > 0이면 값이 클수록 이상치 많음
"""
import sys, pandas as pd, numpy as np
from scipy.stats import chi2_contingency, pointbiserialr
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
df = pd.read_excel(DATA / 'B그룹_데이터셋_260618.xlsx')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year
df['month'] = df['date'].dt.month

total = len(df)
n_ab = df['label'].sum()
print(f'B그룹 전체: {total:,}건  이상치: {n_ab:,}건 ({n_ab/total*100:.1f}%)\n')

# ── 카이제곱 검정 함수 ────────────────────────────────────────
def chi2_test(df, col):
    ct = pd.crosstab(df[col], df['label'])
    chi2, p, dof, _ = chi2_contingency(ct)
    return chi2, p

# ════════════════════════════════════════════════════════════════
# 1. grade vs label
# ════════════════════════════════════════════════════════════════
print('=' * 60)
print('1. grade (등급) vs label')
print('=' * 60)
g = df.groupby('grade')['label'].agg(['sum','count'])
g.columns = ['이상','전체']
g['정상'] = g['전체'] - g['이상']
g['재분류율'] = (g['이상'] / g['전체'] * 100).round(1)
g = g.sort_values('재분류율', ascending=False)
print(g[['전체','정상','이상','재분류율']].to_string())
chi2, p = chi2_test(df, 'grade')
print(f'\n카이제곱: {chi2:.1f}  p값: {p:.2e}  {"유의미 ✅" if p < 0.05 else "비유의미"}')

# ════════════════════════════════════════════════════════════════
# 2. grade_missing vs label
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('2. grade_missing (등급 결측 여부) vs label')
print('=' * 60)
gm = df.groupby('grade_missing')['label'].agg(['sum','count'])
gm.columns = ['이상','전체']
gm['재분류율'] = (gm['이상'] / gm['전체'] * 100).round(1)
print(gm[['전체','이상','재분류율']].to_string())
r, p = pointbiserialr(df['grade_missing'], df['label'])
print(f'\n점이연상관계수: {r:.3f}  p값: {p:.2e}  {"유의미 ✅" if p < 0.05 else "비유의미"}')

# ════════════════════════════════════════════════════════════════
# 3. company (플랫폼) vs label
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('3. company (플랫폼) vs label')
print('=' * 60)
c = df.groupby('company')['label'].agg(['sum','count'])
c.columns = ['이상','전체']
c['정상'] = c['전체'] - c['이상']
c['재분류율'] = (c['이상'] / c['전체'] * 100).round(1)
c = c.sort_values('재분류율', ascending=False)
print(c[['전체','정상','이상','재분류율']].to_string())
# company='-'인 경우는 플랫폼 미상 → 카이제곱 검정에서 제외
chi2, p = chi2_test(df[df['company'] != '-'], 'company')
print(f'\n카이제곱: {chi2:.1f}  p값: {p:.2e}  {"유의미 ✅" if p < 0.05 else "비유의미"}')

# ════════════════════════════════════════════════════════════════
# 4. 연도 vs label
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('4. 연도 vs label')
print('=' * 60)
y = df.dropna(subset=['year']).groupby('year')['label'].agg(['sum','count'])
y.columns = ['이상','전체']
y['재분류율'] = (y['이상'] / y['전체'] * 100).round(1)
print(y[['전체','이상','재분류율']].to_string())
df_y = df.dropna(subset=['year'])
r, p = pointbiserialr(df_y['year'], df_y['label'])
print(f'\n점이연상관계수: {r:.3f}  p값: {p:.2e}  {"유의미 ✅" if p < 0.05 else "비유의미"}')

# ════════════════════════════════════════════════════════════════
# 5. 월 vs label
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('5. 월 vs label')
print('=' * 60)
m = df.dropna(subset=['month']).groupby('month')['label'].agg(['sum','count'])
m.columns = ['이상','전체']
m['재분류율'] = (m['이상'] / m['전체'] * 100).round(1)
print(m[['전체','이상','재분류율']].to_string())
df_m = df.dropna(subset=['month'])
r, p = pointbiserialr(df_m['month'], df_m['label'])
print(f'\n점이연상관계수: {r:.3f}  p값: {p:.2e}  {"유의미 ✅" if p < 0.05 else "비유의미"}')

# ════════════════════════════════════════════════════════════════
# 요약
# ════════════════════════════════════════════════════════════════
print('\n' + '=' * 60)
print('요약: 독립변수별 label과의 관련성')
print('=' * 60)
chi2_grade, p_grade = chi2_test(df, 'grade')
chi2_comp, p_comp = chi2_test(df[df['company'] != '-'], 'company')
r_gm, p_gm = pointbiserialr(df['grade_missing'], df['label'])
r_yr, p_yr = pointbiserialr(df.dropna(subset=['year'])['year'], df.dropna(subset=['year'])['label'])

results = [
    ('grade',         f'p={p_grade:.2e}', '✅ 유의미' if p_grade < 0.05 else '❌'),
    ('grade_missing', f'r={r_gm:.3f}, p={p_gm:.2e}', '✅ 유의미' if p_gm < 0.05 else '❌'),
    ('company',       f'p={p_comp:.2e}', '✅ 유의미' if p_comp < 0.05 else '❌'),
    ('연도',           f'r={r_yr:.3f}, p={p_yr:.2e}', '✅ 유의미' if p_yr < 0.05 else '❌'),
]
for name, stat, sig in results:
    print(f'  {name:<16} {stat:<25} {sig}')

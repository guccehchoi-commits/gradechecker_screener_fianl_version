"""A그룹 상관관계 분석 — 통계 검정 추가"""
import sys, pandas as pd
from scipy.stats import chi2_contingency, pointbiserialr
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
df = pd.read_excel(DATA / 'A그룹_데이터셋_260618.xlsx')
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['year'] = df['date'].dt.year

def chi2(df, col):
    ct = pd.crosstab(df[col], df['label'])
    chi2_val, p, dof, _ = chi2_contingency(ct)
    return chi2_val, p

print('=== A그룹 상관관계 분석 ===')

# grade
chi2_g, p_g = chi2(df, 'grade')
print(f'\ngrade: 카이제곱={chi2_g:.1f}, p={p_g:.2e}')
g = df.groupby('grade')['label'].agg(['sum','count'])
g.columns=['이상','전체']; g['재분류율']=(g['이상']/g['전체']*100).round(1)
print(g.sort_values('재분류율',ascending=False)[['전체','이상','재분류율']].to_string())

# genre (결측 제외)
df_g = df.dropna(subset=['genre'])
chi2_genre, p_genre = chi2(df_g, 'genre')
print(f'\ngenre: 카이제곱={chi2_genre:.1f}, p={p_genre:.2e}')
gn = df_g.groupby('genre')['label'].agg(['sum','count'])
gn.columns=['이상','전체']; gn['재분류율']=(gn['이상']/gn['전체']*100).round(1)
print(gn.sort_values('재분류율',ascending=False)[['전체','이상','재분류율']].to_string())

# platform
df_p = df.dropna(subset=['platform'])
chi2_p, p_p = chi2(df_p, 'platform')
print(f'\nplatform: 카이제곱={chi2_p:.1f}, p={p_p:.2e}')
pl = df_p.groupby('platform')['label'].agg(['sum','count'])
pl.columns=['이상','전체']; pl['재분류율']=(pl['이상']/pl['전체']*100).round(1)
print(pl.sort_values('재분류율',ascending=False)[['전체','이상','재분류율']].to_string())

# 연도
r_y, p_y = pointbiserialr(df.dropna(subset=['year'])['year'], df.dropna(subset=['year'])['label'])
print(f'\n연도: r={r_y:.3f}, p={p_y:.2e}')

print('\n=== 요약 ===')
for name, stat, p in [
    ('grade', f'카이제곱={chi2_g:.1f}', p_g),
    ('genre', f'카이제곱={chi2_genre:.1f}', p_genre),
    ('platform', f'카이제곱={chi2_p:.1f}', p_p),
    ('연도', f'r={r_y:.3f}', p_y),
]:
    sig = '✅ 유의미' if p < 0.05 else '❌ 비유의미'
    print(f'  {name:<10} {stat:<22} p={p:.2e}  {sig}')

"""
연도-이상치 상관관계 심층 분석 (상사 가설 검증)
가설: 연도가 오래된 게임물일수록 이상치 가능성이 낮다
"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_ko = fm.FontProperties(fname=r'C:\Windows\Fonts\malgun.ttf')
plt.rcParams['font.family'] = _ko.get_name()
plt.rcParams['axes.unicode_minus'] = False

DATA  = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')
OUT   = Path(r'C:\Users\User\python')

# ────────────────────────────────────────────────────────────────
# 데이터 로드
# ────────────────────────────────────────────────────────────────
a = pd.read_excel(DATA / 'A그룹_데이터셋_260618.xlsx')
b = pd.read_excel(DATA / 'B그룹_데이터셋_260618_v2.xlsx')

for df in [a, b]:
    df['year'] = pd.to_datetime(df['date'], errors='coerce').dt.year

a = a.dropna(subset=['year']); a['year'] = a['year'].astype(int)
b = b.dropna(subset=['year']); b['year'] = b['year'].astype(int)

# ────────────────────────────────────────────────────────────────
# 상관관계 분석
# ────────────────────────────────────────────────────────────────
print('=' * 60)
print('상사 가설 검증: "연도가 오래될수록 이상치 가능성이 낮다"')
print('= 오래된 연도(낮은 숫자) → label=1 비율이 낮을 것으로 예측')
print('= 예측 방향: 연도와 이상치율 간 양의 상관 (r > 0)')
print('=' * 60)

results = {}
for name, df in [('A그룹(GRAC)', a), ('B그룹(자체등급)', b)]:
    r, p = stats.pointbiserialr(df['year'], df['label'])
    tbl = df.groupby('year')['label'].agg(['count','sum','mean'])
    tbl.columns = ['전체','이상','이상률']
    results[name] = {'r': r, 'p': p, 'tbl': tbl}

    print(f'\n[{name}]')
    print(f'  점이연상관계수 r = {r:+.4f},  p = {p:.2e}')
    print(f'  방향: {"양(+) → 연도 높을수록 이상치 많음" if r>0 else "음(-) → 연도 높을수록 이상치 적음"}')
    print(f'  {"★ 통계적으로 유의미 (p<0.05)" if p<0.05 else "통계적으로 유의미하지 않음"}')
    print(f'  실질 상관 강도: {"강" if abs(r)>0.3 else "중" if abs(r)>0.1 else "약"} (|r|={abs(r):.3f})')
    print()
    print('  연도별 이상치율:')
    for yr, row in tbl.iterrows():
        bar = '█' * int(row['이상률'] * 100 / 1.5)
        print(f'    {yr}년 | {row["이상률"]*100:5.1f}% | {bar}  ({int(row["이상"]):,}/{int(row["전체"]):,}건)')

# ────────────────────────────────────────────────────────────────
# 가설 검증 요약
# ────────────────────────────────────────────────────────────────
print()
print('=' * 60)
print('검증 결과 요약')
print('=' * 60)

a_tbl = results['A그룹(GRAC)']['tbl']
b_tbl = results['B그룹(자체등급)']['tbl']

print('''
[A그룹 판단]
 ▶ r = {a_r:+.4f} (음수) → 연도가 높을수록 이상치 약간 적음
   → 가설 방향과 반대 (오래된 게임이 위험하다는 의미)
   → BUT: 2026년 이상치 0%는 "아직 취소 안 됐을 뿐" 수집 편향
   → 2026 제외 시: 2022(2.3%) < 2023(5.0%) > 2024(2.1%) > 2025(1.4%)
   → 2023이 이상치 최고 — 단조 감소 패턴 없음, 연도별 특수 요인 존재 가능성

[B그룹 판단]
 ▶ r = {b_r:+.4f} (양수, 매우 약함)
   → B이상 데이터의 날짜 = "재분류 결정일" (게임 최초 등록일 아님)
   → 정상 게임의 날짜 = "자체등급 취득일"
   → 두 날짜의 의미가 달라 직접 비교 불가 → 연도 피처 신뢰도 낮음

[종합]
 ▶ 통계적으로는 유의미하나 (A그룹 p<0.001), 실질 상관은 매우 약함 (|r|<0.07)
 ▶ B그룹 날짜 구조상 연도를 메인 모델 피처로 추가하는 것은 부적합
 ▶ A그룹 외부 검증 시 연도 기반 보정은 제한적으로만 적용 권장
'''.format(a_r=results['A그룹(GRAC)']['r'], b_r=results['B그룹(자체등급)']['r']))

# ────────────────────────────────────────────────────────────────
# 시각화
# ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, res) in zip(axes, results.items()):
    tbl = res['tbl']
    colors = ['#d9534f' if yr == 2026 else '#4472C4' for yr in tbl.index]
    bars = ax.bar(tbl.index.astype(str), tbl['이상률'] * 100, color=colors)
    ax.set_title(f'{name}\nr={res["r"]:+.4f}, p={res["p"]:.1e}', fontsize=11)
    ax.set_xlabel('등록 연도')
    ax.set_ylabel('이상치율 (%)')
    for bar, (_, row) in zip(bars, tbl.iterrows()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f'{row["이상률"]*100:.1f}%', ha='center', va='bottom', fontsize=9)

axes[0].set_ylim(0, 8)
axes[1].set_ylim(0, 40)

note = '* 2026년(빨간색)은 수집 시점 편향 — 아직 취소되지 않은 게임이 포함된 것으로, 이상치율이 낮게 나타남'
fig.text(0.5, -0.03, note, ha='center', fontsize=8, color='gray')
fig.suptitle('연도별 이상치율 — 상사 가설 검증\n"연도가 오래된 게임일수록 이상치 가능성이 낮다"', fontsize=12)
plt.tight_layout()
plt.savefig(OUT / 'year_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print(f'차트 저장: {OUT / "year_analysis.png"}')

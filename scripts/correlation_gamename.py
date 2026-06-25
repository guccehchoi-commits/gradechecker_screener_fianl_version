"""
[STEP 3] 게임 타이틀(game_name) 키워드 분석
==============================================
분석 방법: lift(상대 빈도비) 계산
  lift = (이상치 내 키워드 비율) / (정상치 내 키워드 비율)
  lift=30 → 이상치에서 30배 더 자주 등장하는 단어
  lift가 높을수록 해당 키워드가 이상치의 강한 신호임을 의미

결과:
  도박·현금 키워드: cash(65배), slot(47배), bitcoin(42배) 등
  폭력성 키워드: shooting(12배), fps(11배), sniper(7배) 등
"""
import sys, pandas as pd, re
from pathlib import Path
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

DATA = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')

def tokenize(title):
    title = str(title or '').lower().strip()
    # 숫자+특수문자 제거, 단어 추출 (한글/영문 2글자 이상)
    tokens = re.findall(r'[a-z]{2,}|[가-힣]{2,}', title)
    return tokens

# 분석 의미 없는 일반 단어 제외 (모든 게임에 공통으로 등장)
STOPWORDS = {
    'the','of','in','for','to','and','a','an','is','on','at','by','with',
    'game','games','게임','mobile','edition','version','online'
}

def analyze(df, name):
    print(f'\n{"="*60}')
    print(f'【{name}】 game_name 분석')
    print(f'{"="*60}')

    normal = df[df['label']==0]['game_name']
    abnorm = df[df['label']==1]['game_name']

    # 전체 토큰 카운트
    cnt_n = Counter()
    for t in normal:
        cnt_n.update([w for w in tokenize(t) if w not in STOPWORDS])

    cnt_a = Counter()
    for t in abnorm:
        cnt_a.update([w for w in tokenize(t) if w not in STOPWORDS])

    total_n = sum(cnt_n.values()) or 1
    total_a = sum(cnt_a.values()) or 1

    # 이상치에서 상대적으로 많이 등장하는 단어 (lift 계산)
    # lift = (이상치 비율) / (정상치 비율)
    all_words = set(cnt_n.keys()) | set(cnt_a.keys())
    lifts = []
    for w in all_words:
        freq_a = cnt_a.get(w, 0)
        freq_n = cnt_n.get(w, 0)
        if freq_a < 3:  # 너무 드문 단어는 통계적으로 불안정 → 최소 3회 이상
            continue
        ratio_a = freq_a / total_a
        # 정상치에 한 번도 없는 단어는 분모 0 방지를 위해 최솟값 설정
        ratio_n = freq_n / total_n if freq_n > 0 else 0.0001
        lift = ratio_a / ratio_n
        lifts.append((w, freq_a, freq_n, round(lift, 2)))

    lifts.sort(key=lambda x: -x[1])  # 이상치 빈도 기준 정렬

    print(f'\n[이상치에서 많이 등장하는 단어 Top 20]')
    print(f'  {"단어":<20} {"이상치":>6} {"정상치":>6} {"lift":>6}')
    print(f'  {"-"*44}')
    for w, fa, fn, lift in lifts[:20]:
        print(f'  {w:<20} {fa:>6} {fn:>6} {lift:>6}')

    # lift 높은 단어 (이상치에 상대적으로 집중)
    high_lift = sorted([x for x in lifts if x[1] >= 5], key=lambda x: -x[3])
    print(f'\n[lift 높은 단어 Top 15 (이상치에 집중, 최소 5회)]')
    print(f'  {"단어":<20} {"이상치":>6} {"정상치":>6} {"lift":>6}')
    print(f'  {"-"*44}')
    for w, fa, fn, lift in high_lift[:15]:
        print(f'  {w:<20} {fa:>6} {fn:>6} {lift:>6}')

    # 타이틀 길이 분석
    df2 = df.copy()
    df2['title_len'] = df2['game_name'].str.len()
    len_n = df2[df2['label']==0]['title_len'].mean()
    len_a = df2[df2['label']==1]['title_len'].mean()
    print(f'\n[타이틀 평균 길이]')
    print(f'  정상: {len_n:.1f}자  이상: {len_a:.1f}자')

df_b = pd.read_excel(DATA / 'B그룹_데이터셋_260618_v2.xlsx')
df_a = pd.read_excel(DATA / 'A그룹_데이터셋_260618.xlsx')

analyze(df_b, 'B그룹')
analyze(df_a, 'A그룹')

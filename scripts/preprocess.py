"""
[STEP 1] 1차 전처리 — A그룹 / B그룹 데이터셋 생성
=======================================================
입력:
  - 1. grac_api_로우데이터_260615.csv        (GRAC 정식분류, A그룹 원본)
  - 2. 자체등급분류 데이터_260617.xlsx        (자체등급분류 정상, B정상 원본)
  - 3. 등급조정 및 등급취소 로우데이터_260608 (등급조정·취소, B이상 원본)

처리 내용:
  - 완전 중복 제거, 기간 필터(2022~2026)
  - 등급 표기 정규화: 만3세이상→전체이용가, 등급취소→미상 등
  - 플랫폼 코드 변환: GOOG→구글 플레이, AAPL→앱스토어 등
  - 라벨 부여: 정상=0, 이상=1

출력:
  - A그룹_데이터셋_260618.xlsx  (4,484건 / 정상 97.7% / 이상 2.3%)
  - B그룹_데이터셋_260618.xlsx  (15,567건 / 정상 76.2% / 이상 23.8%)
"""
import csv, sys, openpyxl
import pandas as pd
from pathlib import Path
from datetime import datetime

sys.stdout.reconfigure(encoding='utf-8', errors='replace')

BASE = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615\전처리 전')
OUT  = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전\데이터 3종_260615')

# B정상 데이터의 등급분류번호(rateno) 앞 4자리가 플랫폼 코드
# 예: GOOG-SG-240315-0001 → 구글 플레이
# GRAC 공식 코드 기준으로 매핑
RATENO_PREFIX_MAP = {
    'AAPL': '앱스토어',
    'GOOG': '구글 플레이',
    'SSCP': '플레이스테이션 스토어',
    'SECL': '갤럭시 스토어',
    'MSFT': '마이크로소프트 스토어',
    'OCUL': '메타 호라이즌 스토어',
    'NTDO': '닌텐도 e숍',
    'ONIA': '원스토어',
    'ONES': '원스토어',
    'FORT': '포트나이트',
    'EPIA': '에픽게임즈 스토어',
    'EPIC': '에픽게임즈 스토어',
    'SGHS': 'STOVE',
    'KAGA': '카카오게임즈',
    'SIEK': '플레이스테이션 스토어',
}

def company_from_rateno(rateno):
    pfx = str(rateno or '').split('-')[0]
    return RATENO_PREFIX_MAP.get(pfx, pfx)

COMPANY_MAP = {
    # 구글 플레이
    'Google':           '구글 플레이',
    'GOOGLE':           '구글 플레이',
    'Google Play':      '구글 플레이',
    '구글 (간소화)':     '구글 플레이',
    # 앱스토어
    'Apple':            '앱스토어',
    'APPLE':            '앱스토어',
    # 플레이스테이션 스토어
    'Sony':             '플레이스테이션 스토어',
    '소니':             '플레이스테이션 스토어',
    # 갤럭시 스토어
    'Samsung Electronics':  '갤럭시 스토어',
    'Samsung Eelctronics':  '갤럭시 스토어',
    'Samsung Eelctroncis':  '갤럭시 스토어',
    '삼성전자':              '갤럭시 스토어',
    # 마이크로소프트 스토어
    'Microsoft':        '마이크로소프트 스토어',
    '마이크로소프트':    '마이크로소프트 스토어',
    # 메타 호라이즌 스토어
    'Oculus':           '메타 호라이즌 스토어',
    # 닌텐도 e숍
    '닌텐도':           '닌텐도 e숍',
    # 에픽게임즈 스토어
    '에픽':             '에픽게임즈 스토어',
    '에픽게임즈':       '에픽게임즈 스토어',
    # 원스토어
    'ONE store':        '원스토어',
    '원스토어 (간소화)': '원스토어',
    # STOVE
    '스마일게이트홀딩스':   'STOVE',
    '스마일게이트 홀딩스':  'STOVE',
    # 기타
    '구글,애플 (간소화)': '구글 플레이,앱스토어',
    '토스앱(간소화)':    '토스앱',
}

def norm_company(c):
    c = str(c or '').strip()
    return COMPANY_MAP.get(c, c)

# 국가별·플랫폼별로 등급 표기가 다양하므로 GRAC 공식 5개 등급으로 통일
# 등급취소·등급거부는 더 이상 유효하지 않은 등급 → 미상 처리
GRADE_MAP = {
    '4+': '전체이용가', '9+': '전체이용가', '7세이상': '전체이용가', '전체이용가': '전체이용가',
    '12+': '12세이용가', '12세이용가': '12세이용가', '12세이상이용가': '12세이용가',
    '15+': '15세이용가', '15세이용가': '15세이용가', '15세이상이용가': '15세이용가',
    '17+': '청소년이용불가', '18+': '청소년이용불가',
    '청소년이용불가': '청소년이용불가', '청소년이용 불가': '청소년이용불가',
    '만3세이상': '전체이용가', '3+': '전체이용가',
    '만12세이상': '12세이용가',
    '등급취소': '미상', '등급거부': '미상',
}

def norm_grade(g):
    g = str(g or '').strip().replace('＋', '+').replace(' ', '').replace(' ', '')
    if not g or g in ('-', '없음', 'nan', 'None'):
        return '미상'
    return GRADE_MAP.get(g, g)

def parse_date(s):
    s = str(s or '').strip()
    if not s or s in ('None', 'nan', ''):
        return None
    for fmt in ('%Y-%m-%d', '%Y/%m/%d', '%Y.%m.%d', '%Y%m%d'):
        try:
            return datetime.strptime(s, fmt).strftime('%Y-%m-%d')
        except:
            pass
    return None

def date_from_rateno(rateno):
    # XXXX-SG-YYMMDD-NNNN → 20YY-MM-DD
    parts = str(rateno or '').split('-')
    if len(parts) >= 3:
        d = parts[2]
        if len(d) == 6 and d.isdigit():
            return f'20{d[:2]}-{d[2:4]}-{d[4:6]}'
    return None

def in_range(date_str):
    if not date_str:
        return False
    try:
        return 2022 <= int(date_str[:4]) <= 2026
    except:
        return False


# ════════════════════════════════════════════════════════════════
# A그룹: GRAC API 로우데이터
# ════════════════════════════════════════════════════════════════
print('=' * 60)
print('A그룹 전처리')
print('=' * 60)

rows_a = []
with open(BASE / '1. grac_api_로우데이터_260615.csv', encoding='utf-8-sig', newline='') as f:
    for row in csv.DictReader(f):
        rows_a.append(row)
print(f'원본: {len(rows_a):,}건')

# STEP 7: 완전 중복 제거
seen = set()
deduped = []
for r in rows_a:
    key = tuple(r.values())
    if key not in seen:
        seen.add(key)
        deduped.append(r)
print(f'완전중복 제거: -{len(rows_a) - len(deduped)}건 → {len(deduped):,}건')
rows_a = deduped

# STEP 8: 기간 필터 2022~2026
for r in rows_a:
    r['_date'] = parse_date(r.get('rateddate'))
rows_a_filtered = [r for r in rows_a if in_range(r['_date'])]
print(f'기간 필터(2022~2026): -{len(rows_a) - len(rows_a_filtered)}건 → {len(rows_a_filtered):,}건')
rows_a = rows_a_filtered

# STEP 7: rateno 기준 중복제거 (게임 단위 통합)
seen_rno = {}
for r in rows_a:
    rno = r.get('rateno', '')
    if rno not in seen_rno:
        seen_rno[rno] = r
before = len(rows_a)
rows_a = list(seen_rno.values())
print(f'rateno 중복제거: -{before - len(rows_a)}건 → {len(rows_a):,}건')

# STEP 2 + 3 + 4 + 6: 라벨링, 컬럼명 표준화, 등급 정규화, 불필요 컬럼 제거
# 제거: orgname, summary, descriptors, canceleddate, cancelstatus, rateno
# 유지: game_name, company, grade, genre, platform, date, label
records_a = []
for r in rows_a:
    records_a.append({
        'game_name': str(r.get('gametitle') or '').strip(),
        'company':   str(r.get('entname') or '').strip(),
        'grade':     norm_grade(r.get('givenrate')),
        'genre':     str(r.get('genre') or '').strip(),
        'platform':  str(r.get('platform') or '').strip(),
        'date':      r['_date'],
        # cancelstatus=True인 게임이 등급취소된 게임 → 이상치(1)
        'label':     1 if str(r.get('cancelstatus', '')).strip().lower() == 'true' else 0,
    })

df_a = pd.DataFrame(records_a)
lbl = df_a['label'].value_counts().to_dict()
print(f'label: 정상(0)={lbl.get(0,0):,}건, 이상(1)={lbl.get(1,0):,}건')
print(f'A그룹 최종: {len(df_a):,}건\n')


# ════════════════════════════════════════════════════════════════
# B그룹: B정상(자체등급분류) + B이상(등급조정·취소)
# ════════════════════════════════════════════════════════════════
print('=' * 60)
print('B정상 전처리')
print('=' * 60)

wb = openpyxl.load_workbook(BASE / '2. 자체등급분류 데이터_260617.xlsx')
ws = wb.active
hdrs = [c.value for c in ws[1]]
b_normal_src = [dict(zip(hdrs, [c.value for c in r])) for r in ws.iter_rows(min_row=2)]
# IR- 로 시작하는 번호는 실제 게임이 아닌 시스템 테스트용 데이터 → 제거
b_normal_src = [r for r in b_normal_src if not str(r.get('rateno') or '').startswith('IR-')]
print(f'원본(grade 공백·IR 테스트 제거 후): {len(b_normal_src):,}건')

records_b_normal = []
for r in b_normal_src:
    # date: rateddate 우선, 없으면 rateno에서 추출
    date = parse_date(r.get('rateddate'))
    if not date:
        date = date_from_rateno(r.get('rateno'))
    records_b_normal.append({
        'game_name':     str(r.get('gametitle') or '').strip(),
        'company':       company_from_rateno(r.get('rateno')),
        'grade':         norm_grade(r.get('grade')),
        'grade_missing': 0,
        'date':          date,
        'label':         0,
    })

df_b_normal = pd.DataFrame(records_b_normal)
print(f'B정상 최종: {len(df_b_normal):,}건\n')


print('=' * 60)
print('B이상 전처리')
print('=' * 60)

wb2 = openpyxl.load_workbook(
    BASE / '3. 등급조정 및 등급취소 로우데이터_260608'
         / '0. 등급조정 및 등급취소 로우데이터_260608.xlsx'
)
ws2 = wb2.active
hdrs2 = [c.value for c in ws2[1]]
b_abnormal_src = [dict(zip(hdrs2, [c.value for c in r])) for r in ws2.iter_rows(min_row=2)]
print(f'원본: {len(b_abnormal_src):,}건')

# 번호 결측 처리: 등급분류번호 '-'인 경우 게임명 기준으로 중복 확인
# (별도 dedup 없이 전체 유지, 번호 '-' 행은 그대로 포함)
no_num = sum(1 for r in b_abnormal_src if str(r.get('등급분류번호') or '').strip() == '-')
print(f'등급분류번호 \"-\" (게임명 기준 구분 대상): {no_num:,}건')

# 토스앱: 게임이 아닌 금융앱 → 제거
# 구글,애플(간소화): 동시 등록 게임으로 개별 플랫폼 구분 불가 → 제거
b_abnormal_src = [r for r in b_abnormal_src
                  if str(r.get('자체등급분류사업자') or '').strip() not in ('토스앱(간소화)', '구글,애플 (간소화)')]
print(f'토스앱·구글,애플 제거 후: {len(b_abnormal_src):,}건')

records_b_abnormal = []
for r in b_abnormal_src:
    grade_raw = r.get('자체등급')
    is_missing = (not str(grade_raw or '').strip()) or str(grade_raw) in ('None', 'nan', '-', 'none')
    records_b_abnormal.append({
        'game_name':     str(r.get('게임물명') or '').strip(),
        'company':       norm_company(r.get('자체등급분류사업자')),
        'grade':         '미상' if is_missing else norm_grade(grade_raw),
        'grade_missing': 1 if is_missing else 0,
        'date':          parse_date(r.get('결정일자')),
        'label':         1,
    })

df_b_abnormal = pd.DataFrame(records_b_abnormal)
gm = df_b_abnormal['grade_missing'].value_counts().to_dict()
print(f'grade_missing: 있음(1)={gm.get(1,0):,}건, 없음(0)={gm.get(0,0):,}건')
print(f'B이상 최종: {len(df_b_abnormal):,}건\n')


# ════════════════════════════════════════════════════════════════
# B그룹 합치기
# ════════════════════════════════════════════════════════════════
df_b = pd.concat([df_b_normal, df_b_abnormal], ignore_index=True)
total_b = len(df_b)
ab_b = df_b['label'].sum()
print('=' * 60)
print(f'B그룹 최종: {total_b:,}건 (정상={total_b-ab_b:,} / 이상={ab_b:,} / 이상치비율={ab_b/total_b*100:.1f}%)')
print('=' * 60)


# ════════════════════════════════════════════════════════════════
# 저장
# ════════════════════════════════════════════════════════════════
today = '260618'
out_a = OUT / f'A그룹_데이터셋_{today}.xlsx'
out_b = OUT / f'B그룹_데이터셋_{today}.xlsx'

df_a.to_excel(out_a, index=False)
print(f'\n저장 완료: {out_a.name}')
df_b.to_excel(out_b, index=False)
print(f'저장 완료: {out_b.name}')

print('\n[컬럼 구조]')
print(f'A그룹: {list(df_a.columns)}')
print(f'B그룹: {list(df_b.columns)}')

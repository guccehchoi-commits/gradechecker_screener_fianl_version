"""
GRAC 자체등급분류 게임물 조회 → 테스트용 데이터셋 생성 (Playwright 방식)
2026-01-01 ~ 2026-06-20 전 등급 수집
출력: 테스트_데이터셋_크롤링.xlsx
"""
import asyncio, re, sys, calendar
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pathlib import Path
from bs4 import BeautifulSoup
from playwright.async_api import async_playwright
import pandas as pd

URL = 'https://www.grac.or.kr/Statistics/SelfRateGameStatistics.aspx'
OUT_DIR  = Path(r'Z:\사무국\2. 정책기획팀\4. 센터 주관 사업\2026\6. 2026 문화 디지털혁신 및 데이터 활용 공모전')
OUT_FILE = OUT_DIR / '테스트_데이터셋_크롤링.xlsx'

GRADES = [
    ('전체이용가',    '01'),
    ('12세이용가',    '04'),
    ('15세이용가',    '03'),
    ('청소년이용불가', '02'),
    ('등급거부',      '05'),
    ('4+',           '06'),
    ('9+',           '07'),
    ('12+',          '08'),
    ('17+',          '09'),
]

GRADE_NORM = {
    '전체이용가': '전체이용가', '4+': '전체이용가', '9+': '전체이용가',
    '12세이용가': '12세이용가', '12+': '12세이용가',
    '15세이용가': '15세이용가',
    '청소년이용불가': '청소년이용불가', '17+': '청소년이용불가',
    '등급거부': '미상', '등급취소': '미상',
}

GRADE_IMG = {
    'rating_all':  '전체이용가',
    'rating_12':   '12세이용가',
    'rating_15':   '15세이용가',
    'rating_18':   '청소년이용불가',
    'rating_deny': '등급거부',
    'rating_4':    '4+',
    'rating_9':    '9+',
    'rating_12p':  '12+',
    'rating_17':   '17+',
}

# 2026년 월별 수집 범위
MONTHS_2026 = [(2026, m) for m in range(1, 7)]   # 1월~6월
# 6월은 20일까지
def month_range(year, month):
    if year == 2026 and month == 6:
        return f'{year}-{month:02d}-01', f'{year}-{month:02d}-20'
    last = calendar.monthrange(year, month)[1]
    return f'{year}-{month:02d}-01', f'{year}-{month:02d}-{last:02d}'


def img_to_grade(td):
    if td is None:
        return ''
    img = td.find('img')
    if not img:
        return td.get_text(strip=True)
    alt = img.get('alt', '').strip()
    if alt:
        return alt
    stem = Path(img.get('src', '')).stem
    return GRADE_IMG.get(stem, stem)


def parse_html(html):
    soup = BeautifulSoup(html, 'html.parser')
    items = []
    rows = soup.select("table.statistics tr, table[class*='board'] tr")
    if not rows:
        rows = soup.find_all('tr', id=lambda x: x and 'rptGradeDoc' in x)
    for row in rows:
        cols = row.find_all('td')
        if not cols:
            continue
        item = {}
        grade_td = None
        cancelled_td = None
        for td in cols:
            label = td.get('data-label', '').strip()
            if '번호' in label:
                item['rateno'] = td.get_text(strip=True)
            elif '게임물명' in label or '게임명' in label:
                item['game_name'] = td.get_text(strip=True)
            elif '일자' in label or '분류일' in label:
                item['date'] = td.get_text(strip=True)
            elif '장르' in label:
                item['genre'] = td.get_text(strip=True)
            elif '신청자' in label or '사업자' in label or '플랫폼' in label:
                item['company'] = td.get_text(strip=True)
            elif '취소' in label:
                cancelled_td = td
            elif '등급' in label and '취소' not in label:
                grade_td = td
        item['grade_raw'] = img_to_grade(grade_td)
        if cancelled_td is not None:
            cancelled_text = cancelled_td.get_text(strip=True)
            item['label'] = 1 if cancelled_text in ('Y', '예', '취소', '등급취소') else 0
        else:
            item['label'] = 0
        if item.get('game_name') or item.get('rateno'):
            items.append(item)
    return items


def get_total_pages(soup):
    # p.total-count 또는 페이지 링크 숫자에서 추출
    total_el = soup.select_one('p.total-count')
    if total_el:
        m = re.search(r'(\d[\d,]*)\s*/\s*(\d[\d,]*)', total_el.get_text())
        if m:
            return int(m.group(2).replace(',', ''))
    # 페이지 링크 숫자 방식
    nums = []
    for a in soup.select('a.bgNone span'):
        t = a.get_text(strip=True)
        if t.isdigit():
            nums.append(int(t))
    if nums:
        return max(nums)
    return 1


async def safe_content(page, retries=5):
    for _ in range(retries):
        try:
            content = await page.content()
            if content and len(content) > 500:
                return content
        except Exception:
            pass
        await asyncio.sleep(2)
    return ''


async def do_search(page, grade_val, start, end):
    await page.evaluate("""
        document.querySelector("select[name='ctl00$ContentHolder$ddlGrade']").value = '%s';
        document.querySelector("input[name='ctl00$ContentHolder$CalendarPicker$txtCalStartDate']").value = '%s';
        document.querySelector("input[name='ctl00$ContentHolder$CalendarPicker$txtCalEndDate']").value = '%s';
    """ % (grade_val, start, end))
    await asyncio.sleep(0.5)
    await page.evaluate("document.getElementById('ctl00_ContentHolder_lbtnSearch').click()")
    try:
        await page.wait_for_selector("table.statistics, td.nodata, table.board", timeout=25000)
    except Exception:
        await asyncio.sleep(5)
    await asyncio.sleep(0.8)


async def click_page_num(page, target_num):
    target = str(target_num)
    links = await page.query_selector_all('a.bgNone')
    for link in links:
        span = await link.query_selector('span')
        if span and (await span.inner_text()).strip() == target:
            async with page.expect_navigation(wait_until='domcontentloaded', timeout=20000):
                await link.click()
            await asyncio.sleep(0.8)
            return True
    return False


async def main():
    all_rows = []
    seen_ratenums = set()

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(locale='ko-KR')
        page    = await context.new_page()
        page.set_default_timeout(60000)

        for grade_name, grade_val in GRADES:
            grade_rows = 0
            print(f'\n[{grade_name}]', flush=True)

            for year, month in MONTHS_2026:
                start, end = month_range(year, month)
                print(f'  {year}-{month:02d} ({start}~{end}) 검색 중...', flush=True)

                try:
                    await page.goto(URL, wait_until='domcontentloaded', timeout=60000)
                    await asyncio.sleep(1)
                    await do_search(page, grade_val, start, end)

                    html = await safe_content(page)
                    if not html or '검색된 내용이 없습니다' in html:
                        print(f'    결과없음', flush=True)
                        continue

                    soup = BeautifulSoup(html, 'html.parser')
                    total_pages = get_total_pages(soup)
                    print(f'    총 {total_pages}페이지', flush=True)

                    # 1페이지
                    items = parse_html(html)
                    added = 0
                    for item in items:
                        rn = item.get('rateno', '')
                        if rn not in seen_ratenums:
                            seen_ratenums.add(rn)
                            item['grade_searched'] = grade_name
                            all_rows.append(item)
                            added += 1
                    print(f'    p1: {added}건', flush=True)

                    # 2페이지 이후
                    for pg in range(2, total_pages + 1):
                        ok = await click_page_num(page, pg)
                        if not ok:
                            print(f'    p{pg} 클릭 실패', flush=True)
                            break
                        pg_html = await safe_content(page)
                        pg_items = parse_html(pg_html)
                        if not pg_items:
                            break
                        new = 0
                        for item in pg_items:
                            rn = item.get('rateno', '')
                            if rn not in seen_ratenums:
                                seen_ratenums.add(rn)
                                item['grade_searched'] = grade_name
                                all_rows.append(item)
                                new += 1
                        added += new
                        print(f'    p{pg}: +{new}건', flush=True)

                    grade_rows += added

                except Exception as e:
                    print(f'    오류: {str(e)[:100]}', flush=True)
                    await asyncio.sleep(3)
                    continue

                await asyncio.sleep(0.5)

            print(f'  → {grade_name} 소계: {grade_rows}건', flush=True)

        await browser.close()

    print(f'\n크롤링 완료: 총 {len(all_rows)}건', flush=True)

    if not all_rows:
        print('수집된 데이터 없음')
        return

    df = pd.DataFrame(all_rows)
    print(f'샘플:\n{df.head(3).to_string()}', flush=True)

    # 전처리
    if 'game_name' in df.columns:
        df['game_name'] = df['game_name'].str.strip()
        df = df[df['game_name'].str.len() > 0]

    if 'grade_raw' in df.columns:
        df['grade'] = df['grade_raw'].apply(
            lambda g: GRADE_NORM.get(str(g).strip(), str(g).strip() or '미상')
        )
    elif 'grade_searched' in df.columns:
        df['grade'] = df['grade_searched'].apply(
            lambda g: GRADE_NORM.get(str(g).strip(), str(g).strip())
        )

    if 'company' not in df.columns:
        df['company'] = ''
    df['company'] = df['company'].fillna('').str.strip()

    # 중복 제거
    before = len(df)
    df = df.drop_duplicates(subset=['game_name']).reset_index(drop=True)
    print(f'중복 제거: {before} → {len(df)}건', flush=True)

    # 저장 컬럼
    save_cols = [c for c in ['game_name', 'grade', 'company', 'date', 'label'] if c in df.columns]
    df_save = df[save_cols].copy()

    print('\n=== 등급 분포 ===')
    if 'grade' in df_save.columns:
        print(df_save['grade'].value_counts().to_string())
    if 'label' in df_save.columns:
        n0 = (df_save['label'] == 0).sum()
        n1 = (df_save['label'] == 1).sum()
        print(f'\n라벨 — 정상:{n0}건 / 이상:{n1}건')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_save.to_excel(OUT_FILE, index=False)
    print(f'\n저장 완료: {OUT_FILE}')
    print(f'최종: {len(df_save):,}건')


if __name__ == '__main__':
    asyncio.run(main())

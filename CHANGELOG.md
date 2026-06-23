# CHANGELOG — GUCC 게임 등급재분류 이상탐지 대시보드

배포 레포: https://github.com/guccehchoi-commits/gradechecker_screener_fianl_version  
배포 URL: https://gucc-screener.streamlit.app

---

## [v0.3.1] — 2026-06-23

### Fixed
- SHAP 분석 페이지 한글 글자 깨짐 현상 근본 해결
  - 원인: matplotlib이 Streamlit Cloud(Linux) 환경에서 Windows 한글 폰트(맑은 고딕)를 찾지 못함
  - 해결: matplotlib 전면 제거 → Plotly로 교체 (HTML/SVG 렌더링, 별도 폰트 불필요)
- `requirements.txt`: `matplotlib` 삭제, `plotly` 추가

### Changed — SHAP 페이지 디자인 전면 개편
- **Global SHAP 차트**: 인터랙티브 수평 막대, 항목 유형별 3색 구분(등급미상·FastText·기타), hover 툴팁
- **Local SHAP 차트**: 발산형 수평 막대, 방향 annotation(위험도 ▲/▼), 위험도 카드(metric 3개)
- 차트 배경 투명 처리 → 앱 테마와 자연스럽게 통합
- 범례 차트 하단 직접 표시
- 탐지 사유 한국어 자동 요약

### Files Changed
- `pages/3_🔍_SHAP분석.py` (전면 재작성)
- `requirements.txt`

### Backup
- 이전 버전: `backup/before-shap-threshold-refactor` 브랜치

---

## [v0.3.0] — 2026-06-23

### Added
- 사이드바 Threshold 슬라이더 전역 배치 (`app.py`)
  - 민감도 수준별 설명 텍스트 (폭넓게 탐지 / 권장 설정 ★ / 정밀 탐지 / 고정밀 모드)
  - Threshold별 예상 재현율·정밀도 힌트 표시
- 홈화면 `Threshold란?` 안내 expander 추가
- 예측결과 페이지: 탐지 건수 실시간 요약 배너 (`기준값 0.40 → 검토 대상 N건`)
- SHAP 페이지: Local 분석에 Threshold 맥락 표시 (`✅ 검토 대상` / `ℹ️ 검토 대상 아님`)
- SHAP 페이지: 규칙 기반 보정 내역 섹션 (도박 키워드 감지, 장르 부스트 여부 명시)

### Fixed
- SHAP Local 분석 인덱스 버그 수정
  - 원인: `sv[row_idx]`에 pandas label index를 그대로 사용 → 엉뚱한 게임 SHAP 표시
  - 해결: `df_feat.index.get_loc()` 으로 integer position 변환 후 사용
- FastText ft_0~ft_49 차원 개별 노출 문제 해결
  - 원인: 50개 추상 차원이 피처 목록에 그대로 노출 (예: `remainder__ft_23`)
  - 해결: ft 차원 전체를 `게임명 텍스트 (FastText)` 단일 항목으로 집계

### Changed
- `utils/model.py`: `risk_label(p, thr=0.40)` — Threshold 파라미터 추가, 라벨 3종 정리
  - 기존: 고위험(≥0.80) / 중위험(≥0.50) / 저위험
  - 변경: 고위험(≥0.80) / 검토 대상(≥thr) / 이상 없음
- SHAP 피처명 한국어 변환 (`pretty_name()` 함수)
  - `cat__grade__미상` → `등급: 미상 ★`
  - `cat__company__EA` → `제조사: EA`
  - `cat__language_type__한국어` → `게임명 언어: 한국어`
  - `cat__grade_company__미상_EA` → `등급+제조사: 미상 ★ / EA`
- 예측결과 페이지: 필터 라벨 한국어화, 위험도 컬럼 Threshold 연동

### Files Changed
- `app.py`
- `pages/2_📊_예측결과.py`
- `pages/3_🔍_SHAP분석.py`
- `utils/model.py`

### Backup
- 이전 버전: `backup/before-shap-threshold-refactor` 브랜치 (GitHub)
- 로컬 백업: `C:\Users\User\python\GUCC\deploy_backup_20260623\`

---

## [v0.2.0] — 2026-06-22

### Added
- 분석 완료 시 Google Sheets 자동 저장
  - Apps Script doPost 엔드포인트 연동 (`requests.post`)
  - 저장 항목: game_name, grade, company, prob, risk, uploaded_at
  - Streamlit Secret `SHEETS_URL`로 URL 관리
  - 저장 실패 시 silent fail 처리 (앱 동작에 영향 없음)

### Files Changed
- `pages/1_📂_파일업로드.py`

---

## [v0.1.4] — 2026-06-22

### Fixed
- Python 3.12 고정 (`runtime.txt`) — llvmlite가 Python 3.14와 호환되지 않아 빌드 실패
- `packages.txt` 추가 — 시스템 의존성 명시
- `requirements.txt` 버전 핀 전면 제거 — Streamlit Cloud 빌드 충돌 방지

---

## [v0.1.0] — 2026-06-22

### Added
- 초기 배포: GUCC 게임 등급재분류 이상탐지 대시보드
- 파일 업로드 (Excel/CSV), 컬럼 자동 감지 및 매핑 UI
- XGBoost 모델 예측 (재분류 확률 0~1)
- 장르 부스트 (+0.05), 도박 키워드 부스트 (+0.20) 규칙 적용
- 예측 결과 테이블 (위험도 색상 강조, CSV/Excel 다운로드)
- SHAP 분석 페이지 (Global 피처 중요도, Local 탐지 근거)
- gensim 제거 → `word_vectors.pkl` (dict) 로 대체 (Streamlit Cloud 빌드 호환)
- Streamlit Cloud 배포: https://gucc-screener.streamlit.app

### Model Info
- 알고리즘: XGBoost
- 외부 검증 AUC: 0.9906 (A그룹 4,484건), 0.9086 (신규 1,420건)
- 채택 Threshold: 0.40 (재현율 72.9%, FN 최소화 우선)
- 공모전 제출 기준 F1: 0.743 (Threshold 0.50)

# GUCC Grade Checker
### 자체등급분류 게임물 등급재분류 가능성 예측 서비스

> 2026 문화체육관광 인공지능·데이터 활용 공모전 — 데이터 분석 분야 제출작  
> 주관: 게임이용자보호센터(GUCC)

---

## 서비스 소개

**Grade Checker**는 게임물관리위원회(GRAC) 공공 데이터를 기반으로, 자체등급분류 게임물의 등급재분류 가능성을 AI가 사전 예측하는 스크리닝 서비스입니다.

연간 100만 건 이상 출시되는 게임물 전수를 수작업으로 모니터링하는 것은 물리적으로 불가능합니다. 본 서비스는 XGBoost + FastText 임베딩 기반 모델로 고위험 게임물을 자동 선별하여, 담당자가 우선순위를 두고 검토할 수 있도록 지원합니다.

🔗 **배포 서비스**: [gucc-screener.streamlit.app](https://gucc-screener.streamlit.app)

---

## 주요 기능

| 기능 | 설명 |
|---|---|
| 파일 업로드 | Excel / CSV 드래그 업로드, 컬럼 자동 감지 |
| 등급재분류 확률 예측 | XGBoost + FastText 임베딩, 위험도(고/중/저) 분류 |
| 결과 다운로드 | Excel / CSV 즉시 다운로드 |
| 탐지 근거 분석 | SHAP 피처 중요도, 게임별 히트맵, Waterfall 차트 |
| AI 심층분석 | HuggingFace Inference API (Qwen2.5-7B) 기반 자동 요약 |
| 결과 자동 저장 | Google Sheets 연동 (게임명·등급·확률·위험도·일시) |

---

## 모델 성능

| 검증 단계 | 데이터 | AUC | F1(이상) |
|---|---|---|---|
| 내부 검증 | B그룹 hold-out 3,114건 | 0.9363 | 0.7621 |
| 외부 검증 | A그룹 GRAC 정식분류 4,484건 | 0.9906 | 0.9261 |
| 신규 외부 검증 | 2026 크롤링 + 이상 321건 (1,420건) | 0.9086 | 0.743 |

**현장 운영 Threshold: 0.40** (재현율 72.9% 확보 — 미탐지 비용 최소화 우선)

---

## 기술 스택

| 구분 | 내용 |
|---|---|
| UI | Streamlit 멀티페이지, Plotly, Pandas |
| 모델 | XGBoost (scale_pos_weight 3.2, n_estimators 261), SHAP, scikit-learn |
| 데이터 | FastText 임베딩 50차원 (10,237 어휘), NumPy, openpyxl |
| 외부 API | HuggingFace Inference API (Qwen2.5-7B), Google Sheets API (Apps Script) |
| 배포 | Python 3.10+, Streamlit Community Cloud, Pickle 모델 직렬화 |

---

## 파일 구조

```
deploy/                         # Streamlit Cloud 배포 버전
├── app.py                      # 메인 페이지 (Threshold 슬라이더, 서비스 소개)
├── pages/
│   ├── 1_📂_파일업로드.py        # 파일 업로드 · 컬럼 매핑 · 분석 실행
│   ├── 2_📊_예측 결과.py         # 위험도 분류 결과 테이블 · 다운로드
│   └── 3_🔍_예측_결과_분석.py    # SHAP 분석 · AI 심층분석
├── utils/
│   ├── model.py                # 모델 로드 · 예측 · 위험도 레이블
│   ├── preprocess.py           # 전처리 · FastText 임베딩 · 보정 로직
│   ├── footer.py               # 공모전 안내 푸터 공통 컴포넌트
│   └── logo.py                 # GUCC 로고 사이드바 공통 컴포넌트
├── models/
│   ├── xgb_model.pkl           # 학습된 XGBoost 모델
│   ├── xgb_preprocessor.pkl    # scikit-learn 전처리 파이프라인
│   └── word_vectors.pkl        # FastText 임베딩 벡터 (dict, gensim 없이 동작)
├── requirements.txt
├── packages.txt
└── runtime.txt

scripts/                        # 분석 스크립트 (로컬 실행용)
├── crawl_grac.py               # STEP 0: GRAC 자체등급분류 크롤링
├── preprocess.py               # STEP 1: 원본 3종 전처리
├── preprocess_v2.py            # STEP 2: 파생변수 · FastText 임베딩 생성
├── correlation_analysis.py     # STEP 3: 카이제곱·점이연상관·lift 분석
├── correlation_analysis_a.py   # STEP 4: A그룹 상관관계 분석
├── year_analysis.py            # STEP 5: 연도-이상치 심층 분석
├── train_baseline.py           # STEP 6: Logistic Regression 베이스라인
├── train_xgboost.py            # STEP 7: XGBoost + SHAP 학습
├── evaluate_grade_split.py     # STEP 8: 등급 미상 분리 평가
├── make_test_dataset.py        # STEP 9: 외부 검증 데이터셋 구성
├── validate_external.py        # STEP 10: A그룹 · 신규 외부 검증
├── calc_threshold_metrics.py   # STEP 11: Threshold별 성능 산출
└── validate_resampled.py       # STEP 12: A그룹 이상비율 기준 리샘플링 검증 ×30회
```

---

## 로컬 실행 방법

```bash
# 1. 패키지 설치
pip install -r requirements.txt

# 2. 대시보드 실행 (배포 버전)
streamlit run deploy/app.py
```

> **주의**: `scripts/` 내 분석 스크립트는 `gensim` 필요 (로컬 전용).  
> 배포 버전(`deploy/`)은 gensim 없이 `word_vectors.pkl`(dict)로 동작.

---

## 주요 분석 인사이트

- **등급 미입력(미상)**: 100% 재분류 대상 — 모델 최강 예측 신호 (SHAP 0.73)
- **15세이용가**: 31.1% 재분류율 — 가장 위험한 등급 경계 구간
- **베팅·현금성 키워드**: cash(65배), slot(47배), bitcoin(42배) — lift 분석 기반 후처리 보정 적용
- **5-Fold CV AUC**: 0.9335 ± 0.005 — 안정적 성능

---

## 라이선스

본 레포지터리는 2026 문화체육관광 인공지능·데이터 활용 공모전 제출 목적으로 작성되었습니다.  
© 2026 GUCC (게임이용자보호센터)

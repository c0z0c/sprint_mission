# 운수종사자 교통사고 위험 예측 - 리더보드 제출 패키지

## 📋 개요
운수종사자 자격검사 데이터를 활용한 교통사고 위험 예측 AI 모델 제출 패키지입니다.

## 📁 파일 구조
```
submission_package/
├── script.py              # 예측 실행 스크립트
├── requirements.txt       # Python 패키지 의존성
├── README_SUBMISSION.md             # 이 문서
├── model/                # 학습된 모델 파일
│   ├── lgbm_A.pkl       # A검사 모델
│   └── lgbm_B.pkl       # B검사 모델
└── output/               # 결과 출력 디렉토리 (자동 생성)
```

## 🔧 환경 설정

### 1. Python 버전
- Python 3.8 이상 권장

### 2. 패키지 설치
```bash
pip install -r requirements.txt
```

## 🚀 실행 방법

### 1. 데이터 준비
다음 파일들을 `./data/` 디렉토리에 위치시켜야 합니다:
- `data/A.csv` - A검사 테스트 데이터
- `data/B.csv` - B검사 테스트 데이터
- `data/sample_submission.csv` (선택사항)

### 2. 스크립트 실행
```bash
python script.py
```

### 3. 결과 확인
실행 완료 후 `./output/submission.csv` 파일이 생성됩니다.

## 📊 모델 정보

### A검사 모델 (`lgbm_A.pkl`)
- **알고리즘**: LightGBM
- **피처 개수**: 91개
- **검증 AUC**: 0.6136

### B검사 모델 (`lgbm_B.pkl`)
- **알고리즘**: LightGBM
- **피처 개수**: 117개
- **검증 AUC**: 0.5255

## ⚠️ 주의사항

1. **데이터 경로**: `./data/` 디렉토리에 A.csv, B.csv 필수
2. **모델 파일**: `./model/` 디렉토리에 lgbm_A.pkl, lgbm_B.pkl 필수
3. **Python 버전**: Python 3.8 이상 사용 권장

---
**버전**: 1.0  
**최종 업데이트**: 2025-11-12

# 운수종사자 적성검사 분석 시스템 - 사용 가이드

## 📦 프로젝트 구조

```
transport_worker_analysis/
│
├── 📄 config.py                    # 프로젝트 설정 (데이터, 모델, 학습 파라미터)
├── 📄 data_loader.py               # 데이터 로드 및 전처리 파이프라인
├── 📄 data_analyzer.py             # 데이터 분석 및 시각화 파이프라인
├── 📄 model_trainer.py             # 모델 학습 및 평가 파이프라인
├── 📄 main.py                      # 메인 실행 스크립트 (전체 파이프라인)
├── 📄 test_pipeline.py             # 간단한 테스트 스크립트
│
├── 📓 analysis_notebook.ipynb      # Jupyter 노트북 (대화형 분석)
│
├── 📋 README.md                    # 프로젝트 문서
├── 📋 USAGE_GUIDE.md              # 이 파일 (사용 가이드)
└── 📋 requirements.txt             # 필요 라이브러리
```

## 🚀 빠른 시작 (3단계)

### 1️⃣ 환경 준비

```bash
# 프로젝트 디렉토리로 이동
cd /mnt/user-data/outputs/transport_worker_analysis

# 필요 라이브러리 설치 (처음 한 번만)
pip install -r requirements.txt --break-system-packages
```

### 2️⃣ 데이터 확인

데이터 파일이 다음 위치에 있는지 확인:
- `/mnt/user-data/uploads/train_A.json`

### 3️⃣ 실행

```bash
# 방법 1: 전체 파이프라인 실행
python main.py

# 방법 2: 빠른 테스트
python test_pipeline.py

# 방법 3: Jupyter 노트북 (권장)
jupyter notebook analysis_notebook.ipynb
```

## 📊 상세 사용법

### A. Python 스크립트 사용

#### 1. 전체 파이프라인 (main.py)

```bash
# 기본 실행 (LightGBM, 전체 분석)
python main.py

# 빠른 실행 (시각화 최소화)
python main.py --quick

# XGBoost 사용
python main.py --model-type xgboost

# Random Forest 사용
python main.py --model-type random_forest

# 단계별 실행
python main.py --step load      # 데이터 로드만
python main.py --step analyze   # 분석만
python main.py --step train     # 학습만
python main.py --step evaluate  # 평가만
```

#### 2. Python 코드로 직접 사용

```python
from config import ProjectConfig
from data_loader import DataLoader
from data_analyzer import DataAnalyzer
from model_trainer import ModelTrainer

# 설정 로드
config = ProjectConfig()

# 데이터 로드 및 전처리
loader = DataLoader(config)
loader.load_json_data("/mnt/user-data/uploads/train_A.json")
loader.preprocess_data()
loader.split_data()

# 데이터 분석
analyzer = DataAnalyzer(config, loader)
analyzer.plot_label_distribution()
analyzer.plot_feature_importance_preliminary()

# 모델 학습
trainer = ModelTrainer(config, loader)
trainer.train()

# 모델 평가
trainer.evaluate()
trainer.plot_confusion_matrix()
trainer.plot_feature_importance()

# 모델 저장
trainer.save_model()
```

### B. Jupyter 노트북 사용 (권장)

```bash
# Jupyter 실행
jupyter notebook analysis_notebook.ipynb
```

노트북 구조:
1. **환경 설정** - 라이브러리 임포트
2. **설정 로드** - Config 확인
3. **데이터 로드** - JSON 데이터 읽기 및 전처리
4. **탐색적 분석** - 다양한 시각화
5. **모델 학습** - LightGBM 학습
6. **모델 평가** - 성능 지표 및 시각화
7. **모델 비교** - 여러 모델 비교 (선택사항)
8. **결과 저장** - 모델 및 데이터 저장

## ⚙️ 설정 변경

### config.py 주요 설정

```python
# 데이터 분할 비율 변경
config.data.train_ratio = 0.8
config.data.val_ratio = 0.1
config.data.test_ratio = 0.1

# 랜덤 시드 변경
config.data.random_seed = 42

# 모델 타입 변경
config.model.model_type = "lightgbm"  # lightgbm, xgboost, random_forest, logistic

# LightGBM 하이퍼파라미터
config.model.lgb_params = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 31,
    "learning_rate": 0.05,
    ...
}

# 학습 파라미터
config.training.num_boost_round = 1000
config.training.early_stopping_rounds = 50
config.training.handle_imbalance = True
```

## 📈 출력 결과

### 1. 시각화 파일 (plots/)

```
/mnt/user-data/outputs/plots/
├── 01_label_distribution.png           # Label 분포
├── 02_feature_distributions.png        # 피처 분포
├── 03_correlation_matrix_*.png         # 상관관계
├── 04_feature_importance_preliminary.png # 사전 중요도
├── 05_test_group_analysis.png          # 검사 그룹 분석
├── 06_boxplots_by_label.png           # 박스플롯
├── confusion_matrix_test.png           # Confusion Matrix
├── roc_curve_test.png                  # ROC Curve
└── feature_importance.png              # 모델 피처 중요도
```

### 2. 모델 파일 (models/)

```
/mnt/user-data/outputs/models/
├── lightgbm_model.pkl                  # 저장된 모델
└── best_model_lgb.pkl                  # 최고 성능 모델
```

### 3. 전처리 데이터

```
/mnt/user-data/outputs/
├── train_processed.csv                 # 전처리된 학습 데이터
├── val_processed.csv                   # 전처리된 검증 데이터
└── test_processed.csv                  # 전처리된 테스트 데이터
```

## 🔍 데이터 이해

### A검사 구조

| 검사 | 측정 항목 | 피처 수 |
|------|-----------|---------|
| A1 | 행동반응 (좌/우, 속도) | 9개 |
| A2 | 행동반응 (가속/감속) | 9개 |
| A3 | 주의력 (Valid/Invalid) | 14개 |
| A4 | Stroop 효과 | 10개 |
| A5 | 변화 감지 | 4개 |
| A6 | 판단능력 | 4개 |
| A7 | 지각성향 | 5개 |
| A8 | 타당도 척도 | 8개 |
| A9 | 충동 억제 | 28개 |

### 주요 피처 예시

- **A1L_fail_rate**: 좌측 방향 실패율
- **A2_accel_fail**: 가속 구간 실패율
- **A3_valid_correct**: Valid 시행 정답률
- **A4_stroop_effect**: Stroop 효과 크기
- **A5_sensitivity**: 변화 감지 민감도
- **A9_nogo_commission**: NoGo 억제 실패율 (충동성)

## 🎯 MVP 목표 및 지표

### 목표
- A검사 데이터로 사고 위험 예측
- Label 0 (Safe) vs Label 1 (Risk)

### 주요 평가 지표

| 지표 | 설명 | 목표 |
|------|------|------|
| **Accuracy** | 전체 정확도 | > 0.85 |
| **Precision** | 위험 예측 정밀도 | > 0.80 |
| **Recall** | 위험 감지율 | > 0.75 |
| **F1-Score** | 조화평균 | > 0.77 |
| **AUC** | ROC 곡선 면적 | > 0.90 |

## 💡 팁 & 트릭

### 1. 빠른 실험

```python
# 학습 속도를 높이려면
config.training.num_boost_round = 100  # 기본 1000
config.training.early_stopping_rounds = 20  # 기본 50
```

### 2. 메모리 절약

```python
# 분석 시 피처 수 제한
analyzer.plot_feature_distributions(max_features=12)  # 기본 20
```

### 3. 클래스 불균형 처리

```python
# 자동 가중치 조정 (기본 활성화)
config.training.handle_imbalance = True

# 수동 가중치 설정
config.training.scale_pos_weight = 10.0  # Label 1 가중치
```

### 4. 여러 모델 빠르게 비교

```python
for model_type in ['lightgbm', 'xgboost', 'random_forest']:
    config.model.model_type = model_type
    trainer = ModelTrainer(config, loader)
    trainer.train()
    results = trainer.evaluate()
    print(f"{model_type}: AUC = {results['metrics']['auc']:.4f}")
```

## 🐛 문제 해결

### 문제 1: "train_A.json not found"

**해결**: 데이터 파일 경로 확인
```bash
ls -l /mnt/user-data/uploads/train_A.json
```

### 문제 2: 메모리 부족

**해결**: 
- 데이터 샘플링 사용
- 피처 수 줄이기
- 시각화 최소화 (`--quick` 옵션)

### 문제 3: 학습이 너무 느림

**해결**:
```python
config.training.num_boost_round = 200  # 줄이기
config.training.early_stopping_rounds = 30  # 줄이기
```

### 문제 4: 모듈을 찾을 수 없음

**해결**:
```bash
pip install -r requirements.txt --break-system-packages
```

## 📞 추가 도움말

### 각 모듈 독립 사용

```python
# 데이터만 로드하고 싶을 때
from data_loader import DataLoader
from config import config

loader = DataLoader(config)
df = loader.load_json_data()

# 분석만 하고 싶을 때
from data_analyzer import DataAnalyzer

analyzer = DataAnalyzer(config, loader)
analyzer.plot_label_distribution()

# 학습만 하고 싶을 때
from model_trainer import ModelTrainer

trainer = ModelTrainer(config, loader)
model = trainer.train()
```

### 저장된 모델 로드

```python
trainer = ModelTrainer(config, loader)
trainer.load_model("best_model_lgb.pkl")
results = trainer.evaluate()
```

## 📚 참고 자료

- **PDF 문서**: `/mnt/user-data/uploads/A검사_신규검사__명세.pdf`
- **MD 문서**: `/mnt/user-data/uploads/운수종사자_A*.md`
- **README**: `README.md`

---

**프로젝트 버전**: 1.0.0  
**최종 업데이트**: 2025-11-09  
**작성자**: 김명환

문의사항이 있으시면 언제든지 연락주세요! 🚀
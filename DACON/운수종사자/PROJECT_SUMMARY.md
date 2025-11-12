# 🎯 운수종사자 적성검사 분석 시스템 - 프로젝트 요약

## ✨ 완성된 시스템 개요

운수종사자의 A검사(신규 자격 검사) 데이터를 활용한 **사고 위험 예측 MVP 시스템**이 완성되었습니다.

---

## 📦 전체 구조

```
transport_worker_analysis/
│
├── 🔧 Core Modules (핵심 모듈)
│   ├── config.py              # 프로젝트 설정 관리
│   ├── data_loader.py         # 데이터 로드 및 전처리
│   ├── data_analyzer.py       # 데이터 분석 및 시각화
│   └── model_trainer.py       # 모델 학습 및 평가
│
├── 🚀 Execution Scripts (실행 스크립트)
│   ├── main.py                # 통합 파이프라인
│   └── test_pipeline.py       # 간단한 테스트
│
├── 📓 Notebooks (노트북)
│   └── analysis_notebook.ipynb # Jupyter 대화형 분석
│
└── 📋 Documentation (문서)
    ├── README.md               # 프로젝트 소개
    ├── USAGE_GUIDE.md         # 상세 사용 가이드
    └── requirements.txt        # 필요 라이브러리
```

---

## 🎨 주요 기능

### 1️⃣ 데이터 파이프라인 (`data_loader.py`)

✅ **기능**:
- JSON 데이터 로드
- 자동 전처리 (결측치, 무한대 값, 범주형 인코딩)
- Label 기준 균등 분할 (Train 8 : Val 1 : Test 1)
- 전처리 데이터 저장

```python
loader = DataLoader(config)
loader.load_json_data()
loader.preprocess_data()
loader.split_data(stratify=True)
```

### 2️⃣ 분석 파이프라인 (`data_analyzer.py`)

✅ **기능**:
- Label 분포 분석
- 피처 분포 시각화
- 상관관계 분석
- 검사 그룹별 비교
- 사전 피처 중요도
- 박스플롯

```python
analyzer = DataAnalyzer(config, loader)
analyzer.generate_full_report()  # 전체 리포트 생성
```

**생성되는 시각화** (9종):
- Label 분포
- 피처 분포 (20개)
- 상관관계 행렬
- 사전 피처 중요도
- 검사 그룹 분석
- 박스플롯
- Confusion Matrix
- ROC Curve
- Feature Importance

### 3️⃣ 학습 파이프라인 (`model_trainer.py`)

✅ **기능**:
- 4가지 모델 지원 (LightGBM, XGBoost, RandomForest, Logistic)
- 자동 클래스 불균형 처리
- Early Stopping
- 모델 저장/로드
- 성능 평가 (Accuracy, Precision, Recall, F1, AUC)
- Feature Importance 추출

```python
trainer = ModelTrainer(config, loader)
trainer.train()
trainer.evaluate()
trainer.plot_confusion_matrix()
trainer.plot_feature_importance()
trainer.save_model()
```

### 4️⃣ 통합 실행 (`main.py`)

✅ **기능**:
- 전체 파이프라인 자동 실행
- 단계별 실행 옵션
- 커맨드라인 인터페이스

```bash
python main.py                    # 전체 실행
python main.py --quick            # 빠른 실행
python main.py --model-type xgboost  # XGBoost 사용
python main.py --step analyze     # 분석만
```

---

## 🔬 데이터 구조

### A검사 (9개)

| 검사 | 내용 | 피처 수 | 주요 지표 |
|------|------|---------|-----------|
| **A1** | 행동반응 (좌/우) | 9 | 실패율, 평균오차, 방향차이 |
| **A2** | 행동반응 (속도) | 9 | 가속/감속 실패율, 오차 |
| **A3** | 주의력 | 14 | Valid/Invalid 정답률, 반응시간 |
| **A4** | Stroop 효과 | 10 | Congruent/Incongruent 정답률 |
| **A5** | 변화 감지 | 4 | 민감도, 변화/무변화 정답률 |
| **A6** | 판단능력 | 4 | 정답 수, 정답률 |
| **A7** | 지각성향 | 5 | 정답 수, 정답률, 수행수준 |
| **A8** | 타당도 척도 | 8 | 왜곡점수, 일관성, 신뢰도 |
| **A9** | 충동 억제 | 28 | Go/NoGo 정답률, 충동성지수 |

**총 피처**: 100+ 개 (파생 피처 포함)

---

## 📊 성능 지표

### 평가 메트릭

| 메트릭 | 설명 | 목표 |
|--------|------|------|
| **Accuracy** | 전체 정확도 | > 85% |
| **Precision** | 위험 예측 정밀도 | > 80% |
| **Recall** | 위험 감지율 | > 75% |
| **F1-Score** | Precision-Recall 조화평균 | > 77% |
| **AUC** | ROC 곡선 아래 면적 | > 90% |

### Confusion Matrix 해석

```
           Predicted
           Safe  Risk
Actual Safe  TN    FP   ← False Positive (안전한데 위험으로 예측)
       Risk  FN    TP   ← True Positive (위험을 정확히 감지)
```

---

## 🚀 실행 방법 (3가지)

### 방법 1: Python 스크립트

```bash
cd /mnt/user-data/outputs/transport_worker_analysis
python main.py
```

### 방법 2: Jupyter 노트북 (권장)

```bash
jupyter notebook analysis_notebook.ipynb
```

### 방법 3: Python 코드

```python
from main import MVPPipeline

pipeline = MVPPipeline()
results = pipeline.run_full_pipeline()
```

---

## 💾 출력 결과

### 1. 시각화 (`/plots/`)

```
01_label_distribution.png
02_feature_distributions.png
03_correlation_matrix_*.png
04_feature_importance_preliminary.png
05_test_group_analysis.png
06_boxplots_by_label.png
confusion_matrix_test.png
roc_curve_test.png
feature_importance.png
```

### 2. 모델 (`/models/`)

```
lightgbm_model.pkl
xgboost_model.pkl  (옵션)
random_forest_model.pkl  (옵션)
```

### 3. 데이터 (루트)

```
train_processed.csv
val_processed.csv
test_processed.csv
```

---

## 🔧 설정 커스터마이징

### config.py 주요 설정

```python
# 데이터 분할 비율
config.data.train_ratio = 0.8
config.data.val_ratio = 0.1
config.data.test_ratio = 0.1

# 모델 선택
config.model.model_type = "lightgbm"  # lightgbm, xgboost, random_forest

# 학습 파라미터
config.training.num_boost_round = 1000
config.training.early_stopping_rounds = 50
config.training.handle_imbalance = True  # 자동 가중치 조정
```

---

## 📈 워크플로우

```
1. 데이터 로드
   └─> train_A.json 읽기
   
2. 전처리
   ├─> 결측치 처리
   ├─> 무한대 값 처리
   └─> 범주형 인코딩
   
3. 데이터 분할
   └─> Train (80%) / Val (10%) / Test (10%)
       (Label 기준 균등 분배)
   
4. 탐색적 분석
   ├─> Label 분포
   ├─> 피처 분포
   ├─> 상관관계
   └─> 검사 그룹 비교
   
5. 모델 학습
   ├─> 클래스 불균형 처리
   ├─> Early Stopping
   └─> Best Iteration 선택
   
6. 모델 평가
   ├─> Test Set 예측
   ├─> 성능 지표 계산
   └─> 시각화 생성
   
7. 결과 저장
   ├─> 모델 파일
   ├─> 시각화 이미지
   └─> 전처리 데이터
```

---

## 🎯 MVP 달성 목표

### ✅ 완료된 기능

- [x] JSON 데이터 로드 파이프라인
- [x] 자동 전처리 (결측치, 이상치)
- [x] Label 기준 균등 분할
- [x] 9가지 분석 시각화
- [x] 4가지 ML 모델 지원
- [x] 자동 클래스 불균형 처리
- [x] 성능 평가 및 시각화
- [x] Feature Importance 분석
- [x] 모델 저장/로드
- [x] Jupyter 노트북
- [x] 상세 문서화

### 📋 향후 개선 방향

- [ ] Hyperparameter Tuning (Optuna)
- [ ] Cross-Validation
- [ ] Ensemble 모델
- [ ] SHAP Value 분석
- [ ] 웹 대시보드 (Streamlit)
- [ ] B검사 데이터 통합
- [ ] 실시간 예측 API

---

## 💡 핵심 특징

1. **클래스 파이프라인 구조**
   - 모듈화된 설계
   - 재사용 가능한 컴포넌트
   - 쉬운 확장성

2. **Label 균등 분배**
   - Stratified Split
   - 클래스 불균형 자동 처리
   - 신뢰성 있는 평가

3. **종합적인 분석**
   - 9가지 시각화
   - 다각도 데이터 탐색
   - 검사별 특성 분석

4. **다중 모델 지원**
   - LightGBM (기본)
   - XGBoost
   - Random Forest
   - Logistic Regression

5. **완벽한 문서화**
   - README
   - USAGE_GUIDE
   - Jupyter Notebook
   - 코드 주석

---

## 📞 사용 시작하기

### 1️⃣ 라이브러리 설치

```bash
cd /mnt/user-data/outputs/transport_worker_analysis
pip install -r requirements.txt --break-system-packages
```

### 2️⃣ 빠른 테스트

```bash
python test_pipeline.py
```

### 3️⃣ 전체 실행

```bash
python main.py
```

또는

```bash
jupyter notebook analysis_notebook.ipynb
```

---

## 📚 문서

| 문서 | 설명 | 위치 |
|------|------|------|
| **README.md** | 프로젝트 개요 | 루트 |
| **USAGE_GUIDE.md** | 상세 사용법 | 루트 |
| **analysis_notebook.ipynb** | 대화형 분석 | 루트 |
| **이 파일** | 프로젝트 요약 | 루트 |

---

## 🏆 결론

**운수종사자 적성검사 분석 시스템 MVP**가 성공적으로 완성되었습니다!

### ✨ 주요 성과

- ✅ 완전한 데이터 파이프라인
- ✅ 종합적인 EDA 도구
- ✅ 다중 ML 모델 지원
- ✅ 자동화된 평가 시스템
- ✅ Jupyter 노트북
- ✅ 상세한 문서화

### 🚀 바로 시작하세요!

```bash
cd /mnt/user-data/outputs/transport_worker_analysis
python main.py
```

---

**프로젝트 완료**  
**버전**: 1.0.0  
**날짜**: 2025-11-09  
**작성자**: 김명환  

💪 **Happy Analyzing!** 🎉
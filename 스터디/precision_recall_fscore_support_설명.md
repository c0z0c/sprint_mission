---
layout: default
title: "precision_recall_fscore_support 상세 설명"
description: "precision_recall_fscore_support 상세 설명"
date: 2025-10-30
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# precision_recall_fscore_support 상세 설명

## 함수 개요

`precision_recall_fscore_support`는 scikit-learn에서 제공하는 분류 모델 평가 메트릭을 한 번에 계산하는 함수입니다.

```python
from sklearn.metrics import precision_recall_fscore_support

precision, recall, f1, support = precision_recall_fscore_support(
    y_true,      # 실제 레이블
    y_pred,      # 예측 레이블
    average='macro'  # 평균 계산 방식
)
```

---

## 반환값 설명

| 반환값 | 수식 | 의미 | 해석 |
|--------|------|------|------|
| **Precision<br>(정밀도)** | `TP / (TP + FP)` | 모델이 Positive라고 **예측한 것** 중<br>실제로 Positive인 비율 | **예측의 정확성**<br>- 높을수록: 거짓 양성(FP) 적음<br>- 예: "이 이메일은 스팸이다" 예측이 얼마나 정확한가? |
| **Recall<br>(재현율)** | `TP / (TP + FN)` | 실제 Positive 중에서<br>모델이 Positive로 **찾아낸** 비율 | **탐지 능력**<br>- 높을수록: 놓친 것(FN) 적음<br>- 예: 실제 스팸 중 얼마나 많이 찾아냈는가? |
| **F1-Score** | `2 × (Precision × Recall) / (Precision + Recall)` | Precision과 Recall의<br>조화평균 | **균형 지표**<br>- 둘 중 하나만 높으면 낮은 값<br>- 둘 다 높아야 높은 값 |
| **Support** | 각 클래스의 실제 샘플 수 | 평가 데이터에서<br>해당 클래스가 몇 개 있는지 | **클래스 분포 확인**<br>- 불균형 정도 파악<br>- 통계적 신뢰도 평가 |

---

## 용어 설명

| 용어 | 영문 | 설명 |
|------|------|------|
| **TP** | True Positive | 실제 Positive를 Positive로 **정확히 예측** |
| **FP** | False Positive | 실제 Negative를 Positive로 **잘못 예측** (거짓 양성) |
| **FN** | False Negative | 실제 Positive를 Negative로 **잘못 예측** (놓침) |
| **TN** | True Negative | 실제 Negative를 Negative로 **정확히 예측** |

---

## Average 옵션 비교

| Average 옵션 | 계산 방식 | 특징 | 적합한 상황 |
|--------------|-----------|------|-------------|
| **`'macro'`** | 각 클래스 메트릭의<br>**산술 평균**<br>`(f1_0 + f1_1 + f1_2) / 3` | - 모든 클래스를 **동등하게** 취급<br>- 소수 클래스도 중요하게 고려 | - **클래스 불균형** 데이터<br>- 모든 클래스가 중요한 경우<br>- **감정 분석** (모든 감정 동등) |
| **`'weighted'`** | 각 클래스 메트릭의<br>**가중 평균**<br>`Σ(f1_i × support_i) / total` | - 샘플 수에 비례한 가중치<br>- 다수 클래스에 더 큰 영향 | - 클래스 분포가 실제를 반영<br>- 다수 클래스가 더 중요한 경우 |
| **`'micro'`** | 전체 TP, FP, FN 합산<br>`total_TP / (total_TP + total_FP)` | - 모든 샘플을 하나로 합산<br>- Accuracy와 유사 | - 다중 레이블 분류<br>- 전체 정확도가 중요한 경우 |
| **`None`** | 반환: 배열<br>`[f1_0, f1_1, f1_2]` | - 각 클래스별 값 개별 반환<br>- 평균 계산 안 함 | - **클래스별 상세 분석** 필요<br>- Confusion Matrix와 함께 사용 |

---

## 실전 예시 (3개 감정 클래스)

### 예시 데이터

```python
from sklearn.metrics import precision_recall_fscore_support

# 감정 레이블: 0=긍정, 1=중립, 2=부정
y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2]  # 실제 레이블
y_pred = [0, 0, 1, 1, 1, 2, 2, 2, 0, 2]  # 모델 예측

# average=None: 클래스별 개별 값
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, average=None
)
```

### 클래스별 결과 (average=None)

| 클래스 | Precision | Recall | F1-Score | Support | 해석 |
|--------|-----------|--------|----------|---------|------|
| **0 (긍정)** | 0.67 | 0.67 | 0.67 | 3 | - 긍정 예측 3개 중 2개 맞음<br>- 실제 긍정 3개 중 2개 찾음 |
| **1 (중립)** | 0.67 | 0.67 | 0.67 | 3 | - 중립 예측 3개 중 2개 맞음<br>- 실제 중립 3개 중 2개 찾음 |
| **2 (부정)** | 0.75 | 0.75 | 0.75 | 4 | - 부정 예측 4개 중 3개 맞음<br>- 실제 부정 4개 중 3개 찾음 |

### 평균 계산 결과

```python
# Macro Average (모든 클래스 동등하게)
macro_precision = (0.67 + 0.67 + 0.75) / 3 = 0.70
macro_recall = (0.67 + 0.67 + 0.75) / 3 = 0.70
macro_f1 = (0.67 + 0.67 + 0.75) / 3 = 0.70

# Weighted Average (샘플 수 비례)
weighted_f1 = (0.67×3 + 0.67×3 + 0.75×4) / 10 = 0.70
```

---

## Precision vs Recall 트레이드오프

### Trade-off 관계

| 상황 | Precision | Recall | 설명 |
|------|-----------|--------|------|
| **보수적 예측**<br>(확실한 것만 예측) | 높음 ↑ | 낮음 ↓ | - 거짓 양성(FP) 줄임<br>- 하지만 많이 놓침(FN 증가) |
| **공격적 예측**<br>(의심되면 예측) | 낮음 ↓ | 높음 ↑ | - 놓치는 것(FN) 줄임<br>- 하지만 오탐(FP 증가) |
| **균형 잡힌 예측** | 중간 | 중간 | - F1-Score가 최대<br>- **일반적으로 선호** |

### 실무 적용 예시

| 도메인 | 중요 지표 | 이유 |
|--------|-----------|------|
| **스팸 필터** | Precision | 정상 메일을 스팸으로 분류(FP)하면 안 됨 |
| **암 진단** | Recall | 암 환자를 놓치면(FN) 치명적 |
| **감정 분석** | **F1-Score** | 모든 감정을 균형있게 정확히 예측해야 함 |
| **사기 탐지** | Recall | 사기를 놓치면(FN) 피해 발생 |

---

## 감정 분석에서의 해석

### 당신의 모델 성능 (예시)

```python
val/f1_macro: 0.882
val/accuracy: 0.912
```

| 메트릭 | 값 | 의미 |
|--------|-----|------|
| **Macro F1** | 0.882 | - 긍정/중립/부정 세 클래스의 평균 F1이 88.2%<br>- 모든 감정을 **골고루 잘** 예측<br>- 클래스 불균형 고려한 **신뢰도 높은 지표** |
| **Accuracy** | 0.912 | - 전체 예측 중 91.2%가 정확<br>- 하지만 클래스 불균형 시 **과대평가** 가능 |

### 클래스별 분석 예시

| 감정 | Precision | Recall | F1 | Support | 분석 |
|------|-----------|--------|-----|---------|------|
| **긍정** | 0.89 | 0.85 | 0.87 | 5000 | 조금 놓치는 경향 (Recall 낮음) |
| **중립** | 0.91 | 0.93 | 0.92 | 8000 | **가장 잘 예측** (다수 클래스) |
| **부정** | 0.84 | 0.87 | 0.85 | 5500 | 약간 오탐 있음 (Precision 낮음) |
| **Macro Avg** | 0.88 | 0.88 | 0.88 | - | 세 감정 평균 성능 |
| **Weighted Avg** | 0.89 | 0.89 | 0.89 | - | 샘플 수 고려 평균 |

---

## 실전 코드 예시

### 1. 기본 사용법

```python
from sklearn.metrics import precision_recall_fscore_support

# 클래스별 상세 분석
precision, recall, f1, support = precision_recall_fscore_support(
    y_true,
    y_pred,
    average=None,  # 클래스별 개별 값
    labels=[0, 1, 2]  # 긍정, 중립, 부정
)

print("클래스별 성능:")
for i, label in enumerate(['긍정', '중립', '부정']):
    print(f"{label}: P={precision[i]:.3f}, R={recall[i]:.3f}, F1={f1[i]:.3f}, N={support[i]}")
```

### 2. Macro 평균 (권장)

```python
# 모든 감정을 동등하게 평가
precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true,
    y_pred,
    average='macro'
)

print(f"Macro F1-Score: {f1_macro:.3f}")  # 클래스 불균형 고려
```

### 3. 시각화용 딕셔너리

```python
from sklearn.metrics import precision_recall_fscore_support, classification_report

# 상세 리포트 (보고서용)
report = classification_report(
    y_true,
    y_pred,
    target_names=['긍정', '중립', '부정'],
    output_dict=True  # 딕셔너리로 반환
)

print(f"긍정 F1: {report['긍정']['f1-score']:.3f}")
print(f"Macro Avg F1: {report['macro avg']['f1-score']:.3f}")
```

---

## 보고서 작성 시 포함할 내용

### 추천 구성

1. **전체 성능 요약**
   - Accuracy, Macro F1, Weighted F1

2. **클래스별 상세 분석**
   - Precision, Recall, F1-Score, Support 표
   - 어떤 감정이 어려운지 분석

3. **Confusion Matrix**
   - 어떤 감정을 어떤 감정으로 헷갈리는지

4. **개선 방향**
   - Precision 낮은 클래스: 오탐 줄이기
   - Recall 낮은 클래스: 놓치는 케이스 줄이기

---

## 참고 자료

- [Scikit-learn 공식 문서](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_recall_fscore_support.html)
- F1-Score는 **조화평균**이므로 Precision과 Recall 중 하나라도 낮으면 크게 하락
- 클래스 불균형 데이터에서는 **Macro F1**이 가장 신뢰할 수 있는 지표

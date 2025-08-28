---
layout: default
title: "Image Segmentation 손실 함수 설계와 최적화"
description: "Image Segmentation 손실 함수 설계와 최적화"
date: 2025-08-27
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# 12. 손실 함수 설계와 최적화

## 목차
1. [Cross-Entropy Loss 심화 분석](#1-cross-entropy-loss-심화-분석)<br/>
   1. 1.1. [수학적 원리와 정보 이론적 해석](#11-수학적-원리와-정보-이론적-해석)<br/>
   1. 1.2. [Weighted Cross-Entropy](#12-weighted-cross-entropy)<br/>
   1. 1.3. [Label Smoothing과 정규화 효과](#13-label-smoothing과-정규화-효과)<br/>

2. [Focal Loss와 Hard Example Mining](#2-focal-loss와-hard-example-mining)<br/>
   2. 2.1. [Focal Loss의 수학적 설계](#21-focal-loss의-수학적-설계)<br/>
   2. 2.2. [파라미터 튜닝과 효과 분석](#22-파라미터-튜닝과-효과-분석)<br/>
   2. 2.3. [다양한 Focal Loss 변형들](#23-다양한-focal-loss-변형들)<br/>

3. [Dice Loss와 Overlap-based Losses](#3-dice-loss와-overlap-based-losses)<br/>
   3. 3.1. [Dice 계수의 미분가능한 구현](#31-dice-계수의-미분가능한-구현)<br/>
   3. 3.2. [IoU Loss와 Jaccard Index](#32-iou-loss와-jaccard-index)<br/>
   3. 3.3. [Tversky Loss와 일반화](#33-tversky-loss와-일반화)<br/>

4. [Boundary-aware Loss Functions](#4-boundary-aware-loss-functions)<br/>
   4. 4.1. [Surface Loss와 거리 변환](#41-surface-loss와-거리-변환)<br/>
   4. 4.2. [Boundary IoU와 경계 정확도](#42-boundary-iou와-경계-정확도)<br/>
   4. 4.3. [Active Contour Loss](#43-active-contour-loss)<br/>

5. [Contrastive Learning과 Metric Learning](#5-contrastive-learning과-metric-learning)<br/>
   5. 5.1. [Contrastive Loss 설계](#51-contrastive-loss-설계)<br/>
   5. 5.2. [Triplet Loss와 Hard Mining](#52-triplet-loss와-hard-mining)<br/>
   5. 5.3. [Center Loss와 Feature Learning](#53-center-loss와-feature-learning)<br/>

---

## 1. Cross-Entropy Loss 심화 분석

### 1.1. 수학적 원리와 정보 이론적 해석

#### 1.1.1. 기본 정의와 유도

**픽셀별 다중 클래스 분류**:
$$\mathcal{L}_{CE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{i,c} \log(\hat{y}_{i,c})$$

여기서:
- $N$: 전체 픽셀 수 ($H \times W$)
- $C$: 클래스 수
- $y_{i,c} \in \{0,1\}$: Ground truth (one-hot)
- $\hat{y}_{i,c} \in [0,1]$: 예측 확률

#### 1.1.2. 정보 이론적 해석

**자기 정보량 (Self-Information)**:
$$I(x) = -\log P(x)$$

확률이 낮은 사건일수록 더 많은 정보를 담고 있다.

**교차 엔트로피의 의미**:
$$H(P,Q) = -\sum_{x} P(x) \log Q(x)$$

실제 분포 $P$와 예측 분포 $Q$ 간의 정보량 차이

#### 1.1.3. KL 발산과의 관계

**KL 발산**:
$$D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)}$$

**관계식**:
$$D_{KL}(P||Q) = H(P,Q) - H(P)$$

Cross-entropy를 최소화하는 것은 KL 발산을 최소화하는 것과 동일 (엔트로피 $H(P)$는 상수)

#### 1.1.4. 그래디언트 분석

**소프트맥스와 결합된 그래디언트**:
$$\frac{\partial \mathcal{L}_{CE}}{\partial z_c} = \hat{y}_c - y_c$$

여기서 $z_c$는 로짓이고, $\hat{y}_c = \text{softmax}(z_c)$이다.

**특성**:
- 선형적 그래디언트 (이차함수 형태)
- 확률과 정답의 차이에 비례
- 잘못된 예측에 큰 그래디언트

### 1.2. Weighted Cross-Entropy

#### 1.2.1. 클래스별 가중치

**가중 Cross-Entropy**:
$$\mathcal{L}_{WCE} = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} w_c \cdot y_{i,c} \log(\hat{y}_{i,c})$$

#### 1.2.2. 가중치 설계 전략

**역빈도 가중치 (Inverse Frequency)**:
$$w_c = \frac{1}{f_c}$$

여기서 $f_c$는 클래스 $c$의 상대 빈도이다.

**균형 가중치 (Balanced Weights)**:
$$w_c = \frac{N_{total}}{C \times N_c}$$

여기서 $N_c$는 클래스 $c$의 샘플 수이다.

**효과적 샘플 수 가중치**:
$$w_c = \frac{1 - \beta}{1 - \beta^{N_c}}$$

여기서 $\beta \in [0,1)$는 하이퍼파라미터이다.

#### 1.2.3. 동적 가중치

**난이도 기반 가중치**:
$$w_{i,c} = \alpha + (1-\alpha) \cdot \left(1 - \hat{y}_{i,c}\right)^\gamma$$

잘못 예측되는 픽셀에 더 높은 가중치 부여

**시간적 가중치 스케줄링**:
$$w_c(t) = w_c^{init} \cdot \exp(-\lambda \cdot t)$$

학습 초기에는 균형을 맞추고, 점진적으로 자연 분포로 전환

### 1.3. Label Smoothing과 정규화 효과

#### 1.3.1. Label Smoothing 정의

**부드러운 라벨**:
$$y'_{i,c} = \begin{cases}
1 - \epsilon + \frac{\epsilon}{C} & \text{if } c = c_{true} \\
\frac{\epsilon}{C} & \text{otherwise}
\end{cases}$$

여기서 $\epsilon \in [0,1]$는 smoothing 파라미터이다.

#### 1.3.2. 정규화 효과

**과신뢰 방지 (Calibration)**:
모델이 지나치게 확신하는 예측을 방지

**일반화 성능 향상**:
$$\text{Regularization Effect} = \frac{\epsilon}{C-1} \sum_{c \neq c_{true}} \log \hat{y}_{i,c}$$

오답 클래스들의 확률도 어느 정도 유지하도록 유도

#### 1.3.3. 최적 $\epsilon$ 선택

**경험적 규칙**:
- 작은 데이터셋: $\epsilon = 0.1$
- 큰 데이터셋: $\epsilon = 0.01$
- 매우 큰 데이터셋: $\epsilon = 0.001$

**적응적 선택**:
$$\epsilon_{adaptive} = \frac{\text{Model Confidence}}{\text{Target Confidence}}$$

## 2. Focal Loss와 Hard Example Mining

### 2.1. Focal Loss의 수학적 설계

#### 2.1.1. 동기와 문제 정의

**클래스 불균형의 문제**:
- 쉬운 배경 픽셀이 손실을 지배
- 어려운 객체 픽셀의 기여도 미미
- 전체적으로 쉬운 예제에 편향된 학습

#### 2.1.2. Focal Loss 공식

**기본 형태**:
$$\mathcal{L}_{FL} = -\alpha_c (1-p_c)^\gamma \log(p_c)$$

여기서:
- $p_c$: 정답 클래스의 예측 확률
- $\alpha_c$: 클래스별 균형 가중치
- $\gamma$: Focusing parameter

#### 2.1.3. $(1-p_c)^\gamma$ 항의 효과

**확률별 가중치 변화**:
- $p_c = 0.9$ (쉬운 예제): $(1-0.9)^2 = 0.01$ (매우 작은 가중치)
- $p_c = 0.6$ (중간 예제): $(1-0.6)^2 = 0.16$ (중간 가중치)
- $p_c = 0.3$ (어려운 예제): $(1-0.3)^2 = 0.49$ (큰 가중치)

**그래디언트 분석**:
$$\frac{\partial \mathcal{L}_{FL}}{\partial p_c} = -\alpha_c \left[\gamma (1-p_c)^{\gamma-1} \log(p_c) + \frac{(1-p_c)^\gamma}{p_c}\right]$$

### 2.2. 파라미터 튜닝과 효과 분석

#### 2.2.1. $\gamma$ 파라미터 효과

**$\gamma = 0$**: 표준 Cross-Entropy와 동일
**$\gamma = 1$**: 선형적 down-weighting
**$\gamma = 2$**: 이차적 down-weighting (가장 일반적)
**$\gamma = 5$**: 매우 강한 focusing

#### 2.2.2. $\alpha$ 파라미터 설정

**클래스별 $\alpha$ 최적화**:
$$\alpha_c = \frac{1}{1 + \beta \cdot f_c}$$

여기서 $\beta$는 밸런싱 강도 파라미터이다.

**적응적 $\alpha$**:
$$\alpha_c(t) = \alpha_c^{init} \cdot \frac{\text{Average CE Loss}_c(t)}{\text{Average CE Loss}(t)}$$

### 2.3. 다양한 Focal Loss 변형들

#### 2.3.1. Class-Balanced Focal Loss

**효과적 샘플 수 고려**:
$$\mathcal{L}_{CB-FL} = -\frac{1-\beta}{1-\beta^{n_c}} (1-p_c)^\gamma \log(p_c)$$

여기서 $n_c$는 클래스 $c$의 샘플 수이다.

#### 2.3.2. Focal Tversky Loss

**Tversky Index와 결합**:
$$\mathcal{L}_{FTL} = (1 - TI)^\gamma$$

$$TI = \frac{TP}{TP + \alpha FN + \beta FP}$$

#### 2.3.3. Adaptive Focal Loss

**동적 $\gamma$ 조정**:
$$\gamma_c(t) = \gamma_0 \cdot \exp\left(-\lambda \cdot \frac{ACC_c(t) - ACC_{min}}{ACC_{max} - ACC_{min}}\right)$$

성능이 낮은 클래스에 더 강한 focusing 적용

## 3. Dice Loss와 Overlap-based Losses

### 3.1. Dice 계수의 미분가능한 구현

#### 3.1.1. Dice 계수 정의

**집합론적 정의**:
$$\text{Dice} = \frac{2|A \cap B|}{|A| + |B|}$$

**확률론적 정의** (세그멘테이션용):
$$\text{Dice}_c = \frac{2\sum_{i} y_{i,c} \hat{y}_{i,c} + \epsilon}{\sum_{i} y_{i,c} + \sum_{i} \hat{y}_{i,c} + \epsilon}$$

#### 3.1.2. Dice Loss

**손실 함수로 변환**:
$$\mathcal{L}_{Dice} = 1 - \text{Dice}_c$$

**멀티클래스 확장**:
$$\mathcal{L}_{mDice} = 1 - \frac{1}{C} \sum_{c=1}^{C} \text{Dice}_c$$

#### 3.1.3. 수치 안정성

**$\epsilon$ 항의 역할**:
- 분모가 0이 되는 경우 방지
- 빈 예측과 빈 GT의 경우 처리
- 일반적으로 $\epsilon = 1$ 또는 $\epsilon = 1e-7$ 사용

**그래디언트 계산**:
$$\frac{\partial \mathcal{L}_{Dice}}{\partial \hat{y}_{i,c}} = -\frac{2(y_{i,c} \sum_j \hat{y}_{j,c} - \hat{y}_{i,c} \sum_j y_{j,c})}{(\sum_j y_{j,c} + \sum_j \hat{y}_{j,c})^2}$$

### 3.2. IoU Loss와 Jaccard Index

#### 3.2.1. IoU (Jaccard Index)

**정의**:
$$\text{IoU}_c = \frac{\sum_{i} y_{i,c} \hat{y}_{i,c} + \epsilon}{\sum_{i} y_{i,c} + \sum_{i} \hat{y}_{i,c} - \sum_{i} y_{i,c} \hat{y}_{i,c} + \epsilon}$$

#### 3.2.2. IoU vs Dice 관계

**수학적 관계**:
$$\text{Dice} = \frac{2 \cdot \text{IoU}}{1 + \text{IoU}}$$

$$\text{Io
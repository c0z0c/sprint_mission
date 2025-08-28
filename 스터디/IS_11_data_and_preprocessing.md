---
layout: default
title: "Image Segmentation 데이터와 전처리"
description: "Image Segmentation 데이터와 전처리"
date: 2025-08-27
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# 11. 데이터와 전처리

## 목차
1. [데이터셋 구조와 Annotation 방법](#1-데이터셋-구조와-annotation-방법)<br/>
   1. 1.1. [픽셀별 라벨링 시스템](#11-픽셀별-라벨링-시스템)<br/>
   1. 1.2. [다중 클래스 인코딩 방법](#12-다중-클래스-인코딩-방법)<br/>
   1. 1.3. [Instance와 Panoptic 어노테이션](#13-instance와-panoptic-어노테이션)<br/>

2. [Data Augmentation 기법](#2-data-augmentation-기법)<br/>
   2. 2.1. [기하학적 변환](#21-기하학적-변환)<br/>
   2. 2.2. [색상 및 조명 변화](#22-색상-및-조명-변화)<br/>
   2. 2.3. [세그멘테이션 특화 증강](#23-세그멘테이션-특화-증강)<br/>

3. [Class Imbalance 문제](#3-class-imbalance-문제)<br/>
   3. 3.1. [클래스 불균형 분석](#31-클래스-불균형-분석)<br/>
   3. 3.2. [샘플링 전략](#32-샘플링-전략)<br/>
   3. 3.3. [손실 함수 기반 해결책](#33-손실-함수-기반-해결책)<br/>

4. [Domain Adaptation과 전이 학습](#4-domain-adaptation과-전이-학습)<br/>
   4. 4.1. [도메인 갭 분석](#41-도메인-갭-분석)<br/>
   4. 4.2. [Adversarial Domain Adaptation](#42-adversarial-domain-adaptation)<br/>
   4. 4.3. [Self-supervised 사전 학습](#43-self-supervised-사전-학습)<br/>

5. [데이터 효율성과 Active Learning](#5-데이터-효율성과-active-learning)<br/>
   5. 5.1. [Uncertainty 기반 샘플 선택](#51-uncertainty-기반-샘플-선택)<br/>
   5. 5.2. [Semi-supervised Learning](#52-semi-supervised-learning)<br/>
   5. 5.3. [Weakly Supervised Segmentation](#53-weakly-supervised-segmentation)<br/>

---

## 1. 데이터셋 구조와 Annotation 방법

### 1.1. 픽셀별 라벨링 시스템

#### 1.1.1. 라벨 맵의 수학적 표현

**픽셀별 클래스 할당**:
$$L: \Omega \rightarrow \{0, 1, 2, ..., C-1\}$$

여기서:
- $\Omega = \{(i,j) : 1 \leq i \leq H, 1 \leq j \leq W\}$: 이미지 도메인
- $C$: 클래스 수 (배경 포함)
- $L(i,j)$: 픽셀 $(i,j)$의 클래스 라벨

**표준 인코딩**:
- 0: 배경 (Background)
- 1, 2, ..., C-1: 전경 클래스
- 255: 무시할 픽셀 (Ignore/Void)

#### 1.1.2. One-hot 인코딩

**밀집 표현**:
$$Y \in \{0, 1\}^{H \times W \times C}$$

$$Y_{i,j,c} = \begin{cases}
1 & \text{if } L(i,j) = c \\
0 & \text{otherwise}
\end{cases}$$

**메모리 효율성**:
- 원본: $H \times W \times 1$ (정수)
- One-hot: $H \times W \times C$ (이진)
- 압축 비율: $1:C$

#### 1.1.3. 경계 처리와 Void 클래스

**Void 픽셀의 필요성**:
- 애매한 경계 지역
- 가림(Occlusion) 영역  
- 라벨링 어려운 구역

**손실 함수에서의 처리**:
$$\mathcal{L} = \frac{1}{N_{valid}} \sum_{(i,j) \in \Omega_{valid}} \ell(y_{i,j}, \hat{y}_{i,j})$$

여기서 $\Omega_{valid} = \{(i,j) : L(i,j) \neq 255\}$이다.

### 1.2. 다중 클래스 인코딩 방법

#### 1.2.1. 색상 기반 인코딩

**RGB 색상 맵핑**:
각 클래스를 고유한 RGB 색상으로 표현

```python
color_map = {
    0: (0, 0, 0),       # 배경 - 검정
    1: (128, 0, 0),     # 사람 - 어두운 빨강
    2: (0, 128, 0),     # 자동차 - 어두운 초록
    3: (128, 128, 0),   # 도로 - 어두운 노랑
    # ...
}
```

**장점**: 시각적 확인 용이
**단점**: RGB 값과 클래스 ID 간 변환 필요

#### 1.2.2. 계층적 라벨링

**상위-하위 클래스 구조**:
$$\text{vehicle} \rightarrow \{\text{car}, \text{truck}, \text{bus}, \text{motorcycle}\}$$

**수학적 표현**:
$$L_{hierarchical}(i,j) = (L_{coarse}(i,j), L_{fine}(i,j))$$

**장점**: 
- 다양한 세밀도에서 평가 가능
- 전이 학습 효과적

#### 1.2.3. 다중 라벨 세그멘테이션

**겹치는 클래스 허용**:
$$L_{multi}(i,j) \subseteq \{1, 2, ..., C\}$$

**예시**: 
- "사람 + 자전거" (사람이 자전거를 타는 경우)
- "도로 + 차선" (차선이 그어진 도로)

**Sigmoid 기반 예측**:
$$P(c \in L(i,j)) = \sigma(f_c(x_{i,j}))$$

### 1.3. Instance와 Panoptic 어노테이션

#### 1.3.1. Instance Annotation

**객체별 고유 ID**:
$$L_{instance}(i,j) = \begin{cases}
0 & \text{if background} \\
\text{instance\_id} & \text{if foreground}
\end{cases}$$

**COCO 스타일 인코딩**:
```json
{
    "id": 12345,
    "category_id": 1,
    "segmentation": [polygon_points],
    "area": 1234.5,
    "bbox": [x, y, width, height]
}
```

#### 1.3.2. Panoptic Annotation

**통합 인코딩**:
$$L_{panoptic}(i,j) = \text{class\_id} \times 1000 + \text{instance\_id}$$

**디코딩**:
```python
class_id = panoptic_id // 1000
instance_id = panoptic_id % 1000
```

**제약 조건**:
- Stuff 클래스: instance_id = 0
- Things 클래스: instance_id > 0
- 픽셀 중복 불허

#### 1.3.3. 어노테이션 품질 관리

**일관성 검증**:
$$\text{Consistency} = \frac{\text{Annotator Agreement}}{\text{Total Pixels}}$$

**Inter-annotator Agreement**:
$$\kappa = \frac{P_o - P_e}{1 - P_e}$$

여기서:
- $P_o$: 관찰된 일치도
- $P_e$: 우연에 의한 일치도

## 2. Data Augmentation 기법

### 2.1. 기하학적 변환

#### 2.1.1. 강체 변환 (Rigid Transform)

**회전 (Rotation)**:
$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}$$

**평행이동 (Translation)**:
$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} x + t_x \\ y + t_y \end{pmatrix}$$

**라벨 변환**: 이미지와 동일한 변환 적용
$$L'(x', y') = L(x, y)$$

#### 2.1.2. 유사 변환 (Similarity Transform)

**스케일링**:
$$\begin{pmatrix} x' \\ y' \end{pmatrix} = s \begin{pmatrix} x \\ y \end{pmatrix}$$

**균등/비균등 스케일링**:
- 균등: $s_x = s_y$ (종횡비 유지)
- 비균등: $s_x \neq s_y$ (왜곡 발생)

**주의사항**: 과도한 비균등 스케일링은 객체 외형을 크게 변화시킴

#### 2.1.3. 아핀 변환 (Affine Transform)

**일반형**:
$$\begin{pmatrix} x' \\ y' \\ 1 \end{pmatrix} = \begin{pmatrix} a & b & t_x \\ c & d & t_y \\ 0 & 0 & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

**전단 변환 (Shear)**:
$$\begin{pmatrix} x' \\ y' \end{pmatrix} = \begin{pmatrix} x + k_x y \\ y + k_y x \end{pmatrix}$$

**보간법**: 변환 후 정수가 아닌 좌표에 대해 보간 적용
- 이미지: Bilinear interpolation
- 라벨: Nearest neighbor (클래스 보존)

#### 2.1.4. 원근 변환 (Perspective Transform)

**호모그래피 행렬**:
$$\begin{pmatrix} wx' \\ wy' \\ w \end{pmatrix} = \begin{pmatrix} h_{11} & h_{12} & h_{13} \\ h_{21} & h_{22} & h_{23} \\ h_{31} & h_{32} & 1 \end{pmatrix} \begin{pmatrix} x \\ y \\ 1 \end{pmatrix}$$

**적용 시 주의사항**:
- 심한 원근 왜곡은 학습에 해로움
- 도로, 건물 등에 자연스러운 효과

### 2.2. 색상 및 조명 변화

#### 2.2.1. 밝기와 대비 조정

**선형 변환**:
$$I'(x,y) = \alpha \cdot I(x,y) + \beta$$

여기서:
- $\alpha > 1$: 대비 증가, $\alpha < 1$: 대비 감소
- $\beta > 0$: 밝기 증가, $\beta < 0$: 밝기 감소

**감마 보정**:
$$I'(x,y) = I(x,y)^\gamma$$
- $\gamma > 1$: 어두운 영역 강조
- $\gamma < 1$: 밝은 영역 강조

#### 2.2.2. 색상 공간 변환

**HSV 공간에서의 조정**:
$$\begin{align}
H' &= H + \delta_H \pmod{360°} \\
S' &= S \times \alpha_S \\
V' &= V \times \alpha_V
\end{align}$$

**장점**: 인간의 색상 인지와 유사한 조정

#### 2.2.3. 노이즈 추가

**가우시안 노이즈**:
$$I'(x,y) = I(x,y) + \mathcal{N}(0, \sigma^2)$$

**Salt-and-Pepper 노이즈**:
$$I'(x,y) = \begin{cases}
0 & \text{with probability } p_{salt} \\
255 & \text{with probability } p_{pepper} \\
I(x,y) & \text{otherwise}
\end{cases}$$

### 2.3. 세그멘테이션 특화 증강

#### 2.3.1. CutMix와 MixUp

**CutMix**:
두 이미지의 일부를 교체하여 새로운 이미지 생성

$$I_{new} = M \odot I_1 + (1-M) \odot I_2$$
$$L_{new} = M \odot L_1 + (1-M) \odot L_2$$

여기서 $M$은 이진 마스크이다.

**MixUp** (픽셀 단위):
$$I_{new} = \lambda I_1 + (1-\lambda) I_2$$
$$L_{new} = \lambda L_1 + (1-\lambda) L_2$$

#### 2.3.2. Copy-Paste

**객체 단위 복사-붙여넣기**:
1. 소스 이미지에서 객체 마스크 추출
2. 타겟 이미지에 객체 붙여넣기
3. 라벨 맵 업데이트

**충돌 처리**:
새로운 객체가 기존 객체와 겹치는 경우의 규칙 정의

#### 2.3.3. Elastic Deformation

**탄성 변형**:
$$\begin{align}
x' &= x + \alpha \cdot G_\sigma * U_x \\
y' &= y + \alpha \cdot G_\sigma * U_y
\end{align}$$

여기서:
- $U_x, U_y$: 균등 분포 랜덤 필드
- $G_\sigma$: 가우시안 필터
- $\alpha$: 변형 강도

**의료 영상에서 효과적**: 장기의 자연스러운 변형 모델링

## 3. Class Imbalance 문제

### 3.1. 클래스 불균형 분석

#### 3.1.1. 불균형 정량화

**클래스 빈도**:
$$f_c = \frac{N_c}{\sum_{i=1}^{C} N_i}$$

여기서 $N_c$는 클래스 $c$의 픽셀 수이다.

**불균형 비율 (Imbalance Ratio)**:
$$IR = \frac{\max_c f_c}{\min_c f_c}$$

**지니 계수 (Gini Coefficient)**:
$$G = 1 - \sum_{c=1}^{C} f_c^2$$

#### 3.1.2. 세그멘테이션에서의 특수성

**배경 편향**: 대부분 데이터셋에서 배경이 60-80% 차지
**Long-tail 분포**: 소수 클래스가 전체의 1% 미만

**공간적 불균형**:
같은 클래스라도 이미지마다 크기가 다름
$$\text{Size Variance} = \frac{\text{Var}(\text{object sizes})}{\text{Mean}(\text{object sizes})}$$

#### 3.1.3. 성능에 미치는 영향

**Accuracy Paradox**:
모든 픽셀을 배경으로 예측해도 높은 정확도 달성 가능

**F1-Score 분석**:
$$F1_c = \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

불균형 클래스에서 F1 점수 급격히 하락

### 3.2. 샘플링 전략

#### 3.2.1. 이미지 레벨 샘플링

**클래스 균형 샘플링**:
각 클래스가 포함된 이미지를 균등하게 선택

$$P(\text{select image } i) \propto \frac{1}{\sum_{c \in \text{classes}(i)} f_c}$$

**효과**: 희귀 클래스 노출 빈도 증가
**한계**: 배경 클래스는 모든 이미지에 존재

#### 3.2.2. 픽셀 레벨 샘플링

**가중 무작위 샘플링**:
손실 계산 시 클래스별 가중치 적용

$$w_c = \frac{1}{f_c} \text{ or } w_c = \frac{1}{\sqrt{f_c}}$$

**역빈도 가중치**:
$$w_c = \frac{\sum_{i=1}^{C} N_i}{C \cdot N_c}$$

#### 3.2.3. 하드 예제 마이닝

**어려운 픽셀 우선 선택**:
$$\text{Difficulty}(i,j) = -\log P(\hat{y}_{i,j} = y_{i,j})$$

상위 $k$%의 어려운 픽셀만 손실에 기여

### 3.3. 손실 함수 기반 해결책

#### 3.3.1. Focal Loss

**기본 아이디어**: 쉬운 예제의 기여도 감소

$$\mathcal{L}_{FL} = -\alpha_c (1-p_c)^\gamma \log p_c$$

여기서:
- $p_c$: 정답 클래스의 예측 확률
- $\gamma$: Focusing parameter (일반적으로 2)
- $\alpha_c$: 클래스별 가중치

**효과 분석**:
- $p_c$ 높음 (쉬운 예제): $(1-p_c)^\gamma$ 작음 → 작은 손실
- $p_c$ 낮음 (어려운 예제): $(1-p_c)^\gamma$ 큼 → 큰 손실

#### 3.3.2. Dice Loss

**Dice 계수 기반**:
$$\mathcal{L}_{Dice} = 1 - \frac{2\sum_{i,j} y_{i,j} \hat{y}_{i,j} + \epsilon}{\sum_{i,j} y_{i,j} + \sum_{i,j} \hat{y}_{i,j} + \epsilon}$$

**장점**:
- 클래스 불균형에 상대적으로 robust
- F1-score와 직접적 연관

#### 3.3.3. Class-Balanced Loss

**효과적 샘플 수 고려**:
$$E_n = \frac{1 - \beta^n}{1 - \beta}$$

여기서:
- $n$: 샘플 수
- $\beta \in [0,1)$: 하이퍼파라미터

**재가중 인수**:
$$w_c = \frac{1}{E_{n_c}}$$

## 4. Domain Adaptation과 전이 학습

### 4.1. 도메인 갭 분석

#### 4.1.1. 도메인 정의

**소스 도메인**: $\mathcal{D}_S = \{(x_i^S, y_i^S)\}_{i=1}^{N_S}$
**타겟 도메인**: $\mathcal{D}_T = \{x_j^T\}_{j=1}^{N_T}$ (라벨 없음)

**분포 차이**:
$$\mathcal{H}\Delta\mathcal{H} \text{ distance} = 2 \sup_{h \in \mathcal{H}} |P_S(h) - P_T(h)|$$

#### 4.1.2. 도메인 갭의 원인

**시각적 차이**:
- 조명 조건 (일광, 야간, 실내)
- 날씨 조건 (맑음, 비, 눈)
- 카메라 특성 (해상도, 색감)

**의미적 차이**:
- 클래스 분포 변화
- 새로운 클래스 등장
- 레이아웃과 구성의 차이

#### 4.1.3. MMD (Maximum Mean Discrepancy)

**특징 분포 간 거리**:
$$\text{MMD}^2(\mathcal{D}_S, \mathcal{D}_T) = ||\frac{1}{n_S}\sum_{i=1}^{n_S} \phi(x_i^S) - \frac{1}{n_T}\sum_{j=1}^{n_T} \phi(x_j^T)||^2$$

여기서 $\phi(\cdot)$는 특징 변환 함수이다.

### 4.2. Adversarial Domain Adaptation

#### 4.2.1. 도메인 적대적 학습

**전체 목적 함수**:
$$\mathcal{L}_{total} = \mathcal{L}_{seg}(S) + \lambda_{adv} \mathcal{L}_{adv}(S,T) + \lambda_{ent} \mathcal{L}_{ent}(T)$$

**세그멘테이션 손실** (소스만):
$$\mathcal{L}_{seg} = \frac{1}{N_S} \sum_{i=1}^{N_S} \ell(y_i^S, G(x_i^S))$$

#### 4.2.2. 적대적 손실

**판별기 손실**:
$$\mathcal{L}_D = -\mathbb{E}_{x^S}[\log D(G(x^S))] - \mathbb{E}_{x^T}[\log(1-D(G(x^T)))]$$

**생성기 손실**:
$$\mathcal{L}_G = -\mathbb{E}_{x^T}[\log D(G(x^T))]$$

#### 4.2.3. 엔트로피 최소화

**타겟 도메인 예측의 신뢰도 향상**:
$\mathcal{L}_{ent} = \mathbb{E}_{x^T}\left[-\sum_{c=1}^{C} p_c(x^T) \log p_c(x^T)\right]$

**효과**: 
- 낮은 엔트로피 → 확신 있는 예측
- 결정 경계 명확화

### 4.3. Self-supervised 사전 학습

#### 4.3.1. Pretext Tasks

**회전 예측 (Rotation Prediction)**:
$\mathcal{L}_{rot} = -\sum_{r \in \{0°, 90°, 180°, 270°\}} \mathbb{I}[r = r_{true}] \log p(r|x)$

**직소 퍼즐 (Jigsaw Puzzle)**:
이미지를 9개 패치로 나누어 순서 맞추기

**색상화 (Colorization)**:
흑백 이미지로부터 색상 예측

#### 4.3.2. Contrastive Learning

**InfoNCE Loss**:
$\mathcal{L} = -\log \frac{\exp(q \cdot k_+ / \tau)}{\sum_{i=0}^{K} \exp(q \cdot k_i / \tau)}$

여기서:
- $q$: query (anchor)
- $k_+$: positive key
- $k_i$: negative keys
- $\tau$: temperature parameter

#### 4.3.3. Masked Image Modeling

**MAE (Masked Autoencoder) 스타일**:
1. 입력 패치의 75% 마스킹
2. 가시적 패치만으로 인코딩
3. 마스킹된 패치 재구성

**손실 함수**:
$\mathcal{L}_{MAE} = \frac{1}{|\mathcal{M}|} \sum_{i \in \mathcal{M}} ||x_i - \hat{x}_i||^2$

여기서 $\mathcal{M}$은 마스킹된 패치 집합이다.

## 5. 데이터 효율성과 Active Learning

### 5.1. Uncertainty 기반 샘플 선택

#### 5.1.1. 불확실성 측정

**예측 엔트로피**:
$H(y|x) = -\sum_{c=1}^{C} p(y=c|x) \log p(y=c|x)$

**상호 정보량**:
$I(y;\theta|x) = H(y|x) - \mathbb{E}_{\theta}[H(y|x,\theta)]$

베이지안 신경망에서 파라미터 불확실성 고려

#### 5.1.2. MC Dropout

**추론 시 드롭아웃 유지**:
$\hat{y}_t = f(x; \theta_t), \quad \theta_t \sim \text{Dropout}(\theta)$

**앙상블 평균**:
$\bar{y} = \frac{1}{T} \sum_{t=1}^{T} \hat{y}_t$

**불확실성 추정**:
$\text{Uncertainty} = \frac{1}{T} \sum_{t=1}^{T} ||\hat{y}_t - \bar{y}||^2$

#### 5.1.3. 픽셀별 불확실성 집계

**이미지 레벨 점수**:
$\text{Score}_{img} = \frac{1}{HW} \sum_{i,j} H(y_{i,j}|x)$

**가중 불확실성**:
클래스 중요도에 따른 가중치 적용
$\text{Score}_{weighted} = \sum_{c=1}^{C} w_c \cdot \text{Uncertainty}_c$

### 5.2. Semi-supervised Learning

#### 5.2.1. 일관성 정규화

**동일 입력의 다른 증강에 대한 일관성**:
$\mathcal{L}_{consistency} = ||f(T_1(x)) - f(T_2(x))||^2$

여기서 $T_1, T_2$는 서로 다른 증강 함수이다.

#### 5.2.2. Pseudo Labeling

**신뢰도 기반 라벨 생성**:
$\tilde{y}_{i,j} = \begin{cases}
\arg\max_c p(y_{i,j}=c|x) & \text{if } \max_c p(y_{i,j}=c|x) > \tau \\
\text{unlabeled} & \text{otherwise}
\end{cases}$

**점진적 임계값 감소**:
$\tau_t = \tau_0 \cdot \exp(-\alpha \cdot t)$

#### 5.2.3. Mean Teacher

**지수 이동 평균 교사 모델**:
$\theta_{teacher}^{(t)} = \alpha \theta_{teacher}^{(t-1)} + (1-\alpha) \theta_{student}^{(t)}$

**일관성 손실**:
$\mathcal{L}_{MT} = MSE(f_{teacher}(x + \xi), f_{student}(x + \xi'))$

### 5.3. Weakly Supervised Segmentation

#### 5.3.1. 이미지 레벨 라벨

**클래스 존재 여부만 알려진 경우**:
$\text{Labels} = \{c_1, c_2, ..., c_k\} \subset \{1, 2, ..., C\}$

**Global Average Pooling 기반**:
$p(c \in \text{image}) = \text{GAP}(f_c(\text{feature map}))$

#### 5.3.2. Class Activation Maps

**가중 활성화 맵**:
$\text{CAM}_c = \sum_{k} w_k^c f_k$

여기서 $w_k^c$는 클래스 $c$에 대한 특징 $k$의 가중치이다.

**Grad-CAM**:
그래디언트 기반 가중치 계산
$w_k^c = \frac{1}{Z} \sum_i \sum_j \frac{\partial y^c}{\partial A_{i,j}^k}$

#### 5.3.3. 바운딩 박스 감독

**박스 내부 픽셀 제약**:
$\mathcal{L}_{box} = \frac{1}{|B|} \sum_{(i,j) \in B} \ell(y_{i,j}, \hat{y}_{i,j})$

여기서 $B$는 바운딩 박스 영역이다.

**배경 제약**:
박스 외부는 배경으로 간주
$\forall (i,j) \notin \bigcup_k B_k : \hat{y}_{i,j} = 0$

---

## 데이터 효율성의 실전적 고려사항

### 라벨링 비용 최적화

**Active Learning 주기**:
- 초기 소량 데이터로 모델 학습
- 불확실성 기반 샘플 선택
- 선택된 샘플만 라벨링
- 모델 재학습 및 반복

**비용-성능 분석**:
$\text{Efficiency} = \frac{\text{Performance Gain}}{\text{Labeling Cost}}$

### 도메인별 전략

**의료 영상**: 
- 전문가 라벨링 비용 매우 높음
- Weak supervision 활용도 높음

**자율주행**:
- 안전성 중요 → 고품질 라벨 필수
- 시뮬레이션 데이터 활용

**일반 객체**:
- 크라우드소싱 활용 가능
- 품질 관리 중요

### 품질 보장

**라벨 노이즈 처리**:
- 불일치 검출 알고리즘
- 다수결 투표 시스템
- 전문가 검증 단계

**일관성 유지**:
- 라벨링 가이드라인 표준화
- 정기적 품질 점검
- 도구와 인터페이스 표준화

---

## 용어 목록

- **Active Learning**: 액티브 러닝 - 능동 학습
- **Adversarial Domain Adaptation**: 어드버새리얼 도메인 어댑테이션 - 적대적 도메인 적응
- **Affine Transform**: 어파인 트랜스폼 - 아핀 변환
- **Annotation**: 어노테이션 - 주석, 라벨링
- **Class Activation Maps (CAM)**: 클래스 액티베이션 맵스 - 클래스 활성화 맵
- **Class Imbalance**: 클래스 임밸런스 - 클래스 불균형
- **Consistency Regularization**: 컨시스턴시 레귤라라이제이션 - 일관성 정규화
- **Contrastive Learning**: 컨트래스티브 러닝 - 대조 학습
- **Copy-Paste**: 카피 페이스트 - 복사-붙여넣기
- **CutMix**: 컷믹스 - 자르기 혼합
- **Data Augmentation**: 데이터 오그멘테이션 - 데이터 증강
- **Domain Adaptation**: 도메인 어댑테이션 - 도메인 적응
- **Domain Gap**: 도메인 갭 - 도메인 차이
- **Elastic Deformation**: 일래스틱 디포메이션 - 탄성 변형
- **Entropy Minimization**: 엔트로피 미니마이제이션 - 엔트로피 최소화
- **Focal Loss**: 포컬 로스 - 초점 손실
- **Gamma Correction**: 감마 코렉션 - 감마 보정
- **Gini Coefficient**: 지니 코에피션트 - 지니 계수
- **Grad-CAM**: 그래드 캠 - 그래디언트 기반 클래스 활성화 맵
- **Hard Example Mining**: 하드 이그잼플 마이닝 - 어려운 예제 채굴
- **Hierarchical Labeling**: 하이어라키컬 레이블링 - 계층적 라벨링
- **HSV Color Space**: 에이치에스브이 칼라 스페이스 - 색조-채도-명도 색공간
- **InfoNCE Loss**: 인포엔씨이 로스 - 정보 잡음 대조 추정 손실
- **Instance Annotation**: 인스턴스 어노테이션 - 개체 주석
- **Inter-annotator Agreement**: 인터 어노테이터 어그리먼트 - 주석자 간 일치도
- **Jigsaw Puzzle**: 직쏘 퍼즐 - 조각 맞추기
- **Maximum Mean Discrepancy (MMD)**: 맥시멈 민 디스크레판시 - 최대 평균 불일치
- **MC Dropout**: 엠씨 드롭아웃 - 몬테카를로 드롭아웃
- **Mean Teacher**: 민 티처 - 평균 교사
- **MixUp**: 믹스업 - 혼합
- **Mutual Information**: 뮤추얼 인포메이션 - 상호 정보량
- **One-hot Encoding**: 원핫 인코딩 - 원핫 부호화
- **Panoptic Annotation**: 파놉틱 어노테이션 - 전체 주석
- **Perspective Transform**: 퍼스펙티브 트랜스폼 - 원근 변환
- **Pretext Tasks**: 프리텍스트 태스크스 - 사전 과제
- **Pseudo Labeling**: 슈도 레이블링 - 의사 라벨링
- **Rigid Transform**: 리지드 트랜스폼 - 강체 변환
- **Salt-and-Pepper Noise**: 솔트 앤 페퍼 노이즈 - 소금후추 잡음
- **Semi-supervised Learning**: 세미 수퍼바이즈드 러닝 - 준지도 학습
- **Shear Transform**: 시어 트랜스폼 - 전단 변환
- **Similarity Transform**: 시밀래리티 트랜스폼 - 유사 변환
- **Uncertainty Estimation**: 언서틴티 에스티메이션 - 불확실성 추정
- **Void Class**: 보이드 클래스 - 빈 클래스, 무시 클래스
- **Weakly Supervised**: 위클리 수퍼바이즈드 - 약지도, 약한 감독
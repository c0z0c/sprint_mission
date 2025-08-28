---
layout: default
title: "Image Segmentation 수학적 기초 이론"
description: "Image Segmentation 수학적 기초 이론"
date: 2025-08-27
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# 2. 수학적 기초 이론

## 목차
1. [컨볼루션 신경망의 수학적 기초](#1-컨볼루션-신경망의-수학적-기초)<br/>
   1. 1.1. [컨볼루션 연산의 정의](#11-컨볼루션-연산의-정의)<br/>
   1. 1.2. [이산 컨볼루션과 상관관계](#12-이산-컨볼루션과-상관관계)<br/>
   1. 1.3. [컨볼루션의 수학적 성질](#13-컨볼루션의-수학적-성질)<br/>

2. [신경망 최적화 이론](#2-신경망-최적화-이론)<br/>
   2. 2.1. [경사 하강법의 수학적 원리](#21-경사-하강법의-수학적-원리)<br/>
   2. 2.2. [역전파 알고리즘](#22-역전파-알고리즘)<br/>
   2. 2.3. [활성화 함수와 미분](#23-활성화-함수와-미분)<br/>

3. [확률론과 정보 이론](#3-확률론과-정보-이론)<br/>
   3. 3.1. [베이즈 정리와 사후 추론](#31-베이즈-정리와-사후-추론)<br/>
   3. 3.2. [엔트로피와 정보량](#32-엔트로피와-정보량)<br/>
   3. 3.3. [KL 발산과 교차 엔트로피](#33-kl-발산과-교차-엔트로피)<br/>

4. [선형대수학 응용](#4-선형대수학-응용)<br/>
   4. 4.1. [텐서 연산과 차원 변환](#41-텐서-연산과-차원-변환)<br/>
   4. 4.2. [행렬 분해와 특이값 분해](#42-행렬-분해와-특이값-분해)<br/>
   4. 4.3. [정규화와 조건수](#43-정규화와-조건수)<br/>

5. [최적화 이론 심화](#5-최적화-이론-심화)<br/>
   5. 5.1. [볼록 최적화와 비볼록 최적화](#51-볼록-최적화와-비볼록-최적화)<br/>
   5. 5.2. [적응적 학습률 알고리즘](#52-적응적-학습률-알고리즘)<br/>
   5. 5.3. [정규화 기법의 수학적 해석](#53-정규화-기법의-수학적-해석)<br/>

---

## 1. 컨볼루션 신경망의 수학적 기초

### 1.1. 컨볼루션 연산의 정의

#### 1.1.1. 연속 함수에서의 컨볼루션

두 함수 $f$와 $g$의 컨볼루션은 다음과 같이 정의된다:

$$(f * g)(t) = \int_{-\infty}^{\infty} f(\tau) g(t - \tau) d\tau$$

이는 함수 $g$를 뒤집고 이동시켜가며 $f$와의 유사도를 측정하는 연산이다.

#### 1.1.2. 이산 2D 컨볼루션

이미지 처리에서 사용하는 이산 2차원 컨볼루션:

$$(I * K)(i,j) = \sum_{m=-\infty}^{\infty} \sum_{n=-\infty}^{\infty} I(m,n) K(i-m, j-n)$$

실제 구현에서는 커널 $K$가 유한한 크기 $k \times k$를 가지므로:

$$(I * K)(i,j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) K(m,n)$$

### 1.2. 이산 컨볼루션과 상관관계

#### 1.2.1. Cross-Correlation vs Convolution

딥러닝에서 실제로 사용하는 연산은 상관관계(cross-correlation)이다:

$$\text{Correlation}: (I \star K)(i,j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) K(m,n)$$

$$\text{Convolution}: (I * K)(i,j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) K(k-1-m, k-1-n)$$

차이점은 커널을 뒤집는지 여부이다. 딥러닝에서는 학습을 통해 커널을 얻으므로 이 차이는 중요하지 않다.

#### 1.2.2. 패딩과 스트라이드

**패딩(Padding)**: 입력 주변에 값을 추가하여 출력 크기 조절

- **Valid Padding**: 패딩 없음
- **Same Padding**: 출력 크기가 입력과 동일하도록 패딩

패딩이 $p$, 스트라이드가 $s$인 경우 출력 크기:

$$H_{out} = \left\lfloor \frac{H_{in} + 2p - k}{s} \right\rfloor + 1$$
$$W_{out} = \left\lfloor \frac{W_{in} + 2p - k}{s} \right\rfloor + 1$$

### 1.3. 컨볼루션의 수학적 성질

#### 1.3.1. 교환 법칙 (Commutative Property)

$$f * g = g * f$$

#### 1.3.2. 결합 법칙 (Associative Property)

$$(f * g) * h = f * (g * h)$$

#### 1.3.3. 분배 법칙 (Distributive Property)

$$f * (g + h) = f * g + f * h$$

#### 1.3.4. 푸리에 변환과의 관계

컨볼루션 정리: 공간 도메인에서의 컨볼루션은 주파수 도메인에서의 곱셈과 동일하다.

$$\mathcal{F}\{f * g\} = \mathcal{F}\{f\} \cdot \mathcal{F}\{g\}$$

이 성질을 이용하여 대용량 커널에 대한 효율적 계산이 가능하다.

## 2. 신경망 최적화 이론

### 2.1. 경사 하강법의 수학적 원리

#### 2.1.1. 기본 경사 하강법

목적 함수 $J(\theta)$를 최소화하기 위해 그래디언트의 반대 방향으로 파라미터를 업데이트:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)$$

여기서 $\alpha$는 학습률(learning rate)이다.

#### 2.1.2. 학습률의 영향

학습률이 너무 크면 발산하고, 너무 작으면 수렴이 느리다. 최적 학습률은 헤시안 행렬(Hessian matrix)의 고유값과 관련:

$$\alpha_{optimal} \approx \frac{2}{\lambda_{max} + \lambda_{min}}$$

여기서 $\lambda_{max}, \lambda_{min}$은 헤시안의 최대, 최소 고유값이다.

#### 2.1.3. 모멘텀과 가속화

모멘텀을 추가한 업데이트:

$$v_{t+1} = \beta v_t + \alpha \nabla_\theta J(\theta_t)$$
$$\theta_{t+1} = \theta_t - v_{t+1}$$

여기서 $\beta$는 모멘텀 계수이다. 이는 지수 가중 이동 평균(exponentially weighted moving average)을 사용하여 그래디언트의 분산을 줄인다.

### 2.2. 역전파 알고리즘

#### 2.2.1. 연쇄 법칙 (Chain Rule)

다층 신경망에서 파라미터 $\theta$에 대한 손실의 그래디언트는 연쇄 법칙을 통해 계산된다:

$\frac{\partial L}{\partial \theta} = \frac{\partial L}{\partial y} \frac{\partial y}{\partial z} \frac{\partial z}{\partial \theta}$

여기서 $z$는 선형 변환, $y$는 활성화 함수의 출력이다.

#### 2.2.2. 컨볼루션 층에서의 역전파

입력 $X$, 가중치 $W$, 출력 $Y$에 대해:

$Y_{i,j} = \sum_{m,n} X_{i+m,j+n} W_{m,n}$

가중치에 대한 그래디언트:

$\frac{\partial L}{\partial W_{m,n}} = \sum_{i,j} \frac{\partial L}{\partial Y_{i,j}} X_{i+m,j+n}$

입력에 대한 그래디언트:

$\frac{\partial L}{\partial X_{i,j}} = \sum_{m,n} \frac{\partial L}{\partial Y_{i-m,j-n}} W_{m,n}$

#### 2.2.3. 그래디언트 소실과 폭발

깊은 네트워크에서 그래디언트가 층을 거슬러 올라가며 전파될 때:

$\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial y_L} \prod_{l=2}^{L} \frac{\partial y_l}{\partial y_{l-1}}$

각 층에서의 야코비안(Jacobian) 행렬의 고유값이 1보다 작으면 그래디언트 소실, 1보다 크면 그래디언트 폭발이 발생한다.

### 2.3. 활성화 함수와 미분

#### 2.3.1. 주요 활성화 함수들

**ReLU (Rectified Linear Unit)**:
$f(x) = \max(0, x)$
$f'(x) = \begin{cases} 1 & \text{if } x > 0 \\ 0 & \text{if } x \leq 0 \end{cases}$

**Sigmoid**:
$f(x) = \frac{1}{1 + e^{-x}}$
$f'(x) = f(x)(1 - f(x))$

**Tanh**:
$f(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
$f'(x) = 1 - f(x)^2$

#### 2.3.2. 활성화 함수의 선택 기준

1. **그래디언트 소실 방지**: ReLU는 양수 구간에서 그래디언트가 1로 일정
2. **계산 효율성**: ReLU는 단순한 max 연산
3. **희소성(Sparsity)**: ReLU는 음수 입력에 대해 0 출력

## 3. 확률론과 정보 이론

### 3.1. 베이즈 정리와 사후 추론

#### 3.1.1. 베이즈 정리

$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$

세그멘테이션에서 픽셀 $(i,j)$가 클래스 $c$에 속할 확률:

$P(y_{i,j} = c | x_{i,j}) = \frac{P(x_{i,j} | y_{i,j} = c) \cdot P(y_{i,j} = c)}{P(x_{i,j})}$

#### 3.1.2. 최대 사후 확률 (Maximum A Posteriori, MAP)

MAP 추정은 사후 확률을 최대화하는 파라미터를 찾는다:

$\hat{\theta}_{MAP} = \arg\max_\theta P(\theta | D) = \arg\max_\theta P(D | \theta) P(\theta)$

로그를 취하면:
$\hat{\theta}_{MAP} = \arg\max_\theta [\log P(D | \theta) + \log P(\theta)]$

#### 3.1.3. 베이지안 세그멘테이션

불확실성을 고려한 세그멘테이션을 위해 파라미터의 분포를 모델링:

$P(y | x, D) = \int P(y | x, \theta) P(\theta | D) d\theta$

실제로는 몬테카를로 근사를 사용:
$P(y | x, D) \approx \frac{1}{S} \sum_{s=1}^{S} P(y | x, \theta_s)$

### 3.2. 엔트로피와 정보량

#### 3.2.1. 정보량 (Information Content)

사건 $x$의 정보량: $I(x) = -\log_2 P(x)$

확률이 낮은 사건일수록 더 많은 정보를 담고 있다.

#### 3.2.2. 엔트로피 (Entropy)

확률 분포 $P$의 엔트로피:
$H(P) = -\sum_{x} P(x) \log P(x)$

엔트로피는 분포의 불확실성을 측정한다. 균등 분포일 때 최대, 결정적 분포일 때 최소이다.

#### 3.2.3. 조건부 엔트로피

$H(Y|X) = \sum_{x} P(x) H(Y|X=x) = -\sum_{x,y} P(x,y) \log P(y|x)$

#### 3.2.4. 상호 정보량 (Mutual Information)

$I(X;Y) = H(X) - H(X|Y) = H(Y) - H(Y|X)$

두 변수 간의 의존성을 측정한다.

### 3.3. KL 발산과 교차 엔트로피

#### 3.3.1. KL 발산 (Kullback-Leibler Divergence)

두 확률 분포 $P$와 $Q$ 사이의 차이를 측정:

$D_{KL}(P||Q) = \sum_{x} P(x) \log \frac{P(x)}{Q(x)} = E_P\left[\log \frac{P(x)}{Q(x)}\right]$

KL 발산의 성질:
- $D_{KL}(P||Q) \geq 0$ (비음성)
- $D_{KL}(P||Q) = 0$ iff $P = Q$ (동일성)
- $D_{KL}(P||Q) \neq D_{KL}(Q||P)$ (비대칭)

#### 3.3.2. 교차 엔트로피

$H(P,Q) = -\sum_{x} P(x) \log Q(x)$

KL 발산과의 관계:
$D_{KL}(P||Q) = H(P,Q) - H(P)$

딥러닝에서는 $H(P)$가 상수이므로 교차 엔트로피를 최소화하는 것이 KL 발산을 최소화하는 것과 동일하다.

## 4. 선형대수학 응용

### 4.1. 텐서 연산과 차원 변환

#### 4.1.1. 텐서의 정의

$n$차원 텐서는 $n$개의 인덱스로 접근하는 다차원 배열이다:
- 0차원: 스칼라
- 1차원: 벡터  
- 2차원: 행렬
- 3차원 이상: 텐서

#### 4.1.2. 텐서 곱 (Tensor Product)

두 텐서 $A \in \mathbb{R}^{m \times n}$와 $B \in \mathbb{R}^{p \times q}$의 크로네커 곱:

$(A \otimes B)_{(i,k),(j,l)} = A_{i,j} B_{k,l}$

결과는 $mp \times nq$ 크기의 행렬이다.

#### 4.1.3. 차원 변환과 재구성

**Reshape**: 텐서의 차원을 변경하되 전체 원소 수는 유지
$\text{reshape}: \mathbb{R}^{a \times b \times c} \rightarrow \mathbb{R}^{d \times e \times f}, \quad abc = def$

**Transpose**: 차원의 순서를 바꿈
$A^T_{i,j} = A_{j,i}$

**Broadcasting**: 서로 다른 크기의 텐서 간 연산을 위한 차원 확장

### 4.2. 행렬 분해와 특이값 분해

#### 4.2.1. 특이값 분해 (Singular Value Decomposition, SVD)

행렬 $A \in \mathbb{R}^{m \times n}$에 대해:

$A = U \Sigma V^T$

여기서:
- $U \in \mathbb{R}^{m \times m}$: 왼쪽 특이벡터들의 직교 행렬
- $\Sigma \in \mathbb{R}^{m \times n}$: 특이값들의 대각 행렬
- $V \in \mathbb{R}^{n \times n}$: 오른쪽 특이벡터들의 직교 행렬

#### 4.2.2. 주성분 분석 (Principal Component Analysis, PCA)

데이터 행렬 $X$의 공분산 행렬 $C = \frac{1}{n-1}X^TX$에 대해 고유값 분해:

$C = Q \Lambda Q^T$

주성분은 $Q$의 열벡터들이며, 고유값 $\lambda_i$는 각 주성분의 분산을 나타낸다.

#### 4.2.3. 저차원 근사

SVD를 이용한 행렬의 저차원 근사:

$A \approx A_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T$

여기서 $k$는 유지할 특이값의 개수이다.

### 4.3. 정규화와 조건수

#### 4.3.1. 벡터와 행렬의 노름

**벡터 노름**:
- L1 노름: $||x||_1 = \sum_{i} |x_i|$
- L2 노름: $||x||_2 = \sqrt{\sum_{i} x_i^2}$
- L∞ 노름: $||x||_\infty = \max_i |x_i|$

**행렬 노름**:
- 프로베니우스 노름: $||A||_F = \sqrt{\sum_{i,j} A_{i,j}^2}$
- 스펙트럴 노름: $||A||_2 = \sigma_{max}(A)$ (최대 특이값)

#### 4.3.2. 조건수 (Condition Number)

행렬 $A$의 조건수:
$\kappa(A) = \frac{\sigma_{max}}{\sigma_{min}}$

조건수가 클수록 수치적으로 불안정하다. 잘 조건화된 행렬은 $\kappa(A) \approx 1$이다.

#### 4.3.3. 정규화 기법

**L1 정규화 (Lasso)**:
$\mathcal{L}_{L1} = \mathcal{L}_{original} + \lambda \sum_{i} |\theta_i|$

희소성(sparsity)을 유도한다.

**L2 정규화 (Ridge)**:
$\mathcal{L}_{L2} = \mathcal{L}_{original} + \lambda \sum_{i} \theta_i^2$

가중치의 크기를 제한한다.

## 5. 최적화 이론 심화

### 5.1. 볼록 최적화와 비볼록 최적화

#### 5.1.1. 볼록 함수의 정의

함수 $f$가 볼록하다는 것은 임의의 $x_1, x_2$와 $\lambda \in [0,1]$에 대해:

$f(\lambda x_1 + (1-\lambda) x_2) \leq \lambda f(x_1) + (1-\lambda) f(x_2)$

#### 5.1.2. 볼록 최적화의 성질

- 지역 최적해가 전역 최적해
- 유일한 전역 최적해 존재 (강볼록 함수의 경우)
- 그래디언트 하강법으로 전역 최적해 보장

#### 5.1.3. 신경망에서의 비볼록성

신경망의 손실 함수는 일반적으로 비볼록이다:
- 여러 지역 최적해 존재
- 안장점(saddle point) 문제
- 초기화에 민감

### 5.2. 적응적 학습률 알고리즘

#### 5.2.1. AdaGrad

각 파라미터에 대해 과거 그래디언트의 제곱합을 누적:

$G_t = G_{t-1} + g_t^2$
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{G_t + \epsilon}} g_t$

장점: 희소한 특징에 대해 큰 학습률 적용
단점: 학습률이 계속 감소하여 조기 정지

#### 5.2.2. RMSprop

지수 가중 이동 평균을 사용하여 AdaGrad의 단점 보완:

$v_t = \beta v_{t-1} + (1-\beta) g_t^2$
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{v_t + \epsilon}} g_t$

#### 5.2.3. Adam

모멘텀과 적응적 학습률을 결합:

$m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t$
$v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2$

편향 보정:
$\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}$

업데이트:
$\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$

### 5.3. 정규화 기법의 수학적 해석

#### 5.3.1. Batch Normalization

각 미니배치에서 평균과 분산을 정규화:

$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$

여기서:
- $\mu_B = \frac{1}{m} \sum_{i=1}^{m} x_i$
- $\sigma_B^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_B)^2$

학습 가능한 파라미터 $\gamma, \beta$를 추가:
$y_i = \gamma \hat{x}_i + \beta$

#### 5.3.2. Dropout의 수학적 모델

훈련 시 각 뉴런을 확률 $p$로 제거:

$\tilde{h}_i = \begin{cases}
0 & \text{with probability } p \\
\frac{h_i}{1-p} & \text{with probability } 1-p
\end{cases}$

이는 앙상블 효과를 통해 과적합을 방지한다.

#### 5.3.3. Early Stopping

검증 손실이 증가하기 시작하면 훈련을 중단:

$\text{Stop when } \mathcal{L}_{val}(t) > \mathcal{L}_{val}(t-patience)$

이는 암시적 정규화 역할을 한다.

---

## 용어 목록

- **AdaGrad**: 아다그라드 - 적응적 그래디언트 알고리즘
- **Associative**: 어소시에이티브 - 결합법칙을 만족하는 성질
- **Backpropagation**: 백프로퍼게이션 - 역전파 알고리즘
- **Commutative**: 커뮤테이티브 - 교환법칙을 만족하는 성질
- **Condition Number**: 컨디션 넘버 - 조건수, 행렬의 수치적 안정성 지표
- **Cross-Correlation**: 크로스 코릴레이션 - 상관관계, 두 신호의 유사성 측정
- **Distributive**: 디스트리뷰티브 - 분배법칙을 만족하는 성질
- **Eigenvalue**: 아이겐밸류 - 고유값
- **Exponentially Weighted Moving Average**: 익스포넨셜리 웨이티드 무빙 애버리지 - 지수 가중 이동 평균
- **Fourier Transform**: 푸리에 트랜스폼 - 푸리에 변환
- **Gradient Descent**: 그래디언트 디센트 - 경사 하강법
- **Hessian Matrix**: 헤시안 매트릭스 - 2차 편미분으로 구성된 행렬
- **Jacobian Matrix**: 야코비안 매트릭스 - 1차 편미분으로 구성된 행렬
- **Kronecker Product**: 크로네커 프로덕트 - 크로네커 곱
- **Learning Rate**: 러닝 레이트 - 학습률
- **Maximum A Posteriori (MAP)**: 맥시멈 어 포스테리오리 - 최대 사후 확률
- **Monte Carlo**: 몬테 카를로 - 확률적 근사 방법
- **Mutual Information**: 뮤추얼 인포메이션 - 상호 정보량
- **Principal Component Analysis (PCA)**: 프린시펄 컴포넌트 어날리시스 - 주성분 분석
- **RMSprop**: 알엠에스프롭 - 루트 평균 제곱 전파법
- **Saddle Point**: 새들 포인트 - 안장점
- **Singular Value Decomposition (SVD)**: 싱귤러 밸류 디컴포지션 - 특이값 분해
- **Spectral Norm**: 스펙트럴 놈 - 스펙트럴 노름
- **Tensor Product**: 텐서 프로덕트 - 텐서 곱
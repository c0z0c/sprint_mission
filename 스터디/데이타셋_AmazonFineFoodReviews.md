---
layout: default
title: "데이타셋 Amazon Fine Food Reviews"
description: "데이타셋 Amazon Fine Food Reviews"
date: 2025-10-01
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# Amazon Fine Food Reviews 데이터셋 분석

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)<br/>
   - 1.1. [데이터셋 정의](#11-데이터셋-정의)<br/>
   - 1.2. [프로젝트 목표](#12-프로젝트-목표)<br/>
   - 1.3. [실용적 응용](#13-실용적-응용)<br/>
2. [데이터셋 개요](#2-데이터셋-개요)<br/>
   - 2.1. [데이터 출처 및 배경](#21-데이터-출처-및-배경)<br/>
   - 2.2. [데이터셋 규모](#22-데이터셋-규모)<br/>
3. [데이터 필드 분석](#3-데이터-필드-분석)<br/>
   - 3.1. [필드 구조](#31-필드-구조)<br/>
   - 3.2. [각 필드의 특성](#32-각-필드의-특성)<br/>
4. [텍스트 데이터 특성 분석](#4-텍스트-데이터-특성-분석)<br/>
   - 4.1. [Summary 필드 분석](#41-summary-필드-분석)<br/>
   - 4.2. [Text 필드 분석](#42-text-필드-분석)<br/>
   - 4.3. [Summary-Text 관계 분석](#43-summary-text-관계-분석)<br/>
5. [Seq2Seq 및 Transformer 모델링 관점](#5-seq2seq-및-transformer-모델링-관점)<br/>
   - 5.1. [입력-출력 구조 설계](#51-입력-출력-구조-설계)<br/>
   - 5.2. [데이터 전처리 요구사항](#52-데이터-전처리-요구사항)<br/>
   - 5.3. [시퀀스 길이 분석](#53-시퀀스-길이-분석)<br/>
6. [어텐션 메커니즘 활용 전략](#6-어텐션-메커니즘-활용-전략)<br/>
   - 6.1. [핵심 정보 추출 패턴](#61-핵심-정보-추출-패턴)<br/>
   - 6.2. [길이 불균형 처리](#62-길이-불균형-처리)<br/>
   - 6.3. [어텐션 시각화](#63-어텐션-시각화)<br/>
7. [모델 아키텍처 권장사항](#7-모델-아키텍처-권장사항)<br/>
   - 7.1. [Seq2Seq with Attention](#71-seq2seq-with-attention)<br/>
   - 7.2. [Transformer 기반 모델](#72-transformer-기반-모델)<br/>
   - 7.3. [사전학습 모델 활용](#73-사전학습-모델-활용)<br/>
   - 7.4. [손실 함수 및 평가 메트릭](#74-손실-함수-및-평가-메트릭)<br/>
8. [데이터 품질 이슈 및 대응](#8-데이터-품질-이슈-및-대응)<br/>
   - 8.1. [중복 데이터](#81-중복-데이터)<br/>
   - 8.2. [노이즈 및 불완전한 데이터](#82-노이즈-및-불완전한-데이터)<br/>
   - 8.3. [데이터 필터링 전략](#83-데이터-필터링-전략)<br/>
9. [실험 설계 제안](#9-실험-설계-제안)<br/>
   - 9.1. [학습/검증/테스트 분할 전략](#91-학습검증테스트-분할-전략)<br/>
   - 9.2. [하이퍼파라미터 튜닝 포인트](#92-하이퍼파라미터-튜닝-포인트)<br/>
10. [PyTorch 구현 개요](#10-pytorch-구현-개요)<br/>
    - 10.1. [필수 라이브러리](#101-필수-라이브러리)<br/>
    - 10.2. [사전학습 모델 사용](#102-사전학습-모델-사용)<br/>
11. [용어 목록](#11-용어-목록)<br/>

---

## 1. 프로젝트 개요

### 1.1. 데이터셋 정의

Amazon Fine Food Reviews는 1999년 10월부터 2012년 10월까지 아마존 플랫폼에서 수집된 **식품 카테고리 제품 리뷰** 데이터이다.<br/>
사용자들이 구매한 커피, 스낵, 애완동물 사료 등 다양한 식품에 대한 상세한 사용 후기를 담고 있다.

**데이터 출처:**
- Kaggle: https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews
- Stanford SNAP (Network Analysis Project)

### 1.2. 프로젝트 목표

**핵심 목표: 자동 리뷰 요약 생성 (Automatic Review Summarization)**

긴 리뷰 본문(Text, 평균 100단어)을 입력받아 핵심 내용을 담은 짧은 요약(Summary, 평균 5단어)을 자동으로 생성하는 Seq2Seq 및 Transformer 모델을 개발한다.

**구체적 학습 목표:**
- Seq2Seq 아키텍처(architecture)의 인코더-디코더(encoder-decoder) 구조 이해
- 어텐션 메커니즘(attention mechanism)의 작동 원리 및 효과 분석
- Transformer 모델의 셀프 어텐션(self-attention) 학습
- 사전학습 모델(BART, T5, Pegasus) 파인튜닝(fine-tuning) 경험

**연구 질문:**
1. 어텐션 메커니즘이 긴 리뷰에서 핵심 정보를 얼마나 효과적으로 포착하는가?
2. Transformer 아키텍처가 RNN 기반 Seq2Seq 대비 어떤 성능 개선을 보이는가?
3. 사전학습 모델의 파인튜닝이 처음부터 학습 대비 얼마나 효율적인가?

### 1.3. 실용적 응용

**전자상거래 플랫폼:**
- 제품 페이지 상단에 리뷰 요약 자동 표시
- 수백 개의 리뷰를 한눈에 파악 가능한 "한줄평" 생성
- 구매 전환율(conversion rate) 향상

**고객 피드백 분석:**
- 대량의 고객 의견을 신속하게 요약
- 제품 개선점 도출
- 마케팅 인사이트(insight) 추출

**확장 가능성:**
- 영화/책 리뷰 요약
- 뉴스 기사 헤드라인 생성
- 문서 요약 시스템

---

## 2. 데이터셋 개요

### 2.1. 데이터 출처 및 배경

Amazon Fine Food Reviews 데이터셋은 **스탠포드 네트워크 애널리시스 프로젝트(Stanford Network Analysis Project, SNAP)**에서 제공하며, 자연어 처리, 특히 텍스트 서머라이제이션(text summarization) 연구에 널리 활용된다.

**핵심 특징:**
- 실제 사용자가 작성한 자연어 텍스트
- Summary(요약)와 Text(본문)의 쌍으로 구성된 병렬 코퍼스(parallel corpus)
- 평점, 시간, 사용자 정보 등 풍부한 메타데이터(metadata) 포함
- 13년간의 시계열(time-series) 데이터

### 2.2. 데이터셋 규모

| 항목 | 수량 |
|------|------|
| 총 리뷰 수 | 568,454개 |
| 사용자 수 | 256,059명 |
| 제품 수 | 74,258개 |
| 시간 범위 | 1999년 10월 ~ 2012년 10월 (약 13년) |
| 평균 리뷰 길이 | 약 75-100 단어 |
| 평균 요약 길이 | 약 4-5 단어 |

---

## 3. 데이터 필드 분석

### 3.1. 필드 구조

데이터셋은 10개의 컬럼(column)으로 구성되어 있다:

| 필드명 | 데이터 타입 | 설명 |
|--------|------------|------|
| Id | Integer | 리뷰 고유 식별자 |
| ProductId | String | 제품 식별자 (예: B001E4KFG0) |
| UserId | String | 사용자 식별자 (예: A3SGXH7AUHU8GW) |
| ProfileName | String | 사용자 프로필 이름 |
| HelpfulnessNumerator | Integer | 도움됨 투표 수 |
| HelpfulnessDenominator | Integer | 전체 투표 수 |
| Score | Integer | 평점 (1~5) |
| Time | Unix Timestamp | 리뷰 작성 시간 |
| **Summary** | **String** | **리뷰 요약 (제목) - 모델 출력** |
| **Text** | **String** | **리뷰 본문 - 모델 입력** |

### 3.2. 각 필드의 특성

#### 3.2.1. Summary (타겟 시퀀스)

**역할:** Seq2Seq 모델의 디코더 출력 시퀀스

**통계적 특성:**
- 평균 길이: 4-5 단어
- 최대 길이: 대부분 20 단어 이하
- 압축률: 본문 대비 약 1:20

**언어적 특징:**
- 간결하고 핵심적인 표현
- 감정 표현어(sentiment words) 포함 빈도 높음
- 패턴화된 표현 많음

**예시:**
```
"Delicious and healthy"
"Not what I expected"
"Best coffee ever!"
"Good value for money"
```

**모델링 관점에서 중요한 점:**
- 짧은 길이로 그래디언트 베니싱(gradient vanishing) 문제 완화
- 평가 메트릭으로 BLEU, ROUGE, METEOR 사용 가능
- 빔 서치(beam search) 깊이를 작게 설정 가능 (3-5)

#### 3.2.2. Text (소스 시퀀스)

**역할:** Seq2Seq 모델의 인코더 입력 시퀀스

**통계적 특성:**
- 평균 길이: 80-100 단어
- 최대 길이: 수백 단어 (일부 수천 단어)
- 롱 테일 디스트리뷰션(long-tail distribution)

**언어적 특징:**
- 상세한 경험 서술
- 제품 특성, 맛, 품질, 배송 등 다양한 주제
- 비정형 문법, 오타, 구어체 표현 포함

**일반적 구조:**
1. 도입부: 제품 이름, 구매 동기
2. 본론: 상세 경험, 특징 설명
3. 결론: 전체 평가, 추천 여부

**모델링 관점에서 중요한 점:**
- 긴 시퀀스로 인한 롱 디펜던시(long-term dependency) 문제 발생 가능
- 어텐션 메커니즘 필수적
- 포지셔널 인코딩(positional encoding) 중요
- 트렁케이션(truncation) 전략 필요 (보통 150-200 토큰으로 제한)

#### 3.2.3. Score (감정 레이블)

**분포:**
- 1점: 5%
- 2점: 4%
- 3점: 8%
- 4점: 15%
- 5점: 68% (편향성 존재)

**활용 방안:**
- 멀티태스크 러닝(multi-task learning): 요약 생성 + 평점 예측
- 컨디셔널 제너레이션(conditional generation): 평점 조건부 요약
- 데이터 필터링: 특정 평점 범위 선택하여 불균형 완화

#### 3.2.4. Helpfulness (품질 지표)

**계산 방법:**
```
Helpfulness Ratio = HelpfulnessNumerator / HelpfulnessDenominator
```

**활용 방안:**
- 품질 필터링: ratio > 0.7인 리뷰만 선택
- 웨이트 샘플링(weighted sampling): 고품질 리뷰에 높은 가중치
- 학습 데이터 선별에 유용

**주의사항:**
- Denominator가 0인 경우 다수 존재 (투표 미참여 리뷰)
- 최근 리뷰일수록 투표 수 적음 (시간 편향, temporal bias)

#### 3.2.5. Time (시간 정보)

**활용 방안:**
- 템포럴 스플릿(temporal split): 시간 기준 학습/테스트 분할
- 언어 패턴의 시간적 변화 분석
- 최신 데이터로 테스트하여 일반화 성능 확인

---

## 4. 텍스트 데이터 특성 분석

### 4.1. Summary 필드 분석

#### 4.1.1. 길이 분포

**통계:**
- 평균: 4.2 단어
- 중앙값: 4 단어
- 표준편차: 2.5 단어
- 최빈값: 3-5 단어

**모델링 설정:**
- 디코더 최대 길이: 20-25 토큰
- 대부분의 요약이 이 범위 내 포함

#### 4.1.2. 어휘 다양성

**보카뷸러리(vocabulary) 크기:**
- 전체 유니크 토큰: 50,000-60,000개
- 상위 10,000개 단어로 95% 커버
- 상위 20,000개 단어로 98% 커버

**주요 특징:**
- 집프 법칙(Zipf's law) 따름
- 고빈도 단어: "great", "good", "love", "perfect", "excellent"

**권장사항:**
- 보카뷸러리 크기: 30,000-50,000 토큰
- 서브워드 토크나이제이션(subword tokenization) 사용
  - BPE (Byte Pair Encoding)
  - WordPiece
  - SentencePiece

#### 4.1.3. 감정 표현 패턴

**긍정 키워드:**
- excellent, perfect, delicious, amazing, wonderful, fantastic

**부정 키워드:**
- disappointed, terrible, awful, waste, avoid, poor

**중립 키워드:**
- okay, decent, average, fine

**분석 결과:**
- 대부분의 Summary에 명확한 감정 표현 포함
- Score와 Summary의 높은 상관관계
- 감정 강도가 Score와 일치

### 4.2. Text 필드 분석

#### 4.2.1. 길이 분포

**통계:**
- 평균: 75 단어
- 중앙값: 60 단어
- 표준편차: 65 단어
- 분포: 대부분 50-150 단어, 5% 미만이 200 단어 초과

**모델링 설정:**
- 인코더 최대 길이: 150-200 토큰 권장
- 트렁케이션 전략: Head-only (앞부분 우선) 추천

#### 4.2.2. 노이즈 특성

**주요 노이즈 유형:**

1. **오타 및 맞춤법 오류**
   - "recieve" → "receive"
   - "definately" → "definitely"

2. **비정형 문법**
   - 구어체 표현
   - 불완전한 문장

3. **특수 문자 및 HTML 엔티티**
   - `&amp;`, `&lt;`, `&gt;` 등

4. **중복 표현**
   - "very very very good"

5. **무관한 정보**
   - 배송 정보, 가격 정보 (시간 의존적)

**전처리 필요사항:**
- HTML 언이스케이핑(unescaping)
- 케이스 노멀라이제이션(case normalization)
- 펑크추에이션(punctuation) 정규화
- 공백 정규화

### 4.3. Summary-Text 관계 분석

#### 4.3.1. 압축률 (Compression Ratio)

**평균 압축률:**
- Summary 길이 / Text 길이 ≈ 1:18 ~ 1:20
- 본문의 약 5%만 요약으로 표현

**의미:**
- 추상적 요약(abstractive summarization) 필요
- 추출적 요약(extractive summarization)만으로는 부족
- 모델이 정보를 압축하고 재구성하는 능력 필요

#### 4.3.2. 의미적 관계

**관계 유형:**

1. **직접적 표현**
   - Text: "This product is excellent"
   - Summary: "Excellent product"

2. **추상화**
   - Text: "The flavor is rich, smooth, perfect balance"
   - Summary: "Perfect taste"

3. **감정 집약**
   - Text: 긴 긍정적 서술
   - Summary: 강한 긍정 표현 하나로 압축

**중요한 점:**
- 시맨틱 유사도(semantic similarity) 높음
- 파라프레이징(paraphrasing) 능력 필요
- 센티먼트 프리저베이션(sentiment preservation) 중요

#### 4.3.3. 정보 선택 패턴

**Summary에 주로 포함:**
- 전체적 평가
- 감정적 반응
- 핵심 특징
- 추천 여부

**Summary에 생략:**
- 상세 설명
- 배경 스토리
- 부수적 정보

**모델 학습 시 중요한 점:**
- 어텐션 메커니즘이 이러한 선택 패턴을 학습해야 함
- 중요도 가중치 필요
- 컨텐트 셀렉션(content selection) 능력 필요

---

## 5. Seq2Seq 및 Transformer 모델링 관점

### 5.1. 입력-출력 구조 설계

#### 5.1.1. 기본 구조

**입력 시퀀스 (Source):**
```
X = [x₁, x₂, ..., xₙ]  where n ≈ 100-150 토큰
```

**출력 시퀀스 (Target):**
```
Y = [y₁, y₂, ..., yₘ]  where m ≈ 5-20 토큰
```

**확률 모델:**
```
P(Y|X) = ∏ᵢ₌₁ᵐ P(yᵢ | y₁, ..., yᵢ₋₁, X)
```

이는 자기회귀적(autoregressive) 생성 방식으로, 각 토큰이 이전 토큰들과 입력 시퀀스에 조건부로 생성된다.

#### 5.1.2. 토크나이제이션 전략

**권장 방법:**

1. **BPE (Byte Pair Encoding)**
   - 서브워드 단위 분할
   - OOV (Out-of-Vocabulary) 문제 완화
   - 보카뷸러리 크기: 32,000-50,000

2. **SentencePiece**
   - 언어 독립적
   - 전처리 없이 raw text 처리
   - 사전학습 모델과 호환성 좋음

**예시:**
```
Original: "unbelievably delicious"
BPE 토큰: ["un", "believ", "ably", "delicious"]
```

**장점:**
- 희귀 단어 효과적 처리
- 형태소 정보 보존
- 모델 크기 최적화

### 5.2. 데이터 전처리 요구사항

#### 5.2.1. 필수 전처리 단계

**1단계: 기본 정제**
- HTML 엔티티 변환: `&amp;` → `&`
- 케이스 정규화: lowercase 변환
- 공백 정규화: 다중 공백 → 단일 공백

**2단계: 품질 필터링**
- 결측값 제거
- 길이 제약: 10 < Text길이 < 500, 1 < Summary길이 < 50
- 중복 제거

**3단계: 데이터 검증**
- Text와 Summary 의미 관련성 확인
- 언어 감지 (영어 리뷰만 선택)

#### 5.2.2. 선택적 전처리

**하지 않는 것이 좋은 전처리:**
- 스템밍/레마타이제이션: 현대 모델은 서브워드로 충분
- 스톱워드 제거: 문맥 정보 손실
- 과도한 스펠링 교정: 의미 변경 위험

### 5.3. 시퀀스 길이 분석

#### 5.3.1. 입력 시퀀스 길이

**권장 설정:**
- 최대 길이: 200 토큰
- 근거: 95% 이상의 Text가 200 토큰 이내

**트렁케이션 전략:**

| 전략 | 설명 | 권장도 |
|------|------|--------|
| Head-only | 앞 150-200 토큰만 유지 | ⭐⭐⭐ 강력 추천 |
| Tail-only | 뒤 200 토큰만 유지 | ⭐ 비추천 |
| Head-Tail | 앞 100 + 뒤 50 토큰 | ⭐⭐ 선택적 사용 |

**Head-only 추천 이유:**
- 리뷰는 앞부분에 핵심 내용
- 뒷부분은 반복 또는 부가 정보 많음

#### 5.3.2. 출력 시퀀스 길이

**권장 설정:**
- 최대 길이: 25 토큰
- 근거: 99% 이상의 Summary가 25 토큰 이내

**길이 제어 메커니즘:**

1. **EOS 토큰 (End-of-Sequence)**
   - 모델이 자연스럽게 종료 학습
   - 가장 자연스러운 방법

2. **렝스 펄털티 (Length Penalty)**
   ```
   Score = log P(Y) / |Y|^α
   ```
   - α = 0.6-0.8 권장
   - 짧은 요약 방지

---

## 6. 어텐션 메커니즘 활용 전략

### 6.1. 핵심 정보 추출 패턴

#### 6.1.1. 어텐션의 역할

**Seq2Seq with Attention:**
```
컨텍스트 벡터 cᵢ = Σⱼ αᵢⱼ hⱼ
```

여기서:
- cᵢ: 디코더 타임스텝 i의 컨텍스트 벡터
- αᵢⱼ: 어텐션 가중치 (출력 i가 입력 j에 집중하는 정도)
- hⱼ: 인코더 히든 스테이트 j

**의미:**
- 디코더가 각 단어 생성 시 입력의 관련 부분에 집중
- 긴 시퀀스에서도 정보 손실 최소화
- 롱 디펜던시 문제 해결

#### 6.1.2. 예상되는 어텐션 패턴

**1. 감정어에 높은 가중치**
- Text: "This coffee is **absolutely fantastic**"
- Summary: "Fantastic coffee" 생성 시 "fantastic"에 집중

**2. 전체 평가 문장 집중**
- Text: "Overall, I'm very satisfied"
- Summary 생성 시 해당 부분에 높은 어텐션

**3. 위치 편향**
- 도입부와 결론에 높은 가중치
- 중간 부분(상세 설명)은 낮은 가중치

#### 6.1.3. 어텐션 메커니즘 종류

**1. Bahdanau Attention (Additive)**
```
score(hⱼ, sᵢ) = vᵀ tanh(W₁hⱼ + W₂sᵢ)
```
- 표현력 강함
- 계산 비용 높음

**2. Luong Attention (Multiplicative)**
```
score(hⱼ, sᵢ) = sᵢᵀ W hⱼ
```
- 계산 효율적
- 성능 유사

**3. Self-Attention (Transformer)**
```
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```
- 병렬화 가능
- 최신 표준
- 사전학습 모델 활용 가능

### 6.2. 길이 불균형 처리

#### 6.2.1. 문제점

**입력-출력 길이 비율:**
- Text: 100 토큰
- Summary: 5 토큰
- 비율: 약 20:1

**발생하는 문제:**
- 그래디언트 불균형
- 인포메이션 보틀넥(information bottleneck)
- 얼라인먼트(alignment) 어려움

#### 6.2.2. 해결 전략

**1. 어텐션 메커니즘**
- 컨텍스트 벡터를 매 타임스텝마다 동적 계산
- 정보 손실 최소화

**2. 카피 메커니즘 (Copy Mechanism)**
- 입력에서 직접 토큰 복사
- 고유명사, 제품명 정확 전달
- Pointer-Generator Network 활용

**3. 커버리지 메커니즘 (Coverage Mechanism)**
- 이미 어텐션된 부분 추적
- 반복 생성 방지
- 커버리지 벡터 유지

### 6.3. 어텐션 시각화

#### 6.3.1. 시각화 목적

**분석 가능한 정보:**
- 모델이 어느 입력 토큰에 집중하는지
- 어텐션 패턴의 일관성
- 모델의 해석 가능성(interpretability)

#### 6.3.2. 시각화 방법

**히트맵(Heatmap) 방식:**
- X축: 입력 토큰 (Text)
- Y축: 출력 토큰 (Summary)
- 색상 강도: 어텐션 가중치 (0~1)

**예시 해석:**
```
입력: "This coffee is delicious and affordable"
출력: "Great coffee"

어텐션 패턴:
        This  coffee  is  delicious  and  affordable
Great   0.1   0.3    0.0    0.4      0.1     0.1
coffee  0.0   0.8    0.0    0.1      0.0     0.1

→ "Great" 생성 시 "delicious"에 높은 어텐션 (0.4)
→ "coffee" 생성 시 "coffee"에 압도적 어텐션 (0.8) - 복사 패턴
```

**멀티 헤드 어텐션의 경우:**
- 각 헤드별로 다른 패턴 포착
- 일부 헤드는 위치 정보, 일부는 의미 정보 담당

---

## 7. 모델 아키텍처 권장사항

### 7.1. Seq2Seq with Attention

#### 7.1.1. 기본 구조

**인코더 (Encoder):**
- RNN 종류: LSTM 또는 GRU
- 양방향(Bidirectional) 구조
- 레이어 수: 2-4 layers
- 히든 사이즈: 256-512 dim

**디코더 (Decoder):**
- RNN 종류: LSTM 또는 GRU
- 단방향(Unidirectional) 구조
- 레이어 수: 2-4 layers
- 히든 사이즈: 512-1024 dim

**어텐션:**
- Luong Attention 또는 Bahdanau Attention

#### 7.1.2. 장단점

**장점:**
- 구현 상대적으로 단순
- 직관적 이해 가능
- 작은 데이터셋에서도 학습 가능

**단점:**
- 순차 처리로 병렬화 제약
- 긴 시퀀스에서 성능 저하 가능
- 학습 속도 상대적으로 느림

#### 7.1.3. 권장 하이퍼파라미터

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| Encoder Hidden Size | 512 | 인코더 히든 스테이트 크기 |
| Decoder Hidden Size | 512 | 디코더 히든 스테이트 크기 |
| Embedding Dimension | 300 | 단어 임베딩 차원 |
| Num Layers | 2 | 인코더/디코더 레이어 수 |
| Dropout | 0.3-0.5 | 드롭아웃 비율 |
| Batch Size | 32-64 | 배치 크기 |
| Learning Rate | 0.001 | 초기 학습률 |
| Optimizer | Adam | 옵티마이저 |

### 7.2. Transformer 기반 모델

#### 7.2.1. 바닐라 Transformer

**주요 컴포넌트:**

**인코더:**
- 멀티 헤드 셀프 어텐션(Multi-Head Self-Attention)
- 포지션-와이즈 피드포워드(Position-wise Feed-Forward)
- 레이어 노멀라이제이션(Layer Normalization)
- 레지듀얼 커넥션(Residual Connection)

**디코더:**
- 마스크드 멀티 헤드 셀프 어텐션(Masked Self-Attention)
- 인코더-디코더 어텐션(Cross-Attention)
- 포지션-와이즈 피드포워드
- 레이어 노멀라이제이션
- 레지듀얼 커넥션

**포지셔널 인코딩:**
```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

#### 7.2.2. 장단점

**장점:**
- 완전한 병렬화 가능 → 학습 속도 빠름
- 롱 디펜던시 효과적 처리
- 최신 SOTA 성능
- 사전학습 모델 활용 가능

**단점:**
- 메모리 사용량 높음 (O(n²) 복잡도)
- 작은 데이터셋에서 오버피팅 위험
- 하이퍼파라미터 튜닝 복잡

#### 7.2.3. 권장 하이퍼파라미터

| 파라미터 | 권장값 | 설명 |
|---------|--------|------|
| d_model | 512 | 모델 차원 |
| num_heads | 8 | 어텐션 헤드 수 |
| d_ff | 2048 | 피드포워드 히든 차원 |
| num_layers | 6 | 인코더/디코더 레이어 수 |
| Dropout | 0.1 | 드롭아웃 비율 |
| Batch Size | 64-128 | 배치 크기 |
| Learning Rate | 0.0001 | 초기 학습률 (warmup 필요) |
| Warmup Steps | 4000 | 학습률 웜업 스텝 |

### 7.3. 사전학습 모델 활용

#### 7.3.1. BART (Bidirectional and Auto-Regressive Transformers)

**개발:** Facebook AI Research (FAIR)

**특징:**
- 디노이징 오토인코더(denoising autoencoder) 방식 사전학습
- 텍스트 재구성 태스크로 사전학습
- 요약 태스크에 특히 강력

**사전학습 방법:**
- 텍스트 인필링(text infilling): 연속 토큰 마스킹 후 재생성
- 문장 순서 섞기(sentence permutation)
- 문서 회전(document rotation)
- 토큰 마스킹, 삭제

**파인튜닝 권장사항:**
- 모델: `facebook/bart-base` 또는 `facebook/bart-large`
- 학습률: 3e-5 ~ 5e-5
- 에폭: 3-5
- Hugging Face Transformers 라이브러리 사용

**왜 BART인가?**
- 요약 태스크에서 검증된 성능
- 데이터 효율성 높음
- 구현 용이

#### 7.3.2. T5 (Text-to-Text Transfer Transformer)

**개발:** Google Research

**특징:**
- 모든 NLP 태스크를 텍스트-투-텍스트 형식으로 통일
- 프리픽스(prefix) 사용: `"summarize: [TEXT]"`
- C4 데이터셋으로 사전학습

**파인튜닝 권장사항:**
- 모델: `t5-small`, `t5-base`, `t5-large`
- 입력 형식: `"summarize: " + Text`
- 학습률: 1e-4
- 에폭: 3-5

**왜 T5인가?**
- 범용적 아키텍처
- 다양한 크기 선택 가능
- 전이 학습 효과 우수

#### 7.3.3. Pegasus

**개발:** Google Research

**특징:**
- 요약 전용 사전학습 모델
- GSG (Gap Sentence Generation) 태스크
- 중요 문장 마스킹 후 재생성

**파인튜닝 권장사항:**
- 모델: `google/pegasus-xsum` 또는 `google/pegasus-cnn_dailymail`
- 학습률: 5e-5
- 에폭: 3-5

**왜 Pegasus인가?**
- 요약 태스크에 최적화
- 적은 데이터로도 높은 성능
- 짧은 요약에 특히 효과적

#### 7.3.4. 모델 선택 가이드

**데이터셋 크기별:**
- 소규모 (< 10,000): Pegasus
- 중규모 (10,000 ~ 100,000): BART-base 또는 T5-base
- 대규모 (> 100,000): T5-large 또는 BART-large

**계산 자원별:**
- 제한적: T5-small, BART-base
- 충분함: T5-base, BART-large
- 풍부함: T5-large, Pegasus-large

**권장 우선순위:**
1. BART (요약 성능 우수, 범용성)
2. Pegasus (요약 전용, 데이터 효율성)
3. T5 (범용성, 다양한 크기)

### 7.4. 손실 함수 및 평가 메트릭

#### 7.4.1. 손실 함수 (Loss Function)

**1. 크로스 엔트로피 손실 (Cross-Entropy Loss)**
```
L = -Σᵢ Σₜ log P(yₜ | y₁, ..., yₜ₋₁, X)
```
- 표준 시퀀스 생성 손실
- 토큰 레벨 예측
- PyTorch: `nn.CrossEntropyLoss`

**2. 레이블 스무딩 (Label Smoothing)**
```
L = -Σₜ [(1-ε)log P(y*ₜ) + ε/|V| Σᵥ log P(vₜ)]
```
- ε = 0.1 권장
- 오버피팅 방지
- 일반화 성능 향상

**권장 조합:**
- Cross-Entropy + Label Smoothing

#### 7.4.2. 평가 메트릭 (Evaluation Metrics)

**1. ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

**ROUGE-N (Unigram/Bigram Overlap):**
```
ROUGE-N = Σ overlap(n-gram) / Σ count(n-gram in reference)
```

**ROUGE-L (Longest Common Subsequence):**
- 가장 긴 공통 부분수열 기반
- 단어 순서 고려

**특징:**
- 요약 평가의 표준
- ROUGE-1, ROUGE-2, ROUGE-L 주로 사용
- 재현율(recall) 중심

**2. BLEU (Bilingual Evaluation Understudy)**
```
BLEU = BP × exp(Σₙ wₙ log pₙ)
```
- n-gram 정밀도(precision) 기반
- 기계 번역에서 유래
- BLEU-4 주로 사용

**특징:**
- 단어 순서 고려
- 짧은 요약에 민감
- 0-100 스케일

**3. METEOR (Metric for Evaluation with Explicit ORdering)**
```
METEOR = Fₘₑₐₙ × (1 - Penalty)
```
- 동의어, 어간 매칭 고려
- 단어 순서 페널티
- 사람 평가와 높은 상관관계

**4. BERTScore**
- BERT 임베딩 기반 유사도
- 의미적 유사성 측정
- 최신 메트릭

**권장 평가 조합:**
- **필수:** ROUGE-1, ROUGE-2, ROUGE-L
- **보조:** BLEU-4, METEOR
- **고급:** BERTScore

**평가 예시:**
```
Reference: "Excellent coffee, great value"
Hypothesis: "Great coffee, good price"

ROUGE-1: 0.60 (단어 매칭도)
ROUGE-2: 0.33 (2-gram 매칭도)
ROUGE-L: 0.50 (최장 공통 부분수열)
BLEU-2: 0.35
```

#### 7.4.3. 사람 평가 (Human Evaluation)

**평가 차원:**

1. **유창성(Fluency): 1-5점**
   - 문법적 정확성
   - 자연스러운 표현

2. **관련성(Relevance): 1-5점**
   - 원문 내용 반영 정도
   - 정보 정확성

3. **간결성(Conciseness): 1-5점**
   - 불필요한 정보 없음
   - 적절한 길이

**평가 방법:**
- 100-200개 샘플 무작위 선택
- 2-3명 평가자 참여
- 평균 점수 계산
- Inter-annotator agreement 확인

---

## 8. 데이터 품질 이슈 및 대응

### 8.1. 중복 데이터

#### 8.1.1. 중복 유형

**1. 완전 중복**
- 동일한 Text와 Summary
- 동일한 ProductId, UserId 조합

**2. 준 중복**
- 약간의 표현 차이
- 본질적으로 동일한 내용

**3. 사용자 중복 리뷰**
- 동일 사용자가 동일 제품에 여러 리뷰
- 시간차 리뷰

#### 8.1.2. 중복 제거 전략

**방법 1: Text 해시 기반**
- Text의 해시값으로 중복 탐지
- 완전 중복 제거

**방법 2: (ProductId, UserId, Score) 조합**
- 동일 사용자의 동일 제품 리뷰 중 하나만 유지

**방법 3: 코사인 유사도 (고급)**
- TF-IDF 벡터화
- 유사도 > 0.95인 쌍 제거

**권장 전략:**
- 학습 데이터: 중복 제거 필수
- 테스트 데이터: 중복 유지 (실제 분포 반영)

### 8.2. 노이즈 및 불완전한 데이터

#### 8.2.1. 노이즈 유형

**1. 텍스트 품질 문제**
- 극단적으로 짧은 리뷰 (< 10 단어)
- 극단적으로 긴 리뷰 (> 500 단어)
- 비영어 리뷰
- 의미 없는 반복

**2. Summary 품질 문제**
- 빈 Summary
- 너무 긴 Summary (> 50 단어)
- Text와 무관한 Summary

**3. 메타데이터 문제**
- 결측값(missing values)
- 이상치(outliers)

#### 8.2.2. 필터링 기준

**품질 기준:**

1. **길이 기준**
   - 10 < Text 단어 수 < 500
   - 1 < Summary 단어 수 < 50

2. **언어 탐지**
   - langdetect 라이브러리 활용
   - 영어 리뷰만 선택

3. **의미 관련성**
   - Text와 Summary의 공통 단어 > 2개
   - 또는 코사인 유사도 > 0.3

4. **Helpfulness 필터 (선택적)**
   - HelpfulnessDenominator > 0
   - Helpfulness Ratio > 0.5

### 8.3. 데이터 필터링 전략

#### 8.3.1. 계층적 필터링

**Level 1: 기본 필터링**
- 결측값 제거
- 극단적 길이 제거
- 중복 제거

**Level 2: 품질 필터링**
- 언어 필터링
- 의미 관련성 확인

**Level 3: 균형 조정 (선택적)**
- Score 분포 균형
- 길이 분포 균형

#### 8.3.2. 필터링 후 데이터 통계

**예상 결과:**

| 단계 | 리뷰 수 | 제거 비율 |
|------|---------|----------|
| 원본 | 568,454 | - |
| 결측값 제거 | 565,000 | 0.6% |
| 길이 필터링 | 540,000 | 4.4% |
| 중복 제거 | 490,000 | 9.3% |
| 품질 필터링 | 450,000 | 8.2% |
| **최종** | **450,000** | **20.8%** |

**최종 데이터셋 특성:**
- 약 450,000개 고품질 리뷰
- 균형잡힌 길이 분포
- 명확한 Text-Summary 관계

---

## 9. 실험 설계 제안

### 9.1. 학습/검증/테스트 분할 전략

#### 9.1.1. 랜덤 스플릿 (Random Split)

**분할 비율:**
- 학습(Train): 80% (360,000개)
- 검증(Validation): 10% (45,000개)
- 테스트(Test): 10% (45,000개)

**장점:**
- 구현 간단
- IID 가정 만족

**단점:**
- 시간적 편향 무시
- 데이터 누출 가능성

**권장 용도:**
- 연구 및 벤치마크
- 모델 비교 실험

#### 9.1.2. 템포럴 스플릿 (Temporal Split)

**시간 기준 분할:**
- 학습: 1999 ~ 2010
- 검증: 2011
- 테스트: 2012

**장점:**
- 실제 배포 시나리오 반영
- 시간적 일반화 성능 확인
- 언어 트렌드 변화 대응

**단점:**
- 데이터 불균형 가능
- 최신 데이터 적음

**권장 용도:**
- 실전 배포 준비
- 일반화 성능 평가

#### 9.1.3. 제품 기준 스플릿 (선택적)

**ProductId 기준 분할:**
- 학습/검증/테스트에서 완전히 다른 제품 사용
- 새로운 제품에 대한 일반화 능력 테스트

**장점:**
- 제로샷(zero-shot) 일반화 평가
- 콜드 스타트 문제 시뮬레이션

**단점:**
- 더 어려운 태스크
- 성능 저하 예상

**권장 용도:**
- 고급 일반화 실험
- 실무 적용 전 검증

### 9.2. 하이퍼파라미터 튜닝 포인트

#### 9.2.1. 주요 튜닝 대상

**1. 학습률 (Learning Rate)** - 가장 중요
- 범위: [1e-5, 1e-4, 5e-4, 1e-3]
- 학습률 파인더(Learning Rate Finder) 사용 권장

**2. 배치 크기 (Batch Size)**
- 범위: [16, 32, 64, 128]
- GPU 메모리에 따라 조정

**3. 모델 크기**
- Hidden size: [256, 512, 1024]
- Num layers: [2, 4, 6]

**4. 정규화 (Regularization)**
- Dropout: [0.1, 0.3, 0.5]
- Weight decay: [0, 1e-5, 1e-4]

**5. 디코딩 파라미터**
- Beam size: [3, 5, 10]
- Length penalty: [0.6, 0.8, 1.0]

#### 9.2.2. 튜닝 전략

**1. 랜덤 서치 (Random Search)**
- 효율적 탐색
- 대규모 파라미터 공간에 적합

**2. 베이지안 최적화 (Bayesian Optimization)**
- Optuna 라이브러리 추천
- 효율적 탐색
- 자동화 가능

**권장 튜닝 순서:**
1. Learning rate (최우선)
2. Batch size
3. Model size
4. Regularization
5. Decoding parameters

#### 9.2.3. 학습 팁

**1. 그래디언트 클리핑**
- Max norm: 5.0
- 그래디언트 익스플로전(gradient explosion) 방지 필수

**2. 조기 종료 (Early Stopping)**
- Patience: 3-5 에폭
- Validation loss 기준
- 오버피팅 방지

**3. 체크포인트 저장**
- Best validation loss 모델 저장
- 주기적 저장 (매 에폭)

**4. 학습 모니터링**
- TensorBoard 또는 Wandb 사용
- Loss, ROUGE 추적
- 어텐션 시각화

**5. 학습률 스케줄링**
- Warmup + Linear Decay (Transformer)
- ReduceLROnPlateau (Seq2Seq)

---

## 10. PyTorch 구현 개요

### 10.1. 필수 라이브러리

**핵심 라이브러리:**
```python
# 딥러닝 프레임워크
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 데이터 처리
import pandas as pd
import numpy as np

# 평가
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu
```

**사전학습 모델 (Hugging Face):**
```python
from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)
```

### 10.2. 사전학습 모델 사용

**BART 모델 로드:**
```python
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
```

**T5 모델 로드:**
```python
model = T5ForConditionalGeneration.from_pretrained('t5-base')
tokenizer = T5Tokenizer.from_pretrained('t5-base')
```

**Pegasus 모델 로드:**
```python
from transformers import PegasusForConditionalGeneration, PegasusTokenizer

model = PegasusForConditionalGeneration.from_pretrained('google/pegasus-xsum')
tokenizer = PegasusTokenizer.from_pretrained('google/pegasus-xsum')
```

**파인튜닝 설정:**
```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    learning_rate=3e-5,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    evaluation_strategy="epoch"
)
```

---

## 11. 용어 목록

| 용어 (한글) | 영문 | 설명 |
|------------|------|------|
| 추상적 요약 | Abstractive Summarization | 원문을 이해하고 새로운 표현으로 요약 생성 |
| 추출적 요약 | Extractive Summarization | 원문에서 중요 문장을 추출하여 요약 |
| 어텐션 메커니즘 | Attention Mechanism | 입력의 중요한 부분에 집중하는 기법 |
| 인코더 | Encoder | 입력 시퀀스를 고정 길이 벡터로 변환 |
| 디코더 | Decoder | 벡터에서 출력 시퀀스를 생성 |
| 시퀀스 투 시퀀스 | Sequence-to-Sequence | 입력 시퀀스를 출력 시퀀스로 변환 |
| 트랜스포머 | Transformer | 어텐션 기반 신경망 아키텍처 |
| 셀프 어텐션 | Self-Attention | 시퀀스 내부 관계 모델링 |
| 멀티 헤드 어텐션 | Multi-Head Attention | 여러 어텐션을 병렬로 수행 |
| 포지셔널 인코딩 | Positional Encoding | 위치 정보를 임베딩에 추가 |
| 빔 서치 | Beam Search | 여러 후보로 최적 시퀀스 탐색 |
| 그리디 디코딩 | Greedy Decoding | 매 단계 최고 확률 토큰 선택 |
| 티처 포싱 | Teacher Forcing | 학습 시 실제 타겟을 입력으로 사용 |
| 크로스 엔트로피 | Cross-Entropy | 분류 문제의 표준 손실 함수 |
| 레이블 스무딩 | Label Smoothing | 오버피팅 방지 정규화 기법 |
| 그래디언트 클리핑 | Gradient Clipping | 그래디언트 익스플로전 방지 |
| 드롭아웃 | Dropout | 뉴런 무작위 제거 정규화 |
| 레이어 정규화 | Layer Normalization | 레이어 단위 정규화 |
| 레지듀얼 커넥션 | Residual Connection | 입력을 출력에 더하는 숏컷 |
| 임베딩 | Embedding | 단어를 밀집 벡터로 표현 |
| 토크나이제이션 | Tokenization | 텍스트를 토큰으로 분할 |
| 서브워드 | Subword | 단어보다 작은 단위의 토큰 |
| 바이트 페어 인코딩 | Byte Pair Encoding | 서브워드 토크나이제이션 알고리즘 |
| 보카뷸러리 | Vocabulary | 모델이 사용하는 토큰 집합 |
| 컨텍스트 벡터 | Context Vector | 입력 정보를 압축한 벡터 |
| 히든 스테이트 | Hidden State | RNN의 내부 상태 벡터 |
| 롱 디펜던시 | Long-Term Dependency | 멀리 떨어진 토큰 간 의존성 |
| 그래디언트 베니싱 | Gradient Vanishing | 그래디언트 소멸 현상 |
| 그래디언트 익스플로전 | Gradient Explosion | 그래디언트 폭발 현상 |
| 파인튜닝 | Fine-tuning | 사전학습 모델을 특정 태스크에 재학습 |
| 전이 학습 | Transfer Learning | 학습한 지식을 다른 태스크에 적용 |
| 사전학습 | Pre-training | 대규모 데이터로 모델을 먼저 학습 |
| 다운스트림 태스크 | Downstream Task | 사전학습 후 수행하는 특정 태스크 |
| 블루 스코어 | BLEU Score | n-gram 기반 번역 평가 메트릭 |
| 루즈 스코어 | ROUGE Score | 재현율 기반 요약 평가 메트릭 |
| 미티어 스코어 | METEOR Score | 동의어 고려 번역 평가 메트릭 |
| 버트스코어 | BERTScore | BERT 임베딩 기반 유사도 메트릭 |
| 카피 메커니즘 | Copy Mechanism | 입력에서 토큰을 직접 복사 |
| 커버리지 메커니즘 | Coverage Mechanism | 반복 생성 방지 |
| 포인터 제너레이터 | Pointer-Generator | 생성과 복사를 결합한 모델 |
| 오버피팅 | Overfitting | 학습 데이터에 과적합 |
| 언더피팅 | Underfitting | 데이터 패턴 학습 실패 |
| 일반화 | Generalization | 새로운 데이터에 대한 성능 |
| 하이퍼파라미터 | Hyperparameter | 학습 전 설정하는 파라미터 |
| 에폭 | Epoch | 전체 데이터셋을 한 번 학습 |
| 배치 | Batch | 한 번에 처리하는 데이터 묶음 |
| 학습률 | Learning Rate | 파라미터 업데이트 크기 |
| 옵티마이저 | Optimizer | 파라미터 업데이트 알고리즘 |
| 아담 | Adam | 적응형 학습률 옵티마이저 |
| 확률적 경사 하강법 | Stochastic Gradient Descent | 기본 최적화 알고리즘 |
| 백프로퍼게이션 | Backpropagation | 그래디언트 역전파 알고리즘 |
| 순전파 | Forward Pass | 입력에서 출력으로 계산 |
| 역전파 | Backward Pass | 출력에서 입력으로 그래디언트 계산 |
| 웜업 | Warmup | 학습률 점진적 증가 |
| 조기 종료 | Early Stopping | 성능 개선 없을 때 학습 중단 |
| 체크포인트 | Checkpoint | 학습 중간 모델 저장 |
| 데이터 증강 | Data Augmentation | 데이터 변형으로 다양성 증가 |
| 백 트랜슬레이션 | Back Translation | 번역 후 재번역 증강 기법 |
| 템포럴 스플릿 | Temporal Split | 시간 기준 데이터 분할 |
| 크로스 밸리데이션 | Cross-Validation | 여러 폴드로 나누어 검증 |
| 앙상블 | Ensemble | 여러 모델 예측 결합 |
| 병렬 코퍼스 | Parallel Corpus | 쌍으로 구성된 데이터셋 |
| 자기회귀적 | Autoregressive | 이전 출력을 다음 입력으로 사용 |
| 마스킹 | Masking | 특정 토큰을 가리는 기법 |
| 디노이징 | Denoising | 노이즈 제거 학습 |
| 압축률 | Compression Ratio | 입력 대비 출력 길이 비율 |
| 트렁케이션 | Truncation | 시퀀스를 최대 길이로 자르기 |
| 패딩 | Padding | 시퀀스를 고정 길이로 채우기 |
| 언노운 토큰 | Unknown Token | 보카뷸러리에 없는 토큰 |
| 스페셜 토큰 | Special Token | 특수 목적 토큰 (BOS, EOS 등) |
| 집프 법칙 | Zipf's Law | 단어 빈도 분포 법칙 |
| 롱 테일 디스트리뷰션 | Long-Tail Distribution | 소수가 높은 빈도를 차지하는 분포 |
| 시맨틱 유사도 | Semantic Similarity | 의미적 유사성 |
| 파라프레이징 | Paraphrasing | 다른 표현으로 바꾸어 말하기 |
| 센티먼트 | Sentiment | 감정, 정서 |
| 인사이트 | Insight | 통찰, 이해 |
| 메타데이터 | Metadata | 데이터에 대한 데이터 |
| 아키텍처 | Architecture | 모델 구조 |
| 임플리케이션 | Implication | 함의, 영향 |
| 벤치마크 | Benchmark | 성능 비교 기준 |

---

## 참고문헌

### 기초 논문

1. **Seq2Seq**
   - Sutskever, I., Vinyals, O., & Le, Q. V. (2014). "Sequence to Sequence Learning with Neural Networks." *NeurIPS*.

2. **Attention Mechanism**
   - Bahdanau, D., Cho, K., & Bengio, Y. (2015). "Neural Machine Translation by Jointly Learning to Align and Translate." *ICLR*.
   - Luong, M. T., Pham, H., & Manning, C. D. (2015). "Effective Approaches to Attention-based Neural Machine Translation." *EMNLP*.

3. **Transformer**
   - Vaswani, A., et al. (2017). "Attention is All You Need." *NeurIPS*.

### 사전학습 모델

4. **BART**
   - Lewis, M., et al. (2020). "BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension." *ACL*.

5. **T5**
   - Raffel, C., et al. (2020). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer." *JMLR*.

6. **Pegasus**
   - Zhang, J., et al. (2020). "PEGASUS: Pre-training with Extracted Gap-sentences for Abstractive Summarization." *ICML*.

### 평가 메트릭

7. **BLEU**
   - Papineni, K., et al. (2002). "BLEU: a Method for Automatic Evaluation of Machine Translation." *ACL*.

8. **ROUGE**
   - Lin, C. Y. (2004). "ROUGE: A Package for Automatic Evaluation of Summaries." *ACL Workshop*.

9. **METEOR**
   - Banerjee, S., & Lavie, A. (2005). "METEOR: An Automatic Metric for MT Evaluation with Improved Correlation with Human Judgments." *ACL Workshop*.

10. **BERTScore**
    - Zhang, T., et al. (2020). "BERTScore: Evaluating Text Generation with BERT." *ICLR*.

### 데이터셋

11. **Amazon Fine Food Reviews**
    - McAuley, J., & Leskovec, J. (2013). "From amateurs to connoisseurs: modeling the evolution of user expertise through online reviews." *WWW*.

### 추가 참고자료

12. **Copy Mechanism**
    - See, A., Liu, P. J., & Manning, C. D. (2017). "Get To The Point: Summarization with Pointer-Generator Networks." *ACL*.

13. **Coverage Mechanism**
    - Tu, Z., et al. (2016). "Modeling Coverage for Neural Machine Translation." *ACL*.

---

## 프로젝트 체크리스트

### 데이터 준비 단계
- [ ] Kaggle에서 데이터 다운로드
- [ ] CSV 파일 로드 및 탐색
- [ ] 결측값 확인 및 제거
- [ ] 길이 분포 분석
- [ ] 중복 데이터 제거
- [ ] 품질 필터링 적용
- [ ] Train/Validation/Test 분할 (80/10/10)
- [ ] 전처리 파이프라인 구축

### 모델 준비 단계
- [ ] 토크나이저 선택 (BPE/SentencePiece)
- [ ] 보카뷸러리 생성 또는 사전학습 토크나이저 로드
- [ ] 모델 선택 (Seq2Seq/Transformer/BART/T5/Pegasus)
- [ ] 모델 아키텍처 구현 또는 로드
- [ ] Dataset 클래스 구현
- [ ] DataLoader 설정

### 학습 단계
- [ ] 손실 함수 정의 (Cross-Entropy + Label Smoothing)
- [ ] 옵티마이저 설정 (Adam/AdamW)
- [ ] 학습률 스케줄러 설정
- [ ] 그래디언트 클리핑 설정
- [ ] 학습 루프 구현
- [ ] 검증 루프 구현
- [ ] 조기 종료 구현
- [ ] 체크포인트 저장 로직
- [ ] 모니터링 도구 설정 (TensorBoard/Wandb)

### 평가 단계
- [ ] ROUGE 평가 함수 구현
- [ ] BLEU 평가 함수 구현
- [ ] 빔 서치 구현
- [ ] 생성 예시 샘플링 (10-20개)
- [ ] 정량적 평가 수행
- [ ] 정성적 평가 (사람 평가)

### 분석 단계
- [ ] 어텐션 가중치 추출
- [ ] 어텐션 히트맵 시각화
- [ ] 길이별 성능 분석
- [ ] Score별 성능 분석
- [ ] 오류 사례 분석
- [ ] 모델 비교 (baseline vs 최종)

### 문서화 단계
- [ ] 코드 주석 작성
- [ ] README.md 작성
- [ ] 실험 결과 정리
- [ ] GitHub Pages 문서 업데이트
- [ ] 시각화 자료 생성
- [ ] 최종 보고서 작성

---

## 실험 결과 예시 템플릿

### 모델 성능 비교

| 모델 | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU-4 | 학습 시간 |
|------|---------|---------|---------|--------|----------|
| Seq2Seq + Attention | 0.35 | 0.15 | 0.30 | 0.20 | 4시간 |
| Transformer (from scratch) | 0.40 | 0.18 | 0.35 | 0.25 | 6시간 |
| BART-base (fine-tuned) | 0.45 | 0.22 | 0.40 | 0.30 | 2시간 |
| T5-base (fine-tuned) | 0.44 | 0.21 | 0.39 | 0.29 | 2시간 |
| Pegasus (fine-tuned) | 0.47 | 0.24 | 0.42 | 0.32 | 2시간 |

*위 수치는 예시이며, 실제 실험 결과로 대체할 것*

### 생성 예시

**Example 1:**
```
Input Text (Text):
"This coffee is absolutely delicious. The flavor is rich and smooth, 
not bitter at all. The aroma fills the kitchen when brewing. 
The packaging keeps it fresh. Highly recommend for coffee lovers."

Reference Summary:
"Excellent coffee"

Generated Summary (BART):
"Delicious and smooth coffee"

ROUGE-L: 0.50
```

**Example 2:**
```
Input Text (Text):
"I was very disappointed with this product. The taste was bland 
and the texture was weird. Not worth the money at all."

Reference Summary:
"Disappointed with quality"

Generated Summary (BART):
"Poor quality, not recommended"

ROUGE-L: 0.40
```

---

## 추가 실험 아이디어

### 1. 멀티태스크 러닝
- 주 태스크: 요약 생성
- 보조 태스크: Score 예측
- 인코더 공유, 디코더 분리

### 2. 컨디셔널 제너레이션
- Score를 조건으로 요약 생성
- "5점짜리 리뷰처럼 요약해줘"
- 다양한 스타일 요약 가능

### 3. 강화 학습 적용
- Policy Gradient 방법
- Reward: ROUGE 스코어
- Exploration vs Exploitation

### 4. 도메인 적응
- 식품 → 전자제품 리뷰
- 전이 학습 효과 분석

### 5. 모델 경량화
- 지식 증류(Knowledge Distillation)
- 양자화(Quantization)
- 프루닝(Pruning)

---

## 자주 묻는 질문 (FAQ)

**Q1: 어떤 모델로 시작하는 것이 좋나요?**

A: 초보자라면 사전학습 모델(BART 또는 T5)로 시작하는 것을 강력히 권장합니다. 이유는:
- 구현이 간단 (Hugging Face Transformers 활용)
- 학습 시간 짧음 (2-3시간)
- 높은 성능 보장
- 처음부터 구현하려면 Seq2Seq + Attention부터 시작

**Q2: 데이터 전처리에서 가장 중요한 것은?**

A: 세 가지가 핵심입니다:
1. 중복 제거 (학습 데이터 품질)
2. 길이 필터링 (극단값 제거)
3. Text-Summary 관련성 확인 (의미 없는 쌍 제거)

**Q3: ROUGE 스코어가 낮게 나오는데 정상인가요?**

A: 요약 태스크에서 ROUGE-1 0.40-0.50, ROUGE-L 0.35-0.45 정도면 좋은 성능입니다. 참고로:
- ROUGE는 단어 매칭 기반이라 의미적으로 좋은 요약도 낮은 점수 가능
- BERTScore 같은 의미 기반 메트릭 병행 사용 권장
- 정성적 평가도 중요

**Q4: GPU 메모리가 부족한데 어떻게 하나요?**

A: 여러 해결 방법이 있습니다:
1. 배치 크기 줄이기 (32 → 16 → 8)
2. 그래디언트 어큐뮬레이션(gradient accumulation) 사용
3. Mixed Precision Training (FP16)
4. 작은 모델 선택 (BART-base 대신 T5-small)
5. 시퀀스 최대 길이 줄이기 (200 → 150)

**Q5: 학습이 수렴하지 않는데 왜 그런가요?**

A: 일반적인 원인:
1. 학습률이 너무 높음 → 1/10로 줄이기
2. 그래디언트 클리핑 미적용 → max_norm=5.0 설정
3. 배치 크기가 너무 작음 → 32 이상 권장
4. 데이터 품질 문제 → 전처리 재확인

**Q6: 생성된 요약이 반복되는데 어떻게 해결하나요?**

A: 세 가지 방법:
1. `no_repeat_ngram_size=3` 설정 (빔 서치 시)
2. 커버리지 메커니즘 추가
3. 렝스 펄털티 조정 (α=0.8 정도)

**Q7: 얼마나 학습시켜야 하나요?**

A: 모델별 권장:
- 사전학습 모델: 3-5 에폭 (2-3시간)
- Transformer from scratch: 10-20 에폭 (6-12시간)
- Seq2Seq: 15-30 에폭 (4-8시간)
- 조기 종료 사용 권장 (patience=3)

**Q8: 어텐션 시각화는 필수인가요?**

A: 필수는 아니지만 강력히 권장합니다:
- 모델이 무엇을 학습했는지 이해
- 디버깅에 유용
- 보고서/논문에서 좋은 시각 자료
- 구현은 matplotlib으로 간단히 가능

---

## 마무리

### 핵심 요약

본 문서는 Amazon Fine Food Reviews 데이터셋을 활용한 **리뷰 요약 생성(Review Summarization)** 프로젝트의 완전한 가이드를 제공한다.

**주요 내용:**

1. **데이터셋 분석**
   - 568,454개의 식품 리뷰
   - Text(입력) - Summary(출력) 병렬 코퍼스
   - 압축률 약 1:20

2. **모델 선택**
   - Baseline: Seq2Seq with Attention
   - Advanced: Transformer
   - Recommended: 사전학습 모델 (BART/T5/Pegasus)

3. **평가 방법**
   - ROUGE-1, ROUGE-2, ROUGE-L (필수)
   - BLEU, METEOR (보조)
   - 사람 평가 (유창성, 관련성, 간결성)

4. **핵심 기술**
   - 어텐션 메커니즘으로 핵심 정보 포착
   - 서브워드 토크나이제이션으로 OOV 해결
   - 빔 서치로 고품질 요약 생성

### 프로젝트 진행 로드맵

**Week 1: 데이터 준비**
- 데이터 다운로드 및 탐색
- EDA (Exploratory Data Analysis)
- 전처리 파이프라인 구축
- Train/Val/Test 분할

**Week 2: Baseline 모델**
- Seq2Seq + Attention 구현
- 기본 학습 및 평가
- 어텐션 시각화

**Week 3: 사전학습 모델**
- BART 또는 T5 파인튜닝
- 하이퍼파라미터 튜닝
- 성능 비교

**Week 4: 분석 및 문서화**
- 정량적/정성적 평가
- 오류 분석
- 최종 보고서 작성
- GitHub Pages 문서 완성

### 학습 성과

이 프로젝트를 통해 다음을 학습할 수 있다:

1. **이론적 이해**
   - Seq2Seq 아키텍처의 작동 원리
   - 어텐션 메커니즘의 필요성과 효과
   - Transformer의 혁신적 설계

2. **실전 경험**
   - 대규모 텍스트 데이터 전처리
   - PyTorch를 활용한 모델 구현
   - 사전학습 모델 파인튜닝
   - 하이퍼파라미터 튜닝 전략

3. **평가 능력**
   - 자동 평가 메트릭 (ROUGE, BLEU) 이해
   - 정성적 평가 설계
   - 모델 비교 및 분석

4. **엔지니어링 스킬**
   - 효율적인 데이터 파이프라인 구축
   - 학습 모니터링 및 디버깅
   - 모델 배포 준비

### 다음 단계

프로젝트 완료 후 다음으로 나아갈 수 있는 방향:

1. **다른 데이터셋 적용**
   - CNN/DailyMail (뉴스 요약)
   - XSum (극도로 짧은 요약)
   - Multi-News (다중 문서 요약)

2. **고급 기법 적용**
   - 강화 학습 (ROUGE 최적화)
   - 멀티태스크 러닝
   - Few-shot 학습

3. **실전 배포**
   - REST API 구축 (Flask/FastAPI)
   - 웹 인터페이스 개발
   - 모델 경량화 및 최적화

4. **연구 확장**
   - 논문 작성
   - 오픈소스 기여
   - 캐글 컴피티션 참여

### 추천 학습 자료

**온라인 강의:**
- CS224N (Stanford): Natural Language Processing with Deep Learning
- fast.ai: Practical Deep Learning for Coders
- Hugging Face Course: NLP with Transformers

**서적:**
- "Speech and Language Processing" by Jurafsky & Martin
- "Natural Language Processing with Transformers" by Tunstall et al.
- "Deep Learning" by Goodfellow et al.

**블로그/튜토리얼:**
- Jay Alammar's Blog (Illustrated Transformer)
- Hugging Face Documentation
- PyTorch Official Tutorials

---

## 프로젝트 성공을 위한 최종 조언

### Do's (해야 할 것)

1. **작게 시작하기**
   - 전체 데이터셋 대신 10,000개 샘플로 시작
   - 빠른 실험으로 파이프라인 검증
   - 점진적으로 확장

2. **자주 평가하기**
   - 매 에폭 검증
   - 중간 생성 결과 확인
   - 어텐션 시각화로 학습 상태 모니터링

3. **체크포인트 저장**
   - 최고 성능 모델 보존
   - 다양한 설정 실험 가능
   - 재현 가능성 확보

4. **문서화하기**
   - 실험 설정 기록
   - 결과 정리
   - 코드 주석 작성

5. **커뮤니티 활용**
   - Hugging Face Forums
   - Reddit r/MachineLearning
   - Stack Overflow

### Don'ts (하지 말아야 할 것)

1. **처음부터 완벽 추구**
   - 일단 동작하는 baseline 먼저
   - 점진적 개선

2. **과도한 최적화**
   - 조기 최적화는 악의 근원
   - 병목 확인 후 최적화

3. **평가 무시**
   - Loss만 보지 말고 실제 생성 결과 확인
   - ROUGE 스코어만 맹신하지 말 것

4. **데이터 품질 간과**
   - "Garbage in, garbage out"
   - 전처리에 충분한 시간 투자

5. **하나의 메트릭에 집착**
   - 다양한 각도로 평가
   - 정성적 평가도 중요

---

**이 문서가 Amazon Fine Food Reviews 데이터셋 분석과 Seq2Seq/Transformer 학습에 도움이 되기를 바랍니다!**

**프로젝트의 성공을 응원합니다! 🎉**

---

*문서 버전: 1.0*  
*최종 수정일: 2025년 10월*  
*작성 목적: AI 엔지니어 학생의 딥러닝 학습 및 GitHub Pages 문서화*
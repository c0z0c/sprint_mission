---
layout: default
title: "Hugging Face Transformers 라이브러리 완벽 가이드"
description: "Hugging Face Transformers 라이브러리 완벽 가이드"
date: 2025-10-19
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# Hugging Face Transformers 라이브러리 완벽 가이드

## 목차

- [1. Hugging Face Transformers 개요](#1-hugging-face-transformers-개요)<br/>
  - [1.1. 라이브러리의 목적과 철학](#11-라이브러리의-목적과-철학)<br/>
  - [1.2. 주요 특징](#12-주요-특징)<br/>
- [2. Hugging Face 생태계](#2-hugging-face-생태계)<br/>
  - [2.1. Hugging Face Hub 사이트](#21-hugging-face-hub-사이트)<br/>
  - [2.2. 모델 테스트 및 탐색 방법](#22-모델-테스트-및-탐색-방법)<br/>
  - [2.3. 데이터셋과 스페이스](#23-데이터셋과-스페이스)<br/>
- [3. Transformers 라이브러리 핵심 구조](#3-transformers-라이브러리-핵심-구조)<br/>
  - [3.1. 파이프라인 아키텍처](#31-파이프라인-아키텍처)<br/>
  - [3.2. 모델 클래스 계층구조](#32-모델-클래스-계층구조)<br/>
  - [3.3. 토크나이저 시스템](#33-토크나이저-시스템)<br/>
- [4. KoBART를 활용한 문서 요약 미션](#4-kobart를-활용한-문서-요약-미션)<br/>
  - [4.1. KoBART 모델 소개](#41-kobart-모델-소개)<br/>
  - [4.2. 대규모 뉴스 데이터 학습 전략](#42-대규모-뉴스-데이터-학습-전략)<br/>
  - [4.3. 필수 임포트 구성](#43-필수-임포트-구성)<br/>
- [5. 도메인 특화 파인튜닝](#5-도메인-특화-파인튜닝)<br/>
  - [5.1. 사용자 정의 어휘사전 구축](#51-사용자-정의-어휘사전-구축)<br/>
  - [5.2. 토크나이저 커스터마이징](#52-토크나이저-커스터마이징)<br/>
- [6. KoBART 친화적 데이터셋 구조](#6-kobart-친화적-데이터셋-구조)<br/>
  - [6.1. 권장 폴더 및 파일 구조](#61-권장-폴더-및-파일-구조)<br/>
  - [6.2. JSON 포맷 스키마](#62-json-포맷-스키마)<br/>
  - [6.3. 데이터 로딩 클래스 및 라이브러리](#63-데이터-로딩-클래스-및-라이브러리)<br/>
- [용어 목록](#용어-목록)<br/>

---

## 1. Hugging Face Transformers 개요

### 1.1. 라이브러리의 목적과 철학

Hugging Face Transformers는 자연어처리(NLP) 분야에서 사전학습된(프리트레인드, Pre-trained) 트랜스포머 모델들을 쉽게 사용할 수 있도록 설계된 오픈소스 라이브러리입니다. 이 라이브러리의 핵심 철학은 "민주화(데모크러타이제이션, Democratization)"입니다. 최첨단 AI 모델을 연구자나 대기업뿐만 아니라 모든 개발자가 접근하고 활용할 수 있도록 만드는 것이 목표입니다.

라이브러리는 PyTorch, TensorFlow, JAX 등 주요 딥러닝 프레임워크를 모두 지원하며, 동일한 API를 통해 일관된 사용 경험을 제공합니다. 이는 프레임워크 간 전환 비용을 최소화하고, 모델 재현성(리프로듀서빌리티, Reproducibility)을 보장합니다.

### 1.2. 주요 특징

**통합 인터페이스(유니파이드 인터페이스, Unified Interface)**: 수천 개의 서로 다른 모델들이 일관된 API를 통해 제공됩니다. BERT, GPT, T5, BART 등 다양한 아키텍처를 동일한 방식으로 로드하고 사용할 수 있습니다.

**태스크 중립성(태스크 애그노스틱, Task-Agnostic)**: 텍스트 분류, 질의응답, 요약, 번역 등 다양한 다운스트림(Downstream) 태스크에 동일한 모델을 쉽게 적용할 수 있습니다.

**자동 구성(오토 컨피규레이션, Auto-configuration)**: AutoModel, AutoTokenizer 클래스를 통해 모델 이름만으로도 적절한 클래스와 설정이 자동으로 결정됩니다.

**효율적인 메모리 관리**: 그래디언트 체크포인팅(Gradient Checkpointing), 혼합 정밀도(믹스드 프리시전, Mixed Precision) 학습, 모델 병렬화(패럴렐라이제이션, Parallelization) 등 대규모 모델 학습을 위한 최적화 기법들이 내장되어 있습니다.

## 2. Hugging Face 생태계

### 2.1. Hugging Face Hub 사이트

Hugging Face Hub(https://huggingface.co)는 모델, 데이터셋, 데모를 공유하는 중앙 집중식 플랫폼입니다. GitHub와 유사한 버전 관리 시스템을 통해 모델의 웨이트(가중치, Weights), 설정 파일, 토크나이저 등을 관리합니다.

Hub는 크게 세 가지 주요 섹션으로 구성됩니다:

**Models**: 60만 개 이상의 사전학습된 모델이 호스팅됩니다. 각 모델은 모델 카드(Model Card)를 통해 사용법, 성능 메트릭(Metrics), 학습 데이터, 제한사항(리미테이션, Limitations) 등의 정보를 제공합니다.

**Datasets**: 1만 개 이상의 공개 데이터셋이 표준화된 포맷으로 제공됩니다. 데이터셋 뷰어(Viewer)를 통해 웹에서 직접 데이터를 탐색할 수 있습니다.

**Spaces**: Gradio 또는 Streamlit 기반의 인터랙티브(Interactive) 데모를 배포할 수 있는 공간입니다. 별도의 서버 설정 없이 모델을 웹 애플리케이션으로 공유할 수 있습니다.

### 2.2. 모델 테스트 및 탐색 방법

Hub에서 모델을 테스트하는 방법은 여러 가지가 있습니다:

**인퍼런스(Inference) API**: 모델 페이지 오른쪽에 있는 위젯을 통해 브라우저에서 직접 모델을 테스트할 수 있습니다. 입력 텍스트를 넣으면 실시간으로 결과를 확인할 수 있습니다. 이는 서버리스(Serverless) 방식으로 작동하며, 별도의 설치나 설정이 필요 없습니다.

**필터링 및 검색**: 태스크, 라이브러리, 언어, 라이센스 등 다양한 기준으로 모델을 필터링할 수 있습니다. 한국어 요약 모델을 찾고 싶다면 "Task: Summarization", "Language: Korean"으로 필터링하면 됩니다.

**모델 비교**: 리더보드(Leaderboard) 섹션에서 동일한 태스크에 대한 여러 모델의 성능을 비교할 수 있습니다. ROUGE, BLEU 등의 메트릭이 제공됩니다.

**코드 스니펫(Snippet)**: 각 모델 페이지에는 해당 모델을 로드하고 사용하는 샘플 코드가 자동으로 생성되어 제공됩니다. 이를 복사하여 바로 사용할 수 있습니다.

### 2.3. 데이터셋과 스페이스

**데이터셋 구조**: Hugging Face 데이터셋은 Apache Arrow 포맷을 기반으로 하여 메모리 효율적이고 빠른 데이터 로딩을 제공합니다. 데이터셋은 스플릿(Split) 개념을 사용하여 train, validation, test로 구분됩니다.

**스페이스 활용**: 연구 결과나 프로토타입을 빠르게 공유하고 피드백을 받을 수 있습니다. GPU 지원도 가능하여 대규모 모델도 배포할 수 있습니다.

## 3. Transformers 라이브러리 핵심 구조

### 3.1. 파이프라인 아키텍처

파이프라인은 Transformers 라이브러리의 가장 높은 수준의 추상화(앱스트랙션, Abstraction)입니다. 전처리, 모델 추론, 후처리를 하나의 통합된 인터페이스로 제공합니다.

파이프라인의 내부 동작 원리는 다음과 같습니다:

**토큰화(Tokenization) 단계**: 입력 텍스트를 모델이 이해할 수 있는 숫자 시퀀스(Sequence)로 변환합니다. 이 과정에서 스페셜 토큰(Special Tokens)이 추가되고, 어텐션 마스크(Attention Mask)가 생성됩니다.

**모델 추론 단계**: 토큰화된 입력이 모델을 통과하여 로짓(Logits) 또는 임베딩(Embeddings)을 생성합니다. 이 단계에서 GPU 가속이 자동으로 적용됩니다.

**후처리 단계**: 모델의 원시 출력을 사람이 이해할 수 있는 형태로 변환합니다. 예를 들어, 분류 태스크에서는 클래스 레이블과 확률값으로, 생성 태스크에서는 디코딩된 텍스트로 변환됩니다.

### 3.2. 모델 클래스 계층구조

Transformers의 모델 클래스는 계층적으로 설계되어 있습니다:

**베이스 모델 클래스**: PreTrainedModel이 최상위 추상 클래스입니다. 모든 모델은 이를 상속받아 공통 메서드(from_pretrained, save_pretrained 등)를 구현합니다.

**아키텍처별 베이스 클래스**: BertModel, GPT2Model, BartModel 등 각 아키텍처의 베이스 구현입니다. 이들은 트랜스포머 레이어 스택(Stack)만을 포함하며, 태스크별 헤드(Head)는 포함하지 않습니다.

**태스크별 모델 클래스**: BertForSequenceClassification, BartForConditionalGeneration 등 특정 태스크를 위한 출력 레이어가 추가된 클래스입니다. 이들은 손실 함수(로스 펑션, Loss Function)도 내장하고 있습니다.

**Auto 클래스**: AutoModel, AutoModelForSeq2SeqLM 등은 모델 이름이나 체크포인트(Checkpoint) 경로를 기반으로 적절한 모델 클래스를 자동으로 선택합니다. 이는 리플렉션(Reflection) 메커니즘을 사용합니다.

### 3.3. 토크나이저 시스템

토크나이저는 텍스트와 모델 입력 사이의 브리지(Bridge) 역할을 합니다:

**서브워드(Subword) 토큰화**: BERT는 WordPiece, GPT는 BPE(Byte Pair Encoding), BART는 BPE 변형을 사용합니다. 이는 미등록 단어(OOV, Out-of-Vocabulary) 문제를 해결합니다.

**스페셜 토큰 처리**: [CLS], [SEP], [PAD] 등의 토큰이 자동으로 추가됩니다. 이들은 모델이 문장의 시작, 끝, 패딩을 인식하도록 합니다.

**어텐션 마스크**: 실제 토큰과 패딩 토큰을 구분하기 위한 바이너리(Binary) 마스크입니다. 이는 패딩 토큰이 어텐션 계산에 영향을 주지 않도록 합니다.

**토큰 타입 ID**: 멀티 시퀀스 입력(예: 질문-문맥 쌍)에서 각 토큰이 어느 세그먼트(Segment)에 속하는지 표시합니다.

## 4. KoBART를 활용한 문서 요약 미션

### 4.1. KoBART 모델 소개

KoBART(Korean BART)는 SKT에서 공개한 한국어 시퀀스-투-시퀀스(Sequence-to-Sequence) 모델입니다. BART 아키텍처를 기반으로 하며, 한국어 텍스트 생성 태스크에 특화되어 있습니다.

**아키텍처 특징**: KoBART는 인코더-디코더(Encoder-Decoder) 구조를 가집니다. 인코더는 입력 문서를 컨텍스트(Context) 벡터로 변환하고, 디코더는 이를 기반으로 요약문을 생성합니다. 이는 양방향 컨텍스트와 자기회귀적(오토리그레시브, Autoregressive) 생성의 장점을 결합합니다.

**사전학습 방법**: 디노이징(Denoising) 오토인코더 방식으로 학습되었습니다. 입력 텍스트의 일부를 마스킹하거나 순서를 섞은 후, 원본 텍스트를 복원하도록 학습합니다. 이는 모델이 문맥을 이해하고 일관성 있는 텍스트를 생성하는 능력을 키웁니다.

**한국어 특화**: 한국어 위키피디아, 뉴스 등 대규모 한국어 코퍼스(Corpus)로 학습되었습니다. 한국어 형태소 분석을 고려한 토크나이저를 사용하여 조사, 어미 등을 효과적으로 처리합니다.

### 4.2. 대규모 뉴스 데이터 학습 전략

1TB 규모의 뉴스 데이터를 Colab L4 GPU에서 학습하는 것은 메모리와 시간 측면에서 도전적입니다. 효율적인 전략이 필요합니다:

**데이터 스트리밍**: 전체 데이터를 메모리에 로드하지 않고, 배치(Batch) 단위로 스트리밍합니다. Hugging Face Datasets 라이브러리의 `streaming=True` 옵션을 활용하면 디스크에서 직접 읽어옵니다.

**그래디언트 어큐뮬레이션(Gradient Accumulation)**: GPU 메모리 제약으로 작은 배치 사이즈를 사용해야 할 때, 여러 배치의 그래디언트를 누적하여 큰 배치 효과를 냅니다. 이는 accumulation_steps 파라미터로 설정합니다.

**혼합 정밀도 학습**: FP16(16비트 부동소수점) 또는 BF16을 사용하여 메모리 사용량을 절반으로 줄이고 연산 속도를 높입니다. Trainer의 fp16=True 옵션으로 활성화됩니다.

**체크포인팅 전략**: 정기적으로 모델 체크포인트를 저장하여 학습 중단 시 재시작할 수 있도록 합니다. save_steps와 save_total_limit 파라미터로 관리합니다.

**LoRA 또는 프리픽스 튜닝(Prefix Tuning)**: 파라미터 효율적 파인튜닝(PEFT, Parameter-Efficient Fine-Tuning) 기법을 사용하여 전체 모델이 아닌 일부만 학습합니다. 이는 메모리와 시간을 크게 절약합니다.

### 4.3. 필수 임포트 구성

KoBART 파인튜닝을 위한 기본 임포트는 다음과 같습니다:

**모델 및 토크나이저 관련**: transformers 라이브러리에서 BartForConditionalGeneration과 PreTrainedTokenizerFast를 임포트합니다. Auto 클래스를 사용하면 모델 이름만으로 자동 로딩이 가능합니다.

**데이터 처리**: datasets 라이브러리로 데이터셋을 로드하고 전처리합니다. load_dataset 함수는 로컬 파일, Hub 데이터셋, 커스텀 스크립트를 지원합니다.

**학습 관련**: transformers의 Trainer와 TrainingArguments가 학습 루프를 추상화합니다. Seq2SeqTrainer와 Seq2SeqTrainingArguments는 생성 태스크에 특화된 기능을 제공합니다.

**평가 메트릭**: evaluate 라이브러리 또는 datasets.load_metric으로 ROUGE, BLEU 등을 로드합니다. compute_metrics 함수에서 사용됩니다.

**유틸리티**: torch는 GPU 관리, 랜덤 시드 설정 등에 사용됩니다. numpy와 pandas는 데이터 전처리에 활용됩니다.

**PEFT 라이브러리**: 파라미터 효율적 파인튜닝을 위해 peft 패키지를 사용할 수 있습니다. LoraConfig와 get_peft_model이 주요 컴포넌트입니다.

**로깅 및 모니터링**: wandb, tensorboard 등으로 학습 과정을 추적합니다. logging 모듈로 로그 레벨을 설정합니다.

## 5. 도메인 특화 파인튜닝

### 5.1. 사용자 정의 어휘사전 구축

도메인 특화 어휘사전은 모델이 특정 분야의 용어를 더 잘 이해하도록 합니다:

**어휘 추출**: 도메인 텍스트에서 고빈도 토큰을 추출합니다. 기존 토크나이저로 토큰화했을 때 여러 서브워드로 분리되는 용어들을 찾습니다. 예를 들어, "코로나19"가 ["코로나", "##19"]로 분리된다면 이를 단일 토큰으로 추가합니다.

**토큰 선별**: TF-IDF, PMI(Pointwise Mutual Information) 등의 통계 지표로 도메인 특화 용어를 식별합니다. 일반 코퍼스와 비교하여 도메인에서 유의미하게 높은 빈도를 보이는 용어를 선택합니다.

**형태소 분석 활용**: 한국어의 경우 형태소 분석기(Mecab, Okt 등)로 복합명사, 전문용어를 추출합니다. "인공지능", "딥러닝" 같은 용어가 단일 의미 단위로 처리되도록 합니다.

### 5.2. 토크나이저 커스터마이징

**어휘 확장**: 기존 토크나이저의 vocab.json에 새로운 토큰을 추가합니다. tokenizer.add_tokens() 메서드를 사용하면 됩니다. 모델의 임베딩 레이어도 resize_token_embeddings()로 확장해야 합니다.

**스페셜 토큰 추가**: 도메인별 마커(Marker) 토큰을 추가할 수 있습니다. 예를 들어, [PRODUCT], [COMPANY] 같은 엔티티(Entity) 태그를 추가하여 모델이 중요 정보를 인식하도록 합니다.

**정규화(Normalization) 규칙**: 토크나이저의 normalizer를 수정하여 도메인별 전처리를 수행합니다. 예를 들어, 금융 도메인에서는 통화 기호와 숫자를 특별히 처리할 수 있습니다.

**재학습 옵션**: 극단적인 경우, 도메인 코퍼스로 토크나이저를 처음부터 재학습할 수 있습니다. tokenizers 라이브러리의 Trainer를 사용하면 BPE, WordPiece 등의 알고리즘으로 새 토크나이저를 학습할 수 있습니다.

## 5.3. 어휘사전 파일 구조 및 예시

### 5.3.1. vocab.json 구조

vocab.json은 토크나이저의 어휘사전을 정의하는 파일입니다. 각 토큰을 고유한 정수 ID에 매핑하는 딕셔너리(Dictionary) 구조로 되어 있습니다.

**기본 구조**:
```json
{
  "<s>": 0,
  "<pad>": 1,
  "</s>": 2,
  "<unk>": 3,
  "▁안": 4,
  "▁녕": 5,
  "▁하": 6,
  "▁세": 7,
  "▁요": 8,
  "▁인공": 9,
  "▁지능": 10,
  "▁딥": 11,
  "▁러닝": 12
}
```

**구조 설명**:

**스페셜 토큰**: 파일의 최상단에는 모델 동작에 필수적인 스페셜 토큰들이 위치합니다. `<s>`는 시퀀스 시작(BOS, Beginning of Sequence), `</s>`는 시퀀스 종료(EOS, End of Sequence), `<pad>`는 패딩, `<unk>`는 미등록 단어를 나타냅니다.

**서브워드 마커**: `▁` 기호(유니코드 U+2581)는 단어의 시작을 표시하는 센티피스(SentencePiece) 방식의 마커입니다. 예를 들어 "안녕하세요"는 [`▁안`, `▁녕`, `▁하`, `▁세`, `▁요`]로 토큰화됩니다.

**토큰 ID**: 각 토큰에 할당된 정수 값은 임베딩 레이어의 인덱스로 사용됩니다. 일반적으로 스페셜 토큰은 0~99 범위의 작은 ID를 할당받고, 일반 토큰은 100부터 시작합니다.

**도메인 특화 어휘 추가 예시**:
```json
{
  "<s>": 0,
  "<pad>": 1,
  "</s>": 2,
  "<unk>": 3,
  "▁인공지능": 100,
  "▁딥러닝": 101,
  "▁트랜스포머": 102,
  "▁어텐션": 103,
  "▁임베딩": 104,
  "▁파인튜닝": 105,
  "▁토크나이저": 106,
  "▁데이터셋": 107,
  "▁하이퍼파라미터": 108,
  "▁체크포인트": 109,
  "▁코로나19": 110,
  "▁COVID-19": 111,
  "▁백신": 112,
  "▁블록체인": 113,
  "▁NFT": 114,
  "▁메타버스": 115
}
```

**어휘 크기 고려사항**: KoBART의 기본 어휘 크기는 약 30,000개입니다. 도메인 특화 토큰을 추가할 때는 일반적으로 100~1,000개 정도가 적절합니다. 너무 많은 토큰을 추가하면 임베딩 레이어의 파라미터 수가 증가하여 메모리 사용량이 늘어납니다.

**토큰 선정 기준**: 도메인 텍스트에서 높은 빈도로 등장하면서, 기존 토크나이저로는 여러 서브워드로 분리되는 용어들을 우선 선정합니다. 복합명사, 전문용어, 고유명사 등이 좋은 후보입니다.

### 5.3.2. tokenizer_config.json 구조

tokenizer_config.json은 토크나이저의 동작 방식과 설정을 정의하는 파일입니다. 모델 로딩 시 자동으로 읽혀져 토크나이저의 동작을 제어합니다.

**기본 구조**:
```json
{
  "add_prefix_space": false,
  "bos_token": "<s>",
  "eos_token": "</s>",
  "pad_token": "<pad>",
  "unk_token": "<unk>",
  "mask_token": "<mask>",
  "model_max_length": 1024,
  "name_or_path": "gogamza/kobart-base-v2",
  "special_tokens_map_file": null,
  "tokenizer_class": "PreTrainedTokenizerFast",
  "vocab_size": 30000,
  "do_lower_case": false,
  "strip_accents": null,
  "keep_accents": true
}
```

**주요 필드 설명**:

**스페셜 토큰 정의**: `bos_token`, `eos_token`, `pad_token`, `unk_token`, `mask_token`은 각각 특수 목적으로 사용되는 토큰을 지정합니다. 이들은 vocab.json에 정의된 토큰 중 하나여야 합니다.

**모델 최대 길이**: `model_max_length`는 토크나이저가 처리할 수 있는 최대 시퀀스 길이를 정의합니다. KoBART는 기본적으로 1024 토큰까지 처리할 수 있습니다. 이 값보다 긴 입력은 자동으로 잘립니다(truncation).

**토크나이저 클래스**: `tokenizer_class`는 사용할 토크나이저 구현체를 지정합니다. PreTrainedTokenizerFast는 Rust 기반의 고속 토크나이저이며, PreTrainedTokenizer는 Python 기반입니다.

**정규화 옵션**: `do_lower_case`는 입력을 소문자로 변환할지 여부를, `strip_accents`는 액센트를 제거할지 여부를 결정합니다. 한국어의 경우 일반적으로 false로 설정합니다.

**도메인 특화 설정 예시**:
```json
{
  "add_prefix_space": false,
  "bos_token": "<s>",
  "eos_token": "</s>",
  "pad_token": "<pad>",
  "unk_token": "<unk>",
  "mask_token": "<mask>",
  "sep_token": "</s>",
  "cls_token": "<s>",
  "model_max_length": 1024,
  "name_or_path": "custom-kobart-news",
  "tokenizer_class": "PreTrainedTokenizerFast",
  "vocab_size": 31000,
  "do_lower_case": false,
  "strip_accents": null,
  "keep_accents": true,
  "clean_up_tokenization_spaces": true,
  "split_special_tokens": false,
  "additional_special_tokens": [
    "[HEADLINE]",
    "[ARTICLE]",
    "[SUMMARY]",
    "[CATEGORY]"
  ],
  "add_bos_token": true,
  "add_eos_token": true
}
```

**추가 필드 설명**:

**추가 스페셜 토큰**: `additional_special_tokens`는 도메인별로 필요한 커스텀 마커 토큰을 정의합니다. 뉴스 요약 태스크에서는 기사의 헤드라인, 본문, 요약문 등을 구분하는 토큰을 추가할 수 있습니다.

**토큰 자동 추가**: `add_bos_token`과 `add_eos_token`은 인코딩 시 자동으로 시작/종료 토큰을 추가할지 여부를 결정합니다. 대부분의 시퀀스-투-시퀀스 모델에서는 true로 설정합니다.

**스페셜 토큰 분리**: `split_special_tokens`는 스페셜 토큰 내부에 다른 토큰이 포함되어 있을 때 분리할지 여부를 결정합니다. 일반적으로 false로 설정하여 스페셜 토큰을 단일 단위로 처리합니다.

**토큰화 공백 정리**: `clean_up_tokenization_spaces`는 디코딩 시 불필요한 공백을 제거할지 여부를 결정합니다. 한국어의 경우 true로 설정하면 자연스러운 출력을 얻을 수 있습니다.

### 5.3.3. 어휘사전 파일 활용 방법

**기존 어휘사전 로드 및 확장**:

기존 KoBART 토크나이저를 로드한 후, 도메인 특화 토큰을 추가하는 방식입니다. 이는 모델의 사전학습된 지식을 유지하면서 새로운 어휘를 추가할 수 있는 가장 안전한 방법입니다.

토크나이저를 로드하면 자동으로 vocab.json과 tokenizer_config.json이 함께 로드됩니다. 새로운 토큰을 추가한 후에는 반드시 모델의 임베딩 레이어 크기도 함께 조정해야 합니다. 그렇지 않으면 새로 추가된 토큰 ID에 대응하는 임베딩이 존재하지 않아 오류가 발생합니다.

**커스텀 어휘사전 생성**:

완전히 새로운 도메인에 대해서는 처음부터 토크나이저를 학습할 수 있습니다. 이 방법은 도메인 코퍼스의 특성을 최대한 반영할 수 있지만, 사전학습된 모델과의 호환성이 떨어집니다. 일반적으로 BPE 또는 WordPiece 알고리즘을 사용하여 학습합니다.

학습 과정에서는 도메인 텍스트의 통계적 특성을 분석하여 최적의 서브워드 분할을 찾습니다. 어휘 크기(vocab_size)는 보통 10,000~50,000 사이로 설정하며, 크기가 클수록 세밀한 표현이 가능하지만 모델 크기가 증가합니다.

**어휘사전 버전 관리**:

프로덕션 환경에서는 어휘사전의 버전을 체계적으로 관리해야 합니다. vocab.json과 tokenizer_config.json을 체크포인트와 함께 저장하고, 버전 번호나 타임스탬프를 파일명에 포함시키는 것이 좋습니다. 예: `vocab_v1.0_20251019.json`

어휘사전이 변경되면 기존에 학습된 모델의 임베딩 레이어와 호환되지 않을 수 있으므로, 변경 사항을 문서화하고 마이그레이션(Migration) 전략을 수립해야 합니다.

**성능 최적화 고려사항**:

어휘 크기가 증가하면 임베딩 레이어의 파라미터 수가 증가하여 메모리 사용량과 학습 시간이 늘어납니다. 임베딩 차원(dimension)이 768이고 어휘 크기가 30,000인 경우, 임베딩 레이어만 약 90MB의 메모리를 사용합니다.

빈도가 낮은 토큰은 제거하는 것을 고려할 수 있습니다. 일반적으로 전체 코퍼스에서 5회 미만으로 등장하는 토큰은 `<unk>` 토큰으로 대체해도 성능에 큰 영향이 없습니다.

### 5.3.4. 실무 적용 시 주의사항

**인코딩/디코딩 일관성**: vocab.json과 tokenizer_config.json의 설정이 일치해야 합니다. 예를 들어, tokenizer_config.json에 정의된 `bos_token`이 vocab.json에 존재하지 않으면 오류가 발생합니다.

**유니코드 정규화**: 한국어는 조합형과 완성형 유니코드가 혼재할 수 있습니다. 예를 들어 "한글"은 [U+D55C U+AE00] 또는 [U+1112 U+1161 U+11AB U+1100 U+1173 U+11AF]로 표현될 수 있습니다. 토크나이저 학습 전에 NFC(Normalization Form Canonical Composition) 정규화를 적용하는 것이 좋습니다.

**대소문자 처리**: 영어와 한국어가 혼재된 텍스트의 경우, 대소문자 처리 정책을 명확히 해야 합니다. `do_lower_case=false`로 설정하면 "AI", "ai", "Ai"가 모두 다른 토큰으로 처리됩니다.

**백워드 컴패터빌리티(Backward Compatibility)**: 기존 모델과의 호환성을 유지하려면 기존 토큰의 ID를 변경하지 않고, 새로운 토큰만 어휘사전 끝에 추가해야 합니다. 기존 토큰의 ID가 변경되면 사전학습된 임베딩이 잘못 매핑됩니다.

## 6. KoBART 친화적 데이터셋 구조

### 6.1. 권장 폴더 및 파일 구조

KoBART 파인튜닝을 위한 표준 폴더 구조는 다음과 같습니다:
```
project_root/
├── data/
│   ├── raw/
│   │   └── news_corpus.txt
│   ├── processed/
│   │   ├── train.json
│   │   ├── validation.json
│   │   └── test.json
│   └── tokenizer/
│       ├── vocab.json
│       └── tokenizer_config.json
├── models/
│   ├── kobart_base/
│   │   ├── config.json
│   │   ├── pytorch_model.bin
│   │   └── tokenizer/
│   └── checkpoints/
│       ├── checkpoint-1000/
│       └── checkpoint-2000/
├── scripts/
│   ├── preprocess.py
│   ├── train.py
│   └── evaluate.py
├── configs/
│   └── training_args.yaml
└── outputs/
    ├── logs/
    └── predictions/
```

**data/raw**: 원본 뉴스 데이터를 저장합니다. 대용량 파일은 .txt, .csv, .parquet 포맷으로 저장합니다.

**data/processed**: 전처리된 데이터셋을 저장합니다. train, validation, test 스플릿으로 분리합니다.

**models/kobart_base**: 사전학습된 KoBART 체크포인트를 저장합니다. Hub에서 다운로드한 모델이나 커스터마이징한 토크나이저를 포함합니다.

**models/checkpoints**: 학습 중 생성되는 체크포인트를 저장합니다. 각 체크포인트는 모델 웨이트, 옵티마이저(Optimizer) 상태, 학습 진행 정보를 포함합니다.

**scripts**: 데이터 전처리, 학습, 평가를 위한 스크립트를 저장합니다.

**configs**: 하이퍼파라미터(Hyperparameters), 경로 등의 설정 파일을 저장합니다. YAML 또는 JSON 포맷을 사용합니다.

**outputs**: 로그, 예측 결과, 시각화 등을 저장합니다.

### 6.2. JSON 포맷 스키마

KoBART 요약 태스크를 위한 JSON 데이터셋 포맷:
```json
{
  "id": "news_001",
  "document": "전체 뉴스 기사 본문이 여기에 들어갑니다. 여러 문단으로 구성될 수 있으며...",
  "summary": "기사의 요약문이 여기에 들어갑니다.",
  "metadata": {
    "category": "정치",
    "date": "2025-10-19",
    "source": "뉴스통신사",
    "length": 1542
  }
}
```

**필수 필드**:
- **id**: 각 샘플의 고유 식별자(아이덴티파이어, Identifier)
- **document**: 모델 입력이 되는 원문 텍스트
- **summary**: 타겟(Target) 요약문

**선택 필드**:
- **metadata**: 카테고리, 날짜, 출처 등 추가 정보. 필터링이나 분석에 활용

**배치 형태**: JSONL(JSON Lines) 포맷을 사용하면 대용량 데이터를 효율적으로 처리할 수 있습니다. 각 라인이 하나의 JSON 객체가 됩니다:
```jsonl
{"id": "news_001", "document": "...", "summary": "..."}
{"id": "news_002", "document": "...", "summary": "..."}
```

**중첩 구조**: 멀티모달(Multimodal) 데이터나 복잡한 메타데이터의 경우 중첩된 JSON을 사용할 수 있습니다:
```json
{
  "id": "news_001",
  "content": {
    "title": "기사 제목",
    "body": "기사 본문",
    "sections": ["섹션1", "섹션2"]
  },
  "target": {
    "abstractive": "생성 요약",
    "extractive": ["추출 문장1", "추출 문장2"]
  }
}
```

### 6.3. 데이터 로딩 클래스 및 라이브러리

**Hugging Face Datasets**: 가장 권장되는 방법입니다. load_dataset 함수로 다양한 포맷을 로드할 수 있습니다.

JSON 파일을 로드할 때는 다음과 같이 사용합니다. data_files 파라미터를 통해 각 스플릿별 파일을 지정할 수 있으며, JSONL 파일도 동일한 방식으로 로드됩니다. 로컬 폴더 구조를 사용하는 경우, 커스텀 데이터셋 스크립트를 경로로 지정할 수 있습니다.

**데이터셋 스크립트**: 복잡한 데이터 로딩 로직이 필요하다면 커스텀 데이터셋 스크립트를 작성합니다. datasets.GeneratorBasedBuilder를 상속받아 _generate_examples 메서드를 구현합니다.

데이터셋 스크립트는 _info 메서드에서 데이터 구조를 정의하고, _split_generators에서 각 스플릿의 파일 경로를 지정하며, _generate_examples에서 실제 데이터를 yield 형태로 반환합니다. 이 방식은 데이터 로딩 과정을 완전히 제어할 수 있으며, 특수한 전처리나 필터링 로직을 포함할 수 있습니다.

**PyTorch Dataset**: 더 세밀한 제어가 필요하다면 torch.utils.data.Dataset을 직접 구현합니다. __len__과 __getitem__ 메서드를 구현하면 됩니다. 이 방식은 전처리 파이프라인을 완전히 커스터마이징할 수 있습니다.

PyTorch Dataset 방식은 Hugging Face Datasets보다 저수준 제어가 가능하지만, 캐싱, 멀티프로세싱 등의 최적화는 직접 구현해야 합니다. DataLoader와 함께 사용하여 배치 처리와 셔플링을 수행합니다.

**데이터 전처리 파이프라인**: 로드된 데이터셋은 map 함수를 통해 전처리됩니다. 토큰화, 최대 길이 자르기(truncation), 패딩 등이 이 단계에서 수행됩니다. batched=True 옵션으로 배치 단위 처리가 가능하며, num_proc 파라미터로 멀티프로세싱을 활성화할 수 있습니다.

**데이터 콜레이터(Data Collator)**: DataCollatorForSeq2Seq는 배치 내에서 동적 패딩을 수행합니다. 이는 고정 길이 패딩보다 메모리 효율적입니다. 레이블(Labels)의 패딩 토큰은 -100으로 설정되어 손실 계산에서 무시됩니다.

**스트리밍 모드**: 1TB와 같은 대용량 데이터의 경우 streaming=True 옵션을 사용합니다. 이는 데이터를 메모리에 전체 로드하지 않고 필요할 때마다 읽어옵니다. IterableDataset 형태로 반환되며, 일반 Dataset과는 다른 API를 제공합니다.

**캐싱 전략**: Datasets 라이브러리는 전처리 결과를 자동으로 캐싱합니다. 동일한 전처리 코드를 다시 실행하면 캐시된 결과를 사용하여 시간을 절약합니다. 캐시는 ~/.cache/huggingface/datasets 경로에 저장되며, load_from_cache_file 파라미터로 제어할 수 있습니다.

---

## 용어 목록

| 용어 | 설명 |
|------|------|
| API (Application Programming Interface) | 소프트웨어 간 상호작용을 위한 인터페이스 |
| Abstraction | 복잡한 시스템을 단순화하여 표현하는 방법 |
| Attention Mask | 패딩 토큰과 실제 토큰을 구분하는 마스크 |
| Auto-configuration | 모델 이름을 기반으로 자동으로 설정을 결정하는 기능 |
| Autoregressive | 이전 출력을 기반으로 다음 출력을 생성하는 방식 |
| Batch | 모델에 동시에 입력되는 데이터 묶음 |
| BPE (Byte Pair Encoding) | 서브워드 토큰화 알고리즘 |
| Binary | 0과 1로 구성된 이진 데이터 |
| Bridge | 서로 다른 시스템을 연결하는 중간 역할 |
| Checkpoint | 학습 중 저장된 모델 상태 |
| Corpus | 자연어 처리를 위한 대규모 텍스트 모음 |
| Context | 문맥 정보 |
| Data Collator | 배치 생성 시 데이터를 정리하는 함수 |
| Democratization | 기술의 접근성을 높여 누구나 사용할 수 있게 만드는 것 |
| Denoising | 노이즈가 추가된 데이터에서 원본을 복원하는 기법 |
| Downstream | 사전학습 후 수행되는 구체적인 태스크 |
| Ecosystem | 상호 연결된 소프트웨어 및 서비스의 생태계 |
| Embeddings | 텍스트를 벡터로 표현한 것 |
| Encoder-Decoder | 입력을 인코딩하고 출력을 디코딩하는 구조 |
| Entity | 개체명 (사람, 장소, 조직 등) |
| Fine-tuning | 사전학습된 모델을 특정 태스크에 맞게 재학습 |
| Gradient Accumulation | 여러 배치의 그래디언트를 누적하는 기법 |
| Gradient Checkpointing | 메모리 절약을 위해 중간 활성화 값을 재계산하는 기법 |
| Head | 모델의 출력 레이어 |
| Hierarchy | 계층 구조 |
| Hyperparameters | 학습 전에 설정하는 파라미터 |
| Identifier | 고유 식별자 |
| Inference | 학습된 모델로 예측을 수행하는 과정 |
| Interactive | 사용자와 상호작용하는 |
| KoBART (Korean BART) | 한국어 특화 BART 모델 |
| Leaderboard | 모델 성능 순위표 |
| Limitations | 제한사항 |
| Logits | 모델의 원시 출력값 (활성화 함수 적용 전) |
| LoRA (Low-Rank Adaptation) | 파라미터 효율적 파인튜닝 기법 |
| Loss Function | 손실 함수 |
| Marker | 특정 의미를 표시하는 토큰 |
| Metrics | 모델 성능을 측정하는 지표 |
| Mixed Precision | 16비트와 32비트 부동소수점을 혼합 사용 |
| Model Card | 모델의 정보를 담은 문서 |
| Multimodal | 여러 종류의 데이터 (텍스트, 이미지 등)를 처리 |
| NLP (Natural Language Processing) | 자연어 처리 |
| Normalization | 데이터를 표준화하는 과정 |
| OOV (Out-of-Vocabulary) | 어휘사전에 없는 단어 |
| Optimizer | 최적화 알고리즘 |
| PMI (Pointwise Mutual Information) | 두 사건의 연관성을 측정하는 통계 지표 |
| Parallelization | 병렬 처리 |
| Parameter-Efficient Fine-Tuning (PEFT) | 적은 파라미터만 학습하는 파인튜닝 기법 |
| Pipeline | 전처리부터 후처리까지의 일련의 과정 |
| Pre-trained | 사전학습된 |
| Prefix Tuning | 입력 앞에 학습 가능한 프리픽스를 추가하는 기법 |
| Reflection | 프로그램이 자신의 구조를 검사하고 수정하는 기능 |
| Reproducibility | 재현 가능성 |
| Schema | 데이터 구조 정의 |
| Segment | 텍스트의 구분된 부분 |
| Sequence | 순서가 있는 데이터 열 |
| Sequence-to-Sequence | 시퀀스를 입력받아 시퀀스를 출력하는 모델 |
| Serverless | 서버 관리 없이 실행되는 방식 |
| Snippet | 코드 조각 |
| Special Tokens | 특수 토큰 ([CLS], [SEP] 등) |
| Split | 데이터셋의 분할 (train, validation, test) |
| Stack | 여러 레이어가 쌓인 구조 |
| Subword | 단어보다 작은 단위의 토큰 |
| Target | 모델이 예측해야 하는 정답 |
| Task-Agnostic | 특정 태스크에 종속되지 않는 |
| TF-IDF (Term Frequency-Inverse Document Frequency) | 문서 내 단어의 중요도를 측정하는 통계 지표 |
| Tokenization | 텍스트를 토큰으로 분리하는 과정 |
| Tokenizer | 토큰화를 수행하는 도구 |
| Unified Interface | 통합된 인터페이스 |
| Vocabulary | 어휘사전 |
| Weights | 모델의 가중치 파라미터 |
| WordPiece | 서브워드 토큰화 알고리즘의 한 종류 |

---

## 참고사항

이 문서는 Hugging Face Transformers 라이브러리와 KoBART를 활용한 문서 요약 모델 구축을 위한 가이드입니다..<br/>
실제 구현 시에는 공식 문서(https://huggingface.co/docs/transformers)를 참조하시기 바랍니다.

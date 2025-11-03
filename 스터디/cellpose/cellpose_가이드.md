---
layout: default
title: "Cellpose 모델 테스트 가이드"
description: "Jupyter Notebook / Google Colab에서 Cellpose 세포 세그멘테이션 모델의 핵심 기능 테스트"
date: 2025-11-03
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# Cellpose 모델 테스트 가이드

## 목차

1. [개요](#1-개요)
   - 1.1. [Cellpose란?](#11-cellpose란)
   - 1.2. [실행 환경](#12-실행-환경)
   - 1.3. [⚠️ 버전별 주요 차이점](#13-️-버전별-주요-차이점)
2. [환경 설정](#2-환경-설정)
   - 2.1. [라이브러리 설치](#21-라이브러리-설치)
   - 2.2. [GPU 설정 (Colab)](#22-gpu-설정-colab)
3. [핵심 API](#3-핵심-api)
   - 3.1. [모델 초기화](#31-모델-초기화)
   - 3.2. [주요 파라미터](#32-주요-파라미터)
   - 3.3. [입력/출력 형식](#33-입력출력-형식)
4. [MVP 테스트 코드](#4-mvp-테스트-코드)
   - 4.1. [Hugging Face 방식 (권장)](#41-hugging-face-방식-권장)
   - 4.2. [기본 Cellpose 방식](#42-기본-cellpose-방식)
   - 4.3. [실제 이미지 사용 방법](#43-실제-이미지-사용-방법)
5. [용어 목록](#5-용어-목록-glossary)

---

[![Cellpose 유튜브 가이드](https://img.youtube.com/vi/UtfDm3TsqpY/0.jpg)](https://youtu.be/UtfDm3TsqpY)

## 1. 개요

### 1.1. Cellpose란?

**Cellpose**는 생물학적 이미지에서 세포를 자동으로 분할(segmentation)하는 딥러닝 모델입니다.

- **특징**: 다양한 세포 형태에 범용적으로 적용 가능
- **방법**: Flow 기반 세그멘테이션 알고리즘 사용
- **용도**: 현미경 이미지에서 개별 세포 경계 탐지

### 1.2. 실행 환경

| 환경 | 지원 여부 | 비고 |
|------|----------|------|
| Jupyter Notebook | ✅ | 로컬/서버 |
| Google Colab | ✅ | GPU 무료 제공 |
| Python 버전 | 3.8+ | 권장 3.9 이상 |

### 1.3. ⚠️ 버전별 주요 차이점

**Cellpose v4.0+에서 API가 변경되었습니다.**

| 항목 | v3.x (구버전) | v4.0+ (최신) |
|------|--------------|--------------|
| **모델 클래스** | `models.Cellpose()` | `models.CellposeModel()` |
| **반환값 개수** | 4개 | 3개 |
| **channels 파라미터** | 필수 지정 | Deprecated (자동 처리) |
| **diams 반환값** | ✅ 제공됨 | ❌ 제거됨 |

**v4.0+ 사용 예시**:
```python
# ✅ v4.0+ 올바른 사용법
from cellpose import models
model = models.CellposeModel(model_type='cyto2')
masks, flows, styles = model.eval(img, diameter=30)
num_cells = masks.max()

# ❌ v3.x 구버전 방식 (오류 발생)
# model = models.Cellpose(gpu=True, model_type='cyto2')
# masks, flows, styles, diams = model.eval(img, diameter=30, channels=[0,0])
```

---

## 2. 환경 설정

### 2.1. 라이브러리 설치

```bash
# Cellpose 공식 라이브러리
pip install cellpose

# Hugging Face Hub (모델 다운로드용)
pip install huggingface_hub

# 추가 의존성 (이미지 처리)
pip install opencv-python-headless pillow matplotlib numpy
```

**Colab 실행 시**:
```python
!pip install -q cellpose huggingface_hub opencv-python-headless
```

### 2.2. GPU 설정 (Colab)

Google Colab에서 GPU 활성화:
1. 메뉴: `런타임` → `런타임 유형 변경`
2. 하드웨어 가속기: `T4 GPU` 선택

GPU 확인:
```python
import torch
print(f"GPU 사용 가능: {torch.cuda.is_available()}")
```

---

## 3. 핵심 API

### 3.1. 모델 초기화

**⚠️ v4.0+ API 사용 (권장)**:
```python
from cellpose import models

# 모델 객체 생성 (v4.0+)
model = models.CellposeModel(model_type='cyto2')
# GPU는 자동 감지됨
```

**v3.x 구버전** (참고용):
```python
# 구버전 API (v3.x) - 더 이상 지원되지 않음
model = models.Cellpose(
    gpu=True,              # GPU 사용 여부
    model_type='cyto2'     # 모델 타입
)
```

### 3.2. 주요 파라미터

#### `models.Cellpose()` 파라미터

| 파라미터 | 타입 | 설명 | 기본값 |
|---------|------|------|--------|
| `gpu` | bool | GPU 가속 사용 | `False` |
| `model_type` | str | 사전학습 모델 선택 | `'cyto2'` |

**모델 타입 옵션**:
- `'cyto2'`: 세포질(cytoplasm) 세그멘테이션 (범용)
- `'nuclei'`: 세포핵(nucleus) 전용
- `'cyto'`: 구버전 세포질 모델

#### `model.eval()` 파라미터

| 파라미터 | 타입 | 설명 |
|---------|------|------|
| `x` | ndarray | 입력 이미지 (H×W 또는 H×W×C) |
| `diameter` | float | 예상 세포 직경 (픽셀 단위) |
| `channels` | list | 채널 구성 `[cyto_channel, nucleus_channel]` |
| `flow_threshold` | float | Flow 임계값 (0.0~1.0) |

### 3.3. 입력/출력 형식

#### 입력 형식

```python
import numpy as np

# 형식 1: Grayscale (H × W)
img_gray = np.array([[...]])  # shape: (512, 512)

# 형식 2: RGB (H × W × 3)
img_rgb = np.array([[...]])   # shape: (512, 512, 3)

# 형식 3: 다중 채널 (H × W × C)
img_multi = np.array([[...]])  # shape: (512, 512, 2)
```

#### 출력 형식

**v4.0+ (최신)**:
```python
# 3개 값만 반환 (diams 제거됨)
masks, flows, styles = model.eval(img, diameter=30)
```

| 반환값 | 타입 | 설명 |
|--------|------|------|
| `masks` | ndarray | 세그멘테이션 마스크 (0=배경, 1,2,3...=각 세포) |
| `flows` | list | Flow 벡터장 정보 |
| `styles` | ndarray | 스타일 벡터 (잠재 표현) |

**v3.x (구버전)**:
```python
# 4개 값 반환 (diams 포함)
masks, flows, styles, diams = model.eval(img, diameter=30, channels=[0,0])
```

**마스크 해석**:
```python
num_cells = masks.max()  # 탐지된 세포 개수
print(f"총 {num_cells}개 세포 탐지됨")
```

---

## 4. MVP 테스트 코드

### 4.1. Hugging Face 방식 (권장)

**v4.0+ 코드**:
```python
from cellpose import models
import numpy as np

# 1. 모델 초기화 (v4.0+)
model = models.CellposeModel(model_type='cyto2')

# 2. 더미 이미지 생성 (실제 이미지 대신 기능 테스트용)
dummy_img = np.random.rand(256, 256) * 255  # 256×256 랜덤 이미지

# 3. 세그멘테이션 실행 (3개 값만 반환)
masks, flows, styles = model.eval(dummy_img, diameter=30)

# 4. 결과 확인
print(f"✅ 테스트 완료 | 탐지된 객체: {masks.max()}개")
```

**실행 결과 예시**:
```
✅ 테스트 완료 | 탐지된 객체: 12개
```

---

### 4.2. 기본 Cellpose 방식

**v4.0+ 코드**:
```python
from cellpose import models, io
import numpy as np

# 모델 로드 (v4.0+)
model = models.CellposeModel(model_type='nuclei')  # 핵 전용 모델

# 테스트 이미지 (랜덤 노이즈)
test_img = (np.random.rand(512, 512) * 255).astype(np.uint8)

# 추론 실행 (3개 값 반환)
masks, _, _ = model.eval(test_img, diameter=None)  # 자동 직경 추정

# 결과 출력
print(f"Segmentation 완료: {masks.shape} | 세포 수: {masks.max()}")
```

---

### 4.3. 실제 이미지 사용 방법

#### 방법 1: 로컬 파일 로드

```python
import cv2

# 이미지 읽기 (Grayscale)
img = cv2.imread('cell_image.png', cv2.IMREAD_GRAYSCALE)

# Cellpose 실행 (v4.0+)
masks, flows, styles = model.eval(img, diameter=30)
```

#### 방법 2: URL에서 다운로드 (Colab)

```python
from PIL import Image
import requests
from io import BytesIO
import numpy as np

# URL에서 이미지 가져오기
url = "https://example.com/sample_cells.jpg"
response = requests.get(url)
img = Image.open(BytesIO(response.content)).convert('L')  # Grayscale 변환
img_array = np.array(img)

# 모델 실행 (v4.0+)
masks, _, _ = model.eval(img_array, diameter=25)
```

#### 방법 3: Colab 파일 업로드

```python
from google.colab import files
import cv2

# 파일 업로드 위젯 표시
uploaded = files.upload()

# 업로드된 파일 읽기
filename = list(uploaded.keys())[0]
img = cv2.imread(filename, 0)  # 0 = Grayscale

# 세그멘테이션 (v4.0+)
masks, _, _ = model.eval(img, diameter=30)
```

#### 방법 4: Cellpose 샘플 데이터

```python
from cellpose import io

# Cellpose 공식 샘플 다운로드
img = io.imread('https://www.cellpose.org/static/images/img02.png')

# 실행 (v4.0+)
masks, _, _ = model.eval(img, diameter=30)
```

---

### 4.4. 결과 시각화 (선택)

```python
import matplotlib.pyplot as plt
from cellpose import plot

# 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 원본 이미지
axes[0].imshow(dummy_img, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

# 세그멘테이션 마스크
axes[1].imshow(masks, cmap='tab20')
axes[1].set_title(f'Segmentation ({masks.max()} cells)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
```

**마스크 오버레이**:
```python
# Cellpose 내장 시각화 함수
img_with_masks = plot.mask_overlay(dummy_img, masks)
plt.imshow(img_with_masks)
plt.title('Overlay View')
plt.show()
```

---

## 5. 용어 목록 (Glossary)

| 용어 | 영문 | 설명 |
|------|------|------|
| 세그멘테이션 | Segmentation | 이미지를 의미 있는 영역으로 분할하는 기술 |
| 세포질 | Cytoplasm | 세포막 내부의 세포핵을 제외한 영역 |
| 세포핵 | Nucleus | 유전 물질을 포함하는 세포 중심부 |
| 마스크 | Mask | 각 픽셀이 어떤 객체에 속하는지 나타내는 레이블 맵 |
| 플로우 | Flow | Cellpose의 핵심 알고리즘으로 픽셀이 세포 중심으로 흐르는 방향 벡터 |
| 다이어미터 | Diameter | 세포의 평균 직경 (픽셀 단위) |
| 채널 | Channel | 이미지의 색상 또는 형광 채널 구성 |
| GPU 가속 | GPU Acceleration | 그래픽 처리 장치를 사용한 연산 속도 향상 |
| 임계값 | Threshold | 세그멘테이션 결정 경계값 |
| 사전학습 모델 | Pre-trained Model | 대규모 데이터셋으로 미리 학습된 모델 |

---

## 부록: 트러블슈팅

### A. 버전 관련 오류 ⭐

**증상 1**: `AttributeError: module 'cellpose.models' has no attribute 'Cellpose'`

**원인**: v4.0+에서 `models.Cellpose` 클래스가 제거됨

**해결**:
```python
# ❌ 오류 발생
# model = models.Cellpose(gpu=True, model_type='cyto2')

# ✅ 올바른 방법 (v4.0+)
model = models.CellposeModel(model_type='cyto2')
```

---

**증상 2**: `ValueError: not enough values to unpack (expected 4, got 3)`

**원인**: v4.0+에서 `eval()` 반환값이 3개로 변경됨 (diams 제거)

**해결**:
```python
# ❌ 오류 발생 (v3.x 방식)
# masks, flows, styles, diams = model.eval(img, diameter=30)

# ✅ 올바른 방법 (v4.0+)
masks, flows, styles = model.eval(img, diameter=30)
```

---

**증상 3**: `Warning: channels deprecated in v4.0.1+`

**원인**: channels 파라미터가 deprecated됨 (자동 처리)

**해결**:
```python
# ⚠️ 경고 발생하지만 작동함
masks, _, _ = model.eval(img, diameter=30, channels=[0,0])

# ✅ 권장 방법 (channels 생략)
masks, _, _ = model.eval(img, diameter=30)
```

---

### B. GPU 관련 오류

**증상**: `RuntimeError: CUDA out of memory`

**해결**:
```python
# v4.0+는 GPU 자동 감지 (CPU 강제 사용 불필요)
model = models.CellposeModel(model_type='cyto2')
```

### C. 직경 자동 추정

```python
# diameter=None으로 자동 추정 활성화
masks, _, _ = model.eval(img, diameter=None)
print(f"탐지된 세포: {masks.max()}개")
```

---

## 참고 자료

- **공식 문서**: [https://cellpose.readthedocs.io](https://cellpose.readthedocs.io)
- **GitHub**: [https://github.com/MouseLand/cellpose](https://github.com/MouseLand/cellpose)
- **논문**: Stringer, C., et al. (2021). *Cellpose: a generalist algorithm for cellular segmentation*. Nature Methods.
- **Hugging Face**: [https://huggingface.co/models?search=cellpose](https://huggingface.co/models?search=cellpose)



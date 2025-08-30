---
layout: default
title: "YOLO 학습 데이터"
description: "YOLO 학습 데이터의 모든 것: 형식, 구조, COCO와의 비교, 그리고 데이터 증강"
date: 2025-08-30
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# YOLO 학습 데이터 가이드: 형식, 구조, COCO와의 비교, 그리고 데이터 증강

## 목차
1. [YOLO 학습 데이터의 기본 이해](#1-yolo-학습-데이터의-기본-이해)<br/>
2. [YOLO 학습 데이터셋의 표준 구조](#2-yolo-학습-데이터셋의-표준-구조)<br/>
3. [COCO 데이터셋과의 비교 및 주의점](#3-coco-데이터셋과의-비교-및-주의점)<br/>
4. [데이터 증강 및 검증 도구](#4-데이터-증강-및-검증-도구)<br/>
5. [실무 팁과 주의사항](#5-실무-팁과-주의사항)<br/>
6. [용어 목록](#6-용어-목록)<br/>

---

## 1. YOLO 학습 데이터의 기본 이해

YOLO(You Only Look Once)는 객체 감지(Object Detection, 오브젝트 디텍션) 모델로, 학습을 위해 이미지와 그에 해당하는 레이블 파일이 필요합니다. 이 두 파일은 이름이 같고 확장자만 다르게 구성됩니다.

### 1.1. 이미지 파일

학습에 사용되는 이미지는 `.jpg`, `.png`, `.jpeg` 등 일반적인 이미지 형식입니다. 권장 해상도는 640x640 픽셀이지만, 다양한 크기의 이미지를 사용할 수 있습니다.

### 1.2. 레이블 파일 포맷

각 이미지에 해당하는 레이블 파일은 `.txt` 확장자를 가집니다. 이 파일에는 이미지 내의 각 객체에 대한 정보가 한 줄씩 기록되어 있습니다.

```
<class_id> <x_center> <y_center> <width> <height>
```

- **`<class_id>`**: 객체의 클래스 ID(0부터 시작하는 정수)
- **`<x_center>`, `<y_center>`**: 바운딩 박스의 중심 x, y 좌표. 이미지 너비/높이에 대한 **정규화된(normalized)** 값(0.0 ~ 1.0)
- **`<width>`, `<height>`**: 바운딩 박스의 너비와 높이. 이 역시 이미지 너비/높이에 대한 **정규화된** 값(0.0 ~ 1.0)

### 1.3. 예시

이미지 크기가 800x600인 경우, 픽셀 좌표 (100, 50)에서 시작하여 너비 200, 높이 150인 바운딩 박스는 다음과 같이 변환됩니다:

```
0 0.25 0.208 0.25 0.25
```

---

## 2. YOLO 학습 데이터셋의 표준 구조

YOLO 모델 학습은 단순히 이미지와 레이블 파일만으로는 시작할 수 없습니다. 이들을 체계적으로 정리한 폴더 구조와 함께 모델이 학습에 필요한 정보를 담은 설정 파일이 필수적입니다.

### 2.1. 폴더 구조

데이터셋은 일반적으로 다음과 같은 디렉터리 구조를 가집니다:

```
dataset/
├── images/
│   ├── train/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   ├── val/
│   │   ├── val_image1.jpg
│   │   └── ...
│   └── test/ (선택적)
└── labels/
    ├── train/
    │   ├── image1.txt
    │   ├── image2.txt
    │   └── ...
    ├── val/
    │   ├── val_image1.txt
    │   └── ...
    └── test/ (선택적)
```

- **`images/`**: 학습(train), 검증(val), 테스트(test) 이미지가 저장됩니다
- **`labels/`**: `images/` 폴더와 동일한 구조로, 각 이미지에 해당하는 `.txt` 레이블 파일이 저장됩니다

### 2.2. 필요한 설정 파일: `data.yaml`

모델이 데이터셋의 위치와 클래스 정보를 알 수 있도록 **`data.yaml`** 파일을 구성해야 합니다:

```yaml
# 데이터셋 경로
train: ../dataset/images/train/
val: ../dataset/images/val/
test: ../dataset/images/test/  # 선택적

# 클래스 설정
nc: 3  # number of classes
names: ['person', 'car', 'bicycle']

# 추가 설정 (선택적)
roboflow:
  workspace: your-workspace
  project: your-project
  version: 1
```

- **`train`**: 학습 이미지 경로
- **`val`**: 검증 이미지 경로  
- **`nc`**: 클래스 개수(number of classes)
- **`names`**: 클래스 이름 목록

---

## 3. COCO 데이터셋과의 비교 및 주의점

COCO 데이터셋은 YOLO 학습에 자주 사용되는 벤치마크 데이터셋입니다. 하지만 YOLO와 COCO는 어노테이션(Annotation, 애너테이션) 포맷이 달라 변환 작업이 필수적입니다.

### 3.1. 어노테이션 포맷 비교

| 특징 | YOLO | COCO |
|:---:|:---:|:---:|
| **포맷** | 이미지당 `.txt` 파일 | 전체 데이터셋을 담은 `.json` 파일 |
| **좌표 시스템** | **정규화된** 중심 좌표 | **픽셀 기반**의 좌상단 좌표 |
| **포맷 예시** | `<class_id> <x_center> <y_center> <width> <height>` | `[x, y, width, height]` (픽셀 단위) |
| **클래스 ID** | 0부터 시작 | 1부터 시작 |
| **장점** | 간결하고 직관적, 빠른 데이터 로딩 | 풍부한 메타데이터 포함, 다양한 작업에 활용 가능 |
| **단점** | 단순한 정보만 저장 가능 | 복잡하고 파싱(Parsing, 파싱)에 별도 라이브러리 필요 |

### 3.2. COCO 데이터셋을 YOLO 학습에 사용할 때 주의사항

#### 3.2.1. 포맷 변환

**COCO의 JSON 포맷을 YOLO의 TXT 포맷으로 반드시 변환해야 합니다.** 이 과정에서 픽셀 기반의 좌상단 좌표를 YOLO가 요구하는 **정규화된 중심 좌표 및 너비/높이**로 변환해야 합니다.


**중심 좌표 계산:**

---

$$x_{center} = \frac{\text{바운딩박스 왼쪽 x좌표}(x_{pixel}) + \frac{\text{바운딩박스 너비}(width_{pixel})}{2}}{\text{이미지 전체 너비}(image_{width})}$$

$$y_{center} = \frac{\text{바운딩박스 위쪽 y좌표}(y_{pixel}) + \frac{\text{바운딩박스 높이}(height_{pixel})}{2}}{\text{이미지 전체 높이}(image_{height})}$$

---

**정규화된 크기 계산:**

$$정규화된\ 너비(normalized_{width}) = \frac{너비_{픽셀}(width_{pixel})}{이미지_{너비}(image_{width})}$$

$$정규화된\ 높이(normalized_{height}) = \frac{높이_{픽셀}(height_{pixel})}{이미지_{높이}(image_{height})}$$

---

#### 3.2.2. 클래스 ID 재매핑(Remapping, 리매핑)

COCO 데이터셋의 클래스 ID는 1부터 시작하지만, YOLO는 0부터 시작하는 ID를 사용합니다. 변환 스크립트를 작성할 때 COCO의 클래스 ID를 YOLO가 인식할 수 있는 순서(0, 1, 2, ...)로 다시 매핑해야 합니다.

#### 3.2.3. 변환 도구 예시

```python
# COCO to YOLO 변환 (pycocotools 사용)
from pycocotools.coco import COCO
coco = COCO('annotations/instances_train2017.json')
# 변환 로직 구현...
```

---

## 4. 데이터 증강 및 검증 도구

### 4.1. 데이터 증강(Data Augmentation, 데이터 어그멘테이션)

데이터 증강은 제한된 학습 데이터로부터 더 많은 변형된 데이터를 생성하여 모델의 일반화(Generalization, 제너럴라이제이션) 성능을 향상시키는 기법입니다.

#### 4.1.1. Albumentations 라이브러리

가장 널리 사용되는 데이터 증강 라이브러리로, YOLO 형식의 바운딩 박스를 직접 지원합니다:

```python
import albumentations as A

transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.Rotate(limit=15, p=0.5)
], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

#### 4.1.2. 기타 증강 라이브러리들

```python
# imgaug 사용 예시
import imgaug.augmenters as iaa
seq = iaa.Sequential([iaa.Fliplr(0.5), iaa.GaussianBlur(sigma=(0.0, 1.0))])

# torchvision transforms 사용 예시  
import torchvision.transforms as transforms
transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ColorJitter()])
```

### 4.2. 데이터 검증(Validation, 밸리데이션) 도구

#### 4.2.1. 기본 검증 사항

데이터셋을 학습에 사용하기 전에 다음 사항들을 반드시 검증해야 합니다:

- 바운딩 박스 좌표가 [0,1] 범위 내에 있는지 확인
- 클래스 ID가 유효한 범위(0 ~ nc-1) 내에 있는지 확인  
- 이미지와 라벨 파일의 이름이 정확히 매칭되는지 확인
- 빈 라벨 파일이나 손상된 이미지 파일 검출

#### 4.2.2. 검증 코드 예시

```python
# 기본 검증 예시
import os, glob
def validate_yolo_dataset(image_dir, label_dir):
    for img_path in glob.glob(f"{image_dir}/*.jpg"):
        label_path = img_path.replace(image_dir, label_dir).replace('.jpg', '.txt')
        assert os.path.exists(label_path), f"라벨 파일이 없습니다: {label_path}"
```

---

## 5. 실무 팁과 주의사항

### 5.1. 성능 최적화 팁

- **이미지 크기**: 640x640이 일반적이지만, GPU 메모리에 따라 조절
- **배치 크기(Batch Size)**: GPU 메모리에 맞게 설정 (일반적으로 16, 32)
- **데이터셋 분할**: Train 70%, Validation 20%, Test 10% 권장

### 5.2. 흔한 실수와 해결방법

#### 5.2.1. 좌표계 혼동
- **문제**: COCO의 (x, y, w, h)를 YOLO의 (x_center, y_center, w, h)로 잘못 변환
- **해결**: 변환 공식을 정확히 적용하고 시각화로 검증

#### 5.2.2. 정규화 누락  
- **문제**: 픽셀 좌표를 그대로 사용
- **해결**: 모든 좌표값이 0.0 ~ 1.0 사이인지 확인

#### 5.2.3. 클래스 ID 불일치
- **문제**: COCO(1~80)와 YOLO(0~79)의 클래스 ID 차이 무시
- **해결**: 변환 시 클래스 ID에서 1을 빼거나 매핑 테이블 사용

### 5.3. 디버깅 도구

```python
# 바운딩 박스 시각화로 검증
import cv2
def visualize_yolo_labels(image_path, label_path):
    img = cv2.imread(image_path)
    h, w = img.shape[:2]
    with open(label_path, 'r') as f:
        for line in f:
            class_id, x_center, y_center, width, height = map(float, line.split())
            # YOLO -> 픽셀 좌표 변환 후 시각화
```

---

## 6. 용어 목록

- **어노테이션(Annotation)**: 이미지 내 객체의 위치와 종류를 표시하는 주석 작업 또는 그 결과
- **백프로파게이션(Backpropagation)**: 신경망의 가중치를 업데이트하기 위한 역전파 알고리즘
- **옵티마이저(Optimizer)**: 모델의 손실 함수를 최소화하는 데 사용되는 최적화 알고리즘
- **바운딩 박스(Bounding Box)**: 객체를 감싸는 직사각형 형태의 상자
- **정규화(Normalization)**: 데이터의 값을 0에서 1 사이와 같이 특정 범위로 조정하는 과정
- **파싱(Parsing)**: 데이터 파일에서 원하는 정보를 추출하여 구조화하는 작업
- **데이터 증강(Data Augmentation)**: 기존 데이터를 변형하여 학습 데이터의 양과 다양성을 늘리는 기법
- **일반화(Generalization)**: 모델이 학습하지 않은 새로운 데이터에 대해서도 좋은 성능을 보이는 능력
- **오브젝트 디텍션(Object Detection)**: 이미지에서 특정 객체의 위치와 종류를 동시에 찾아내는 컴퓨터 비전 작업
- **리매핑(Remapping)**: 기존 값들을 새로운 값 체계로 다시 매핑하는 과정
- **밸리데이션(Validation)**: 모델의 성능을 평가하고 검증하는 과정
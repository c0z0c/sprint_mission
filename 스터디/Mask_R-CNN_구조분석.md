---
layout: default
title: "Mask R-CNN 구조 분석"
description: "Mask R-CNN 구조 분석"
date: 2025-08-28
cache-control: no-cache
expires: 0
pragma: no-cache
author: "김명환"
---

# Mask R-CNN 구조 분석 및 구현 가이드

## 목차

1. [개요](#1-개요)<br/>
2. [핵심 구조 분석](#2-핵심-구조-분석)<br/>
   1. [nn.Sequential 상속 구조](#21-nnsequential-상속-구조)<br/>
   2. [self.roi_heads 작동 방식](#22-selfroiheads-작동-방식)<br/>
3. [get_maskrcnn_resnet50_fpn 분석](#3-getmaskrcnnresnet50fpn-분석)<br/>
4. [구현 시 필수 고려사항](#4-구현-시-필수-고려사항)<br/>
5. [데이터셋 준비](#5-데이터셋-준비)<br/>
6. [실무 체크리스트](#6-실무-체크리스트)<br/>
7. [추가 보완 내용](#7-추가-보완-내용)<br/>
8. [용어 정리](#8-용어-정리)<br/>

---

## 1. 개요

Mask R-CNN은 instance segmentation을 위한 딥러닝 모델로, Faster R-CNN을 기반으로 마스크 예측 분기(mask branch)를 추가한 구조입니다. PyTorch의 torchvision 라이브러리에서 제공하는 구현체를 중심으로 핵심 구조와 구현 방법을 분석합니다.

## 2. 핵심 구조 분석

### 2.1. nn.Sequential 상속 구조

#### 2.1.1. MaskRCNNHeads의 설계

MaskRCNNHeads 클래스는 nn.Sequential을 상속하여 마스크 헤드의 컨볼루션 레이어들을 순차적으로 구성합니다.

```python
class MaskRCNNHeads(nn.Sequential):
    def __init__(self, in_channels, layers, dilation):
        blocks = []
        next_feature = in_channels
        for layer_features in layers:
            blocks.append(
                Conv2dNormActivation(
                    next_feature, layer_features, 
                    kernel_size=3, stride=1, 
                    padding=dilation, dilation=dilation
                )
            )
            next_feature = layer_features
        super().__init__(*blocks)
```

#### 2.1.2. nn.Sequential 상속의 장점과 주의점

**장점:**
- **자동 Forward 처리**: `self[0](x) → self[1](x) → ...` 순차 실행
- **코드 간소화**: 커스텀 forward() 불필요
- **모듈 관리**: 인덱스로 레이어 접근 가능

**상속 시 주의점:**
```python
# ❌ 잘못된 방법 - 모듈이 등록되지 않음
class BadExample(nn.Sequential):
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 64, 3)  # Sequential에 등록 안됨
        self.conv2 = nn.Conv2d(64, 128, 3)

# ✅ 올바른 방법 - super().__init__()로 모듈 등록
class GoodExample(nn.Sequential):
    def __init__(self):
        super().__init__(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3)
        )

# ✅ 또는 OrderedDict 사용
class AlternativeExample(nn.Sequential):
    def __init__(self):
        super().__init__(OrderedDict([
            ('conv1', nn.Conv2d(3, 64, 3)),
            ('relu1', nn.ReLU()),
            ('conv2', nn.Conv2d(64, 128, 3))
        ]))
```

**커스텀 forward 구현 시 주의:**
```python
class CustomSequential(nn.Sequential):
    def forward(self, x):
        # Sequential의 기본 forward를 오버라이드
        for module in self:
            x = module(x)
            # 중간에 추가 처리 가능
            if isinstance(module, nn.ReLU):
                x = F.dropout(x, training=self.training)
        return x
```

#### 2.1.3. 버전 호환성 처리 보완

```python
def _load_from_state_dict(self, state_dict, prefix, local_metadata, 
                         strict, missing_keys, unexpected_keys, error_msgs):
    version = local_metadata.get("version", None)
    
    # 중요: 이는 사전학습 모델이 아닌 PyTorch 버전 간 호환성을 위함
    if version is None or version < 2:
        num_blocks = len([k for k in state_dict.keys() 
                         if k.startswith(f"{prefix}mask_fcn")])
        
        for i in range(num_blocks):
            for param_type in ["weight", "bias"]:
                old_key = f"{prefix}mask_fcn{i+1}.{param_type}"
                new_key = f"{prefix}{i}.0.{param_type}"
                if old_key in state_dict:
                    state_dict[new_key] = state_dict.pop(old_key)
    
    super()._load_from_state_dict(state_dict, prefix, local_metadata, 
                                  strict, missing_keys, unexpected_keys, error_msgs)
```

### 2.2. self.roi_heads 작동 방식 (보완)

#### 2.2.1. ROI (Region of Interest) 개념

ROI는 이미지에서 객체가 있을 것으로 예상되는 사각형 영역으로, Two-stage 객체 탐지의 핵심입니다.

**ROI의 생명주기:**
1. **RPN 단계**: 앵커 박스 기반으로 객체 위치 추정
2. **NMS 적용**: 중복 제거 후 상위 N개 선택
3. **ROI 생성**: `(batch_idx, x1, y1, x2, y2)` 형태로 정규화

#### 2.2.2. ROI Pooling vs ROI Align 상세 비교

**ROI Pooling의 문제점:**
```python
# 예시: ROI [0, 15.2, 15.2, 87.8, 87.8]을 7x7로 풀링
roi_width = 87.8 - 15.2 = 72.6
bin_size = 72.6 / 7 = 10.37...

# 양자화로 인한 정보 손실
bin_size_int = floor(10.37) = 10  # 소수점 버림
start_pos = floor(15.2) = 15      # 위치 정보 손실
```

**ROI Align의 해결책:**
```python
# 양자화 없이 정확한 샘플링
for i in range(7):
    for j in range(7):
        # 정확한 샘플링 위치 계산
        sample_x = 15.2 + (i + 0.5) * 10.37
        sample_y = 15.2 + (j + 0.5) * 10.37
        
        # Bilinear interpolation으로 값 계산
        value = bilinear_interpolate(feature_map, sample_x, sample_y)
```

#### 2.2.3. RoIHeads 클래스 구조 및 작동 원리

```python
class RoIHeads(nn.Module):
    def __init__(self):
        # Box Detection Branch
        self.box_roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3'], 7, 2)
        self.box_head = TwoMLPHead(256 * 7 * 7, 1024)
        self.box_predictor = FastRCNNPredictor(1024, num_classes)
        
        # Mask Segmentation Branch  
        self.mask_roi_pool = MultiScaleRoIAlign(['0', '1', '2', '3'], 14, 2)
        self.mask_head = MaskRCNNHeads(256, [256, 256, 256, 256], 1)
        self.mask_predictor = MaskRCNNPredictor(256, 256, num_classes)
```

**핵심 차이점:**
- **해상도**: Box(7×7) vs Mask(14×14) - 마스크가 더 정밀한 해상도 필요
- **용도**: Box는 분류+위치, Mask는 픽셀 단위 예측

#### 2.2.4. Forward Pass 상세 단계

**훈련 모드 (Training):**
```python
def forward(self, features, proposals, image_shapes, targets=None):
    if self.training and targets is None:
        raise ValueError("targets가 없으면 훈련 불가")
    
    # 1. 훈련용 샘플 선택 (512개 중 positive:negative = 1:3)
    proposals, labels, regression_targets = self.select_training_samples(
        proposals, targets)
    
    # 2. Box Branch
    box_features = self.box_roi_pool(features, proposals, image_shapes)
    box_features = self.box_head(box_features)  # [N, 1024]
    class_logits, box_regression = self.box_predictor(box_features)
    
    # 3. Mask Branch (positive samples만 사용)
    positive_proposals = proposals[labels > 0]
    mask_features = self.mask_roi_pool(features, positive_proposals, image_shapes)
    mask_features = self.mask_head(mask_features)  # [N_pos, 256, 14, 14]
    mask_logits = self.mask_predictor(mask_features)  # [N_pos, num_classes, 28, 28]
    
    # 4. Loss 계산
    losses = {}
    losses.update(fastrcnn_loss(class_logits, box_regression, labels, regression_targets))
    losses.update(maskrcnn_loss(mask_logits, positive_proposals, gt_masks, labels))
    
    return losses
```

**추론 모드 (Inference):**
```python
def forward(self, features, proposals, image_shapes, targets=None):
    # 1. Box 예측
    box_features = self.box_roi_pool(features, proposals, image_shapes)
    class_logits, box_regression = self.box_predictor(self.box_head(box_features))
    
    # 2. Post-processing (NMS, score filtering)
    boxes, scores, labels = self.postprocess_detections(
        class_logits, box_regression, proposals, image_shapes)
    
    # 3. Mask 예측 (고득점 박스만)
    mask_features = self.mask_roi_pool(features, boxes, image_shapes)
    mask_logits = self.mask_predictor(self.mask_head(mask_features))
    
    # 4. 최종 결과 구성
    detections = []
    for img_boxes, img_scores, img_labels, img_masks in zip(boxes, scores, labels, masks):
        detections.append({
            'boxes': img_boxes,
            'scores': img_scores, 
            'labels': img_labels,
            'masks': img_masks
        })
    
    return detections
```

## 3. get_maskrcnn_resnet50_fpn 분석

### 3.1. 함수의 역할

`get_maskrcnn_resnet50_fpn`은 **모델 아키텍처를 생성하는 빌더 함수**이며, 사전학습 가중치 사용 여부는 선택사항입니다.

```python
def maskrcnn_resnet50_fpn(
    weights: Optional[MaskRCNN_ResNet50_FPN_Weights] = None,
    num_classes: Optional[int] = None,
    weights_backbone: Optional[ResNet50_Weights] = ResNet50_Weights.IMAGENET1K_V1,
    trainable_backbone_layers: Optional[int] = None,
    **kwargs
) -> MaskRCNN:
    # weights가 지정되면 COCO 사전훈련 모델 (91 classes)
    # num_classes가 지정되면 해당 클래스 수로 head 교체
    pass
```

### 3.2. 핵심 구성 요소

#### 3.2.1. ResNet-50 백본

```python
# ResNet-50 구조 (총 50개 레이어)
ResNet50:
├── Stem: conv1 (7×7) + bn + relu + maxpool  
├── Stage1 (conv2_x): 3×(1×1→3×3→1×1) Bottleneck → C2 출력
├── Stage2 (conv3_x): 4×Bottleneck → C3 출력  
├── Stage3 (conv4_x): 6×Bottleneck → C4 출력
└── Stage4 (conv5_x): 3×Bottleneck → C5 출력
```

#### 3.2.2. FPN (Feature Pyramid Network)

```python
def build_fpn(backbone_features):
    # Top-down pathway
    P5 = conv1x1(C5)  # 256 채널로 통일
    P4 = conv1x1(C4) + F.interpolate(P5, scale_factor=2)
    P3 = conv1x1(C3) + F.interpolate(P4, scale_factor=2) 
    P2 = conv1x1(C2) + F.interpolate(P3, scale_factor=2)
    
    # Final 3×3 conv (알리아싱 제거)
    P5 = conv3x3(P5)
    P4 = conv3x3(P4)
    P3 = conv3x3(P3)
    P2 = conv3x3(P2)
    
    return {'0': P2, '1': P3, '2': P4, '3': P5}
```

### 3.3. 사용 패턴

#### 3.3.1. 완전한 사전학습 모델 (권장)
```python
model = maskrcnn_resnet50_fpn(weights="DEFAULT")
# ✅ COCO 91클래스로 완전 학습된 모델
```

#### 3.3.2. 클래스 수 변경
```python
model = maskrcnn_resnet50_fpn(weights="DEFAULT", num_classes=2)
# ✅ 백본+RPN은 사전학습, Box/Mask Head는 새로 초기화
```

## 4. 구현 시 필수 고려사항

### 4.1. 클래스 개수 설정 (중요)

```python
# 반드시 배경 클래스 포함
num_classes = 1 + 실제_클래스_수

# 예시
classes = ['person', 'car', 'bike']
num_classes = 1 + len(classes)  # 4 (배경 + 3개 클래스)
```

### 4.2. 헤드 교체 자동화

```python
def get_model(num_classes):
    model = maskrcnn_resnet50_fpn(weights="DEFAULT")
    
    # Box predictor 교체
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    # Mask predictor 교체
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask, 256, num_classes)
    
    return model
```

## 5. 데이터셋 준비

### 5.1. targets 딕셔너리 구조

```python
# 필수 구조
target = {
    'boxes': torch.tensor([[x1, y1, x2, y2]], dtype=torch.float32),  # [N, 4]
    'labels': torch.tensor([1], dtype=torch.int64),                 # [N] 1부터
    'masks': torch.tensor([mask], dtype=torch.uint8),               # [N, H, W]
}

# 선택 구조 (평가용)  
target.update({
    'image_id': torch.tensor([img_id]),
    'area': torch.tensor([box_area]),
    'iscrowd': torch.tensor([0]),
})
```

### 5.2. 간소화된 Dataset 클래스

```python
class SimpleDataset(torch.utils.data.Dataset):
    def __init__(self, image_paths, mask_paths, class_names):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.class_to_idx = {name: i+1 for i, name in enumerate(class_names)}
    
    def __getitem__(self, idx):
        # 이미지 로드
        image = Image.open(self.image_paths[idx]).convert('RGB')
        
        # 마스크 로드 및 처리
        mask = np.array(Image.open(self.mask_paths[idx]))
        
        # 인스턴스별 분리
        instances = np.unique(mask)[1:]  # 배경 0 제외
        
        masks, boxes, labels = [], [], []
        for inst_id in instances:
            inst_mask = (mask == inst_id).astype(np.uint8)
            pos = np.where(inst_mask)
            
            boxes.append([pos[1].min(), pos[0].min(), 
                         pos[1].max(), pos[0].max()])
            masks.append(inst_mask)
            labels.append(1)  # 단일 클래스 예시
        
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64), 
            'masks': torch.tensor(masks, dtype=torch.uint8)
        }
        
        return transforms.ToTensor()(image), target
```

## 6. 실무 체크리스트

### 6.1. 데이터 검증
- [ ] 이미지 범위: [0-1] 정규화 확인
- [ ] 레이블: 1부터 시작 (0은 배경)
- [ ] 마스크: torch.uint8 또는 bool
- [ ] 박스: x1<x2, y1<y2, 이미지 내부
- [ ] 빈 마스크 제거

### 6.2. 모델 설정  
- [ ] num_classes = 배경(1) + 실제 클래스
- [ ] 사전학습: weights="DEFAULT" 권장
- [ ] 헤드 교체: 클래스 변경 시 필수
- [ ] 훈련 레이어: trainable_backbone_layers 조정

### 6.3. 훈련
- [ ] 배치: 리스트로 전달 (torch.stack 금지)
- [ ] 모드: train()/eval() 전환
- [ ] 메모리: batch_size 1-4 권장
- [ ] 손실: 다중 손실 합계 사용

## 7. 추가 보완 내용

### 7.1. 손실 함수 이해

```python
# Mask R-CNN의 다중 손실
losses = {
    'loss_classifier': F.cross_entropy(class_logits, labels),
    'loss_box_reg': F.smooth_l1_loss(box_regression, targets), 
    'loss_mask': F.binary_cross_entropy_with_logits(mask_logits, gt_masks),
    'loss_rpn_box_reg': rpn_bbox_loss,
    'loss_objectness': rpn_objectness_loss
}

total_loss = sum(losses.values())
```

### 7.2. 추론 시 후처리

```python
def postprocess_predictions(predictions, score_thresh=0.5):
    results = []
    for pred in predictions:
        # Score filtering
        keep = pred['scores'] > score_thresh
        
        filtered_pred = {
            'boxes': pred['boxes'][keep],
            'scores': pred['scores'][keep],
            'labels': pred['labels'][keep], 
            'masks': pred['masks'][keep]
        }
        results.append(filtered_pred)
    
    return results
```

### 7.3. 메모리 최적화

```python
# Mixed Precision 훈련
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
model.train()

for images, targets in dataloader:
    optimizer.zero_grad()
    
    with autocast():
        losses = model(images, targets)
        loss = sum(losses.values())
    
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

### 7.4. 커스텀 백본 사용

```python
# 다른 백본 사용 예시
def get_custom_maskrcnn(backbone_name='resnet101'):
    if backbone_name == 'resnet101':
        backbone = resnet_fpn_backbone('resnet101', weights='DEFAULT')
    
    model = MaskRCNN(backbone, num_classes=num_classes)
    return model
```

## 8. 용어 정리

| 용어 | 영문 | 설명 |
|------|------|------|
| 관심 영역 | ROI (Region of Interest) | 객체가 있을 것으로 예상되는 사각형 영역 |
| 영역 제안 네트워크 | RPN (Region Proposal Network) | 객체가 있을 가능성이 높은 영역을 제안하는 네트워크 |
| 특징 피라미드 네트워크 | FPN (Feature Pyramid Network) | 다중 스케일 특징을 융합하는 넥 네트워크 |
| 관심 영역 정렬 | ROI Align | 양자화 없이 bilinear interpolation으로 특징을 추출하는 방법 |
| 인스턴스 분할 | Instance Segmentation | 개별 객체의 경계를 픽셀 단위로 구분하는 작업 |
| 마스크 분기 | Mask Branch | 픽셀별 마스크를 예측하는 네트워크 분기 |
| 다중 작업 학습 | Multi-task Learning | 분류, 회귀, 분할을 동시에 학습하는 방식 |
| 배경 억제 | NMS (Non-Maximum Suppression) | 중복된 탐지 결과를 제거하는 후처리 기법 |
| 양자화 | Quantization | 연속값을 이산값으로 변환 (정보 손실 발생) |
| 이중 선형 보간 | Bilinear Interpolation | 4개 주변 픽셀 값으로 중간값 계산하는 방법 |
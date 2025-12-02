---
layout: default
title: "스프린트미션15 Docker 기반 협업 워크플로우"
description: "스프린트미션15 Docker 기반 협업 워크플로우"
author: "김명환"
date: 2025-12-02
category: "스프린트미션"
cache-control: no-cache
expires: 0
pragma: no-cache
---

# Mission 15 - 연구자 1 (모델 학습)

## 개요
이 디렉토리는 **연구자 1**의 작업 환경입니다.
- 데이터 전처리
- 탐색적 데이터 분석(EDA)
- 회귀 모델링 (scikit-learn)
- 모델 파일 추출 (`model.pkl`)

## 디렉토리 구조
```
mission15_train/
├── Dockerfile              # 학습 환경 Docker 이미지 정의
├── requirements.txt        # Python 패키지 의존성
├── startup.sh              # 컨테이너 시작 스크립트
├── train_model.py          # 모델 학습 자동화 스크립트
├── notebook/
│   └── train_model.ipynb   # EDA 및 모델링 Jupyter Notebook
└── README.md               # 이 파일
```

## 환경 설정

### Python 버전 및 주요 패키지
- Python: 3.10
- pandas
- numpy
- scikit-learn
- joblib

## Docker 이미지 빌드

### 1. 이미지 빌드
```bash
cd mission15_train
docker build -t mission15-train-image:latest .
```

### 2. Docker Hub 업로드 (선택)
```bash
docker tag mission15-train-image:latest <your-dockerhub-username>/mission15-train-image:latest
docker push <your-dockerhub-username>/mission15-train-image:latest
```

## 사용 방법

### 전제 조건
- 학습 데이터 `mission15_train.csv`가 필요합니다.
- 데이터는 호스트의 `data/` 디렉토리에 위치해야 합니다.

### Docker 컨테이너 실행

#### 방법 1: 단독 실행 (볼륨 마운트)
```bash
# Windows PowerShell
docker run --rm --name mission15_train `
  -v ${PWD}/data:/app/data `
  mission15-train-image:latest

# Linux/Mac
docker run --rm --name mission15_train \
  -v $(pwd)/data:/app/data \
  mission15-train-image:latest
```

#### 방법 2: docker-compose 사용 (권장)
상위 디렉토리의 `docker-compose.yml`을 참조하세요.

```bash
docker-compose up train
```

### 출력 파일
- **model.pkl**: 학습된 모델 파일
  - 저장 위치: `/app/data/model.pkl` (컨테이너 내부)
  - 볼륨 마운트를 통해 호스트의 `data/model.pkl`로 접근 가능

## 주요 파일 설명

### train_model.py
- 데이터 전처리 및 모델 학습 자동화 스크립트
- RMSE 기반 모델 성능 평가
- 학습된 모델을 `model.pkl`로 저장

### notebook/train_model.ipynb
- Jupyter Notebook을 통한 대화형 EDA 및 모델링
- 데이터 시각화 및 분석 과정 포함

### startup.sh
- 컨테이너 시작 시 안내 메시지 출력
- 볼륨 마운트 누락 시 경고 표시

## 문제 해결

### 데이터 파일을 찾을 수 없는 경우
```
ERROR - 오류: 훈련 데이터 파일 '/app/data/mission15_train.csv'을 찾을 수 없습니다.
```
**해결 방법**:
1. 호스트의 `data/` 디렉토리에 `mission15_train.csv` 파일이 있는지 확인
2. 볼륨 마운트 옵션 `-v` 올바르게 지정했는지 확인

### 권한 문제
- Windows: Docker Desktop이 해당 드라이브에 접근 권한이 있는지 확인
- Linux: 파일 소유권 및 권한 확인 (`chmod`, `chown`)

## 연구자 2와의 협업
1. 이 이미지를 Docker Hub에 업로드
2. 연구자 2는 `docker-compose.yml`에서 이 이미지를 참조
3. 공유 볼륨(`data/`)을 통해 `model.pkl` 전달

## 참고 사항
- 컨테이너는 학습 완료 후 자동으로 종료됩니다 (`--rm` 옵션)
- 로그는 표준 출력(stdout)으로 출력됩니다
- 모델 재학습 시 기존 `model.pkl`이 덮어씌워집니다

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

# Mission 15 - 연구자 2 (모델 추론)

## 개요
이 디렉토리는 **연구자 2**의 작업 환경입니다.
- 연구자 1이 생성한 모델(`model.pkl`) 활용
- 테스트 데이터(`mission15_test.csv`)를 활용한 추론
- 추론 결과를 `result.csv`로 저장
- Jupyter Notebook 환경에서 작업

## 디렉토리 구조
```
mission15_inference/
├── Dockerfile              # Jupyter 환경 Docker 이미지 정의
├── startup.sh              # 컨테이너 시작 스크립트
├── notebook/
│   └── inference.ipynb     # 추론 작업 Jupyter Notebook
└── README.md               # 이 파일
```

## 환경 설정

### Python 버전 및 주요 패키지
- Python: 3.10 (연구자 1과 동일)
- pandas
- numpy
- scikit-learn
- joblib
- Jupyter Notebook
- scipy

### 기본 이미지
- `jupyter/scipy-notebook:latest`
  - 데이터 분석에 필요한 대부분의 라이브러리 포함
  - Jupyter Notebook 환경 제공

## Docker Compose 사용 (권장)

### docker-compose.yml 구성 예시
```yaml
version: '3.8'

services:
  train:
    image: <your-dockerhub-username>/mission15-train-image:latest
    container_name: mission15_train
    volumes:
      - ./data:/app/data

  inference:
    image: mission15-inference-image:latest
    container_name: mission15_inference
    ports:
      - "8888:8888"
    volumes:
      - ./data:/home/jovyan/workspace/data
    depends_on:
      - train
```

### 실행 순서
```bash
# 1. 연구자 1의 모델 학습 실행
docker-compose up train

# 2. 연구자 2의 Jupyter Notebook 환경 실행
docker-compose up -d inference

# 3. Jupyter Notebook 접속
# 브라우저에서 http://localhost:8888 접속
# 토큰은 docker logs에서 확인
docker logs mission15_inference
```

## 사용 방법

### 1. Docker 이미지 빌드
```bash
cd mission15_inference
docker build -t mission15-inference-image:latest .
```

### 2. 컨테이너 실행

#### 방법 1: 단독 실행
```bash
# Windows PowerShell
docker run --rm --name mission15_inference `
  -p 8888:8888 `
  -v ${PWD}/data:/home/jovyan/workspace/data `
  mission15-inference-image:latest

# Linux/Mac
docker run --rm --name mission15_inference \
  -p 8888:8888 \
  -v $(pwd)/data:/home/jovyan/workspace/data \
  mission15-inference-image:latest
```

#### 방법 2: docker-compose 사용 (권장)
```bash
docker-compose up -d inference
```

### 3. Jupyter Notebook 접속
1. 브라우저에서 `http://localhost:8888` 접속
2. 토큰 확인:
   ```bash
   docker logs mission15_inference
   ```
3. `inference.ipynb` 노트북 열기
4. 셀 실행하여 추론 수행

### 4. 추론 작업 흐름
1. **모델 로드**: 공유 볼륨에서 `model.pkl` 읽기
2. **데이터 로드**: `mission15_test.csv` 읽기
3. **추론 수행**: 로드한 모델로 예측
4. **결과 저장**: `result.csv`로 저장

## 파일 공유 전략

### 볼륨 공유 방식
```
호스트 (data/)
    ├── mission15_train.csv  (입력)
    ├── mission15_test.csv   (입력)
    ├── model.pkl            (연구자 1 → 연구자 2)
    └── result.csv           (연구자 2 출력)
        ↓ 마운트
연구자 1 컨테이너: /app/data/
연구자 2 컨테이너: /home/jovyan/workspace/data/
```

### docker cp 사용 (대안)
연구자 1의 컨테이너가 실행 중일 때:
```bash
# 연구자 1 컨테이너에서 모델 파일 복사
docker cp mission15_train:/app/data/model.pkl ./data/

# 연구자 2 컨테이너로 파일 복사
docker cp ./data/model.pkl mission15_inference:/home/jovyan/workspace/data/
```

## 출력 파일

### result.csv
- 추론 결과가 저장된 CSV 파일
- 저장 위치: `/home/jovyan/workspace/data/result.csv` (컨테이너 내부)
- 볼륨 마운트를 통해 호스트의 `data/result.csv`로 접근 가능

### inference.ipynb
- 추론 과정이 담긴 Jupyter Notebook
- 데이터 로드, 모델 로드, 예측, 결과 저장 전 과정 포함

## Jupyter Notebook 토큰 확인

### 방법 1: 로그에서 확인
```bash
docker logs mission15_inference | grep token
```

### 방법 2: 컨테이너 내부에서 확인
```bash
docker exec mission15_inference jupyter server list
```

### 방법 3: 토큰 없이 접속 (개발 환경)
Dockerfile에서 아래 설정 추가:
```dockerfile
CMD ["start-notebook.sh", "--NotebookApp.token=''", "--NotebookApp.password=''"]
```

## 문제 해결

### 모델 파일을 찾을 수 없는 경우
**증상**:
```
FileNotFoundError: [Errno 2] No such file or directory: '/home/jovyan/workspace/data/model.pkl'
```

**해결 방법**:
1. 연구자 1의 컨테이너가 정상적으로 실행되었는지 확인
2. 공유 볼륨에 `model.pkl` 파일이 있는지 확인
   ```bash
   ls -la data/
   ```
3. 볼륨 마운트 경로가 올바른지 확인

### Jupyter Notebook에 접속할 수 없는 경우
**해결 방법**:
1. 포트 매핑 확인: `-p 8888:8888`
2. 컨테이너 실행 상태 확인:
   ```bash
   docker ps | grep mission15_inference
   ```
3. 방화벽 설정 확인

### 권한 문제
**증상**: 파일 저장 시 Permission Denied

**해결 방법**:
- Jupyter 이미지는 `jovyan` 사용자로 실행됨
- Dockerfile에서 `--chown=jovyan:users` 옵션 사용
- 호스트 디렉토리 권한 확인

## 연구자 1과의 협업 체크리스트

- [x] 연구자 1의 Docker Hub 이미지 URL 확인
- [x] `docker-compose.yml` 작성
- [x] 공유 볼륨 경로 설정
- [x] Python 및 패키지 버전 일치 확인
- [x] `mission15_test.csv` 데이터 준비
- [x] 연구자 1 컨테이너 실행 및 `model.pkl` 생성 확인
- [x] Jupyter Notebook 접속 확인
- [x] 추론 실행 및 `result.csv` 생성 확인

## 참고 사항
- Jupyter Notebook 컨테이너는 백그라운드로 실행됩니다 (`-d` 옵션)
- 작업 완료 후 `docker-compose down`으로 정리
- 데이터와 결과는 공유 볼륨에 저장되어 호스트에 남습니다
- 보안상 프로덕션 환경에서는 토큰/암호 설정 필수

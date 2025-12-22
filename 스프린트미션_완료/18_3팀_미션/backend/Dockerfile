# Backend Dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 시스템 패키지 업데이트 및 필요한 도구 설치
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# AI 모델 사전 다운로드 (서버 시작 시간 단축)
RUN python -c "from transformers import pipeline; pipeline('text-classification', model='daekeun-ml/koelectra-small-v3-nsmc')"

# 애플리케이션 코드 복사
COPY ./app ./app
COPY ./config ./config
COPY ./data ./data

# data 폴더 생성 (포스터 이미지 저장용)
RUN mkdir -p data/posters

# 포트 설정 (환경변수로 유연하게 설정 가능)
ENV PORT=8000
EXPOSE ${PORT}

# 애플리케이션 실행
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT}

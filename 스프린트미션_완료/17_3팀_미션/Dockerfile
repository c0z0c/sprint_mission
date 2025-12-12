FROM python:3.11-slim

WORKDIR /app

# 시스템 라이브러리 설치 (cv2/OpenGL 의존성 - headless 버전)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY README.md .
COPY app.py .
COPY doc/ ./doc/
COPY src/ ./src/
COPY models/ ./models/

# Python 모듈 경로 설정
ENV PYTHONPATH=/app

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]

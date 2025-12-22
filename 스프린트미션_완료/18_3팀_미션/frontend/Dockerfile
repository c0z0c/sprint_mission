# Frontend Dockerfile
FROM python:3.11-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사 및 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 포트 설정 (환경변수로 유연하게 설정 가능)
ENV PORT=8501
ENV API_BASE_URL=http://mis18_backend:8000
ENV BROWSER_API_URL=http://localhost:8000
EXPOSE ${PORT}

# Streamlit 설정
RUN mkdir -p ~/.streamlit && \
    echo "[server]" > ~/.streamlit/config.toml && \
    echo "port = ${PORT}" >> ~/.streamlit/config.toml && \
    echo "address = \"0.0.0.0\"" >> ~/.streamlit/config.toml && \
    echo "headless = true" >> ~/.streamlit/config.toml && \
    echo "enableCORS = false" >> ~/.streamlit/config.toml && \
    echo "enableXsrfProtection = false" >> ~/.streamlit/config.toml

# 애플리케이션 실행
CMD streamlit run app.py --server.port=${PORT} --server.address=0.0.0.0

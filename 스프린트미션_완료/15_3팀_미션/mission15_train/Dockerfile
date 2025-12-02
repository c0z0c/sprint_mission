# researcher1_training/Dockerfile

# 1. 기본 이미지 지정 (Python 환경)
FROM python:3.10-slim

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. 필요한 Python 패키지 설치
# scikit-learn, pandas, numpy, joblib 등 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# --- requirements.txt 내용 ---
# pandas
# numpy
# scikit-learn
# joblib
# -----------------------------

# 4. startup.sh 스크립트 복사 및 실행 권한 부여
COPY startup.sh /usr/local/bin/startup.sh
RUN chmod +x /usr/local/bin/startup.sh


# 5. 스크립트 및 데이터 복사
COPY train_model.py .
#COPY ../data/mission15_train.csv /app/data/mission15_train.csv
#COPY ../data/mission15_test.csv /app/data/mission15_test.csv

# 6. 환경 변수 설정
ENV RUNNING_IN_DOCKER=True

# 7. ENTRYPOINT 설정 (컨테이너 시작 시 안내 스크립트 실행)
ENTRYPOINT ["/usr/local/bin/startup.sh"]

# 8. CMD 설정 (startup.sh가 실행할 실제 명령)
CMD ["python", "train_model.py"]
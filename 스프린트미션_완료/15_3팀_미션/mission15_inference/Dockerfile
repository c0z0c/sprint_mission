# researcher2_inference/Dockerfile

# 1. 기본 이미지 지정 (Python 환경, 또는 Jupyter 공식 이미지 사용 가능)
# 데이터 분석에 필요한 라이브러리가 포함된 이미지 사용
FROM jupyter/scipy-notebook:latest

# 2. 작업 디렉토리 설정 (기본적으로 /home/jovyan으로 설정되어 있음)
WORKDIR /home/jovyan/workspace

# 3. requirements.txt 복사 및 패키지 설치
COPY --chown=jovyan:users requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. startup.sh 스크립트 복사 및 소유권 변경
COPY --chown=jovyan:users startup.sh /home/jovyan/
RUN chmod +x /home/jovyan/startup.sh

# 5. 추론 노트북 복사
COPY --chown=jovyan:users ./notebook/inference.ipynb .

# 6. 환경 변수 설정
ENV RUNNING_IN_DOCKER=True

ENTRYPOINT ["/home/jovyan/startup.sh"]

# 7. 포트 노출
EXPOSE 8888


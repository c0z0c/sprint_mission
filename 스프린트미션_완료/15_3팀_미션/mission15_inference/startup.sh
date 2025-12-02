#!/bin/bash
set -euo pipefail

# 1. 안내 문구 출력
echo "-----------------------------------------------------------------------------------"
echo "  [스프린트 미션 15 모델 학습 이미지 안내]"
echo "-----------------------------------------------------------------------------------"
echo "  1. 모델 파일 (model.pkl)은 컨테이너 종료 후 /app/data/ 에 저장됩니다."
echo "  2. Jupyter Notebook 접속: http://localhost:8888 (연구자 2 컨테이너)"
echo "  3. 주의: 학습 데이터 mission15_test.csv 파일은 /app/data 경로에 존재해야 합니다."
echo "  4. ex) 데이터 파일이 없으면 학습이 실패할 수 있습니다."
echo "      docker run -p 8888:8888 --rm --name mission15_inference -v \${PWD}/data:/app/data mission15_inference-image"
echo "  5. docker exec -it mission15_inference sh"
echo "-----------------------------------------------------------------------------------"

if [ ! -d "/app/data" ]; then
  echo "[Error] /app/data 디렉토리가 존재하지 않습니다. 호스트의 데이터 디렉토리를 마운트해 주세요."
  exit 1
fi

# 10초 동안 /app/data/model.pkl 존재 여부 확인 (타임아웃 10초)
timeout_secs=10
interval_secs=1
elapsed=0

echo "Checking for /app/data/model.pkl (timeout=${timeout_secs}s)..."
while [ "$elapsed" -lt "$timeout_secs" ]; do
  if [ -f "/app/data/model.pkl" ]; then
    echo "/app/data/model.pkl found."
    break
  fi
  echo "[Info] /app/data/model.pkl을 찾는 중... (${elapsed}s 경과)" >&2
  sleep "$interval_secs"
  elapsed=$((elapsed + interval_secs))
done

if [ ! -f "/app/data/model.pkl" ]; then
  echo "[Error] /app/data/model.pkl을 ${timeout_secs}초 내에 찾지 못했습니다. 모델 파일을 확인하세요." >&2
  exit 1
fi

if [ "$#" -gt 0 ] && { [ "$1" = "start-notebook.sh" ] || [ "$1" = "./start-notebook.sh" ] || [ "$1" = "/home/jovyan/start-notebook.sh" ]; }; then
  shift
fi

exec /usr/local/bin/start-notebook.sh "$@"

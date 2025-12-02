#!/bin/sh

# 1. 안내 문구 출력
echo "-----------------------------------------------------------------------------------"
echo "  [스프린트 미션 15 모델 학습 이미지 안내]"
echo "-----------------------------------------------------------------------------------"
echo "  1. 모델 파일 (model.pkl)은 컨테이너 종료 후 /app/data/ 에 저장됩니다."
echo "  2. Jupyter Notebook 접속: http://localhost:8888 (연구자 2 컨테이너)"
echo "  3. 주의: 학습 데이터 mission15_train.csv 파일은 /app/data 경로에 존재해야 합니다."
echo "  4. ex) 데이터 파일이 없으면 학습이 실패할 수 있습니다."
echo "      docker run --rm --name mission15_train_test -v \${PWD}/data:/app/data mission15_train-image python train_model.py"
echo "-----------------------------------------------------------------------------------"

exec "$@"